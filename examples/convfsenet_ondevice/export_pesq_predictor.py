#!/usr/bin/env python3
"""Export the PESQ predictor for the STM32N6 and check the ST defect is avoided.

The DNSMOS loss graph fails on-target inside ST's int8<->float *software*-epoch
boundary: it hangs there at a 1 s window and returns input-independent garbage
at 0.25 s (see ST_BUG_REPORT.md). That boundary exists because DNSMOS's
hand-written backward carries a large fp32 elementwise mask region
(`Equal`/`Cast`/`Sub`/`Mul`) between int8 convolution blocks.

This model has no such region. It is a plain CNN — Conv2d, InstanceNorm,
PReLU, global max-pool, two linear layers — so if the defect really is
localized to that boundary, this graph should compile and run correctly on the
same silicon that mis-executes the other one. That makes it a direct test of
the diagnosis as well as a deployment step.

It is also the metric the project can own outright: 181,650 parameters trained
in-house on VoiceBank-DEMAND with real PESQ labels, versus a transplanted
third-party model.

Two export details matter:

* **`spectral_norm` must be removed, not just eval()'d.** It is a
  parametrization: the effective weight is recomputed from stored `u`/`v`
  vectors. Exporting with it live either bakes a stale weight or emits the
  power-iteration ops. `remove_parametrizations` folds the normalized weight
  into the tensor once, which is what inference wants.
* **`Flatten` is replaced by `Reshape`** — the former is not in the device op
  vocabulary, and after a global max-pool to [B, C, 1, 1] the two are
  identical.

    python export_pesq_predictor.py --frames 63
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from pesq_predictor import PesqPredictor, N_BINS  # noqa: E402

OPSET = 13


def strip_spectral_norm(model: nn.Module) -> int:
    """Fold spectral_norm into a plain weight tensor.

    Two APIs exist and this model uses the OLD hook-based one, where the
    effective weight is recomputed by a forward-pre-hook from stored `u`/`v`
    vectors. `.eval()` does not disable it, and exporting with it live traces
    the power iteration itself into the graph — 12 MatMuls and 6 Divs of pure
    normalization arithmetic that the device would then have to execute.
    `remove_spectral_norm` folds the normalized weight in once.
    """
    from torch.nn.utils import parametrize, remove_spectral_norm

    n = 0
    for mod in model.modules():
        if parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
            n += 1
        elif any(k.startswith("weight_") for k in mod._buffers) or hasattr(mod, "weight_v"):
            try:
                remove_spectral_norm(mod)
                n += 1
            except (ValueError, RuntimeError):
                pass
    return n


class PesqForONNX(nn.Module):
    """Two-input wrapper: (noisy_mag, enh_mag) -> normalized PESQ scalar.

    Keeps the channel stack inside the graph so the device sees exactly the
    two magnitude planes the host already has, with no host-side packing.
    """

    def __init__(self, core: PesqPredictor) -> None:
        super().__init__()
        self.layers = core.layers

    def forward(self, noisy_mag: torch.Tensor, enh_mag: torch.Tensor) -> torch.Tensor:
        x = torch.stack((noisy_mag, enh_mag), dim=1)
        for m in self.layers:
            x = m(x) if not isinstance(m, nn.Flatten) else x.reshape(x.shape[0], -1)
        return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=HERE / "artifacts" / "pesq_predictor.pt")
    ap.add_argument("--frames", type=int, default=63,
                    help="time frames per call (63 = the 1 s adaptation window)")
    ap.add_argument("--bins", type=int, default=N_BINS)
    ap.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    ap.add_argument("--data", type=Path, default=HERE / "artifacts" / "pesq_dataset.npz")
    ap.add_argument("--calib", type=int, default=64)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    core = PesqPredictor(ck["dim"])
    core.load_state_dict(ck["model"])
    core.eval()
    print(f"removed {strip_spectral_norm(core)} spectral_norm parametrizations")

    model = PesqForONNX(core).eval()
    F_, T = args.bins, args.frames
    ex = (torch.zeros(1, F_, T), torch.zeros(1, F_, T))
    fp32 = args.out_dir / "pesq_predictor_fp32.onnx"
    torch.onnx.export(model, ex, str(fp32),
                      input_names=["noisy_mag", "enh_mag"], output_names=["pesq_norm"],
                      opset_version=OPSET, dynamo=False)
    onnx.checker.check_model(str(fp32))
    m = onnx.load(str(fp32))
    from collections import Counter
    print(f"wrote {fp32.name}  ops: {dict(Counter(n.op_type for n in m.graph.node))}")

    # ---- host parity: exported graph vs the torch model ----
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(fp32), so, providers=["CPUExecutionProvider"])
    z = np.load(args.data)
    feats = z["feats"][: args.calib, :, :, :T].astype(np.float32)
    worst = 0.0
    with torch.no_grad():
        for k in range(min(8, len(feats))):
            a = float(core(torch.from_numpy(feats[k, 0])[None],
                           torch.from_numpy(feats[k, 1])[None])[0])
            b = float(sess.run(None, {"noisy_mag": feats[k, 0][None],
                                      "enh_mag": feats[k, 1][None]})[0].ravel()[0])
            worst = max(worst, abs(a - b))
    print(f"torch-vs-ONNX parity: max|diff| {worst:.2e}")

    # ---- int8 QDQ ----
    from dnsmos_trainable.export import quantize_qdq
    calib = [{"noisy_mag": feats[k, 0][None], "enh_mag": feats[k, 1][None]}
             for k in range(len(feats))]
    int8 = quantize_qdq(fp32, args.out_dir / "pesq_predictor_int8.onnx", calib)
    s8 = ort.InferenceSession(str(int8), so, providers=["CPUExecutionProvider"])
    errs = []
    for k in range(min(16, len(feats))):
        a = sess.run(None, {"noisy_mag": feats[k, 0][None], "enh_mag": feats[k, 1][None]})[0].ravel()[0]
        b = s8.run(None, {"noisy_mag": feats[k, 0][None], "enh_mag": feats[k, 1][None]})[0].ravel()[0]
        errs.append(abs(float(a) - float(b)) * 3.5)          # in PESQ units
    print(f"wrote {int8.name} ({int8.stat().st_size/1e3:.0f} kB)  "
          f"int8-vs-fp32 mean |d| {np.mean(errs):.3f} PESQ, max {np.max(errs):.3f}")

    from dnsmos_trainable.verify import check_device_constraints, software_epoch_ops
    for art in (fp32, int8):
        pr = check_device_constraints(art, max_opset=OPSET,
                                      batched_io={"noisy_mag", "enh_mag", "pesq_norm"})
        sw = software_epoch_ops(art)
        print(f"  {art.name}: lint {'OK' if not pr else pr}"
              f"{'  [M55 SW: ' + str(sorted(sw)) + ']' if sw else ''}")


if __name__ == "__main__":
    main()
