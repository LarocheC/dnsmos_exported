#!/usr/bin/env python3
"""Export the ConvFSENet on-device-training artifacts.

Produces the four ONNX graphs the STM32N6 would load (the fifth, the DNSMOS
loss graph, comes from `scripts/export_all_fp32.py` / `export_int8_loss.py` in
this repo):

    convfsenet_trunk_fp32.onnx        mag window [1,F,L+T] -> h [1,192,T]
    convfsenet_trunk_int8.onnx        same, QDQ int8 (Neural-ART NPU)
    convfsenet_head_fwd.onnx          (h, W, b)             -> mask [1,F,T]
    convfsenet_head_bwd.onnx          (dmask, mask, h)      -> (dW [F,192], db [F])

Export + quantization follow the upstream eco8-neaixt contract exactly:
opset 17, dynamo=False, static shapes except batch, QDQ with SIGNED int8
activations (the Neural-ART rejects unsigned), per-channel weights, MinMax
calibration, asymmetric activations / symmetric weights, and — the load-bearing
one — the magnitude-compression prologue ((|m|+1e-9)**0.3) is EXCLUDED from
quantization so the raw |STFT| never lands on a coarse int8 grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import dsp  # noqa: E402
from convfsenet_arch import build_split, load_from_eco8_checkpoint  # noqa: E402
from head import HeadBackward, HeadForward  # noqa: E402

OPSET = 17  # upstream eco8-neaixt convention (ST Edge AI accepts <= 20)


def _synthetic_audio(n: int, seed: int, length: int) -> np.ndarray:
    """Speech-like calibration signals (harmonic stack + shaped noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length) / 16000.0
    out = np.zeros((n, length), dtype=np.float32)
    for i in range(n):
        f0 = rng.uniform(90, 260)
        sig = sum(rng.uniform(0.1, 1.0) / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 6))
        sig *= 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * rng.uniform(2, 5) * t), 0, None)
        sig = sig / (np.abs(sig).max() + 1e-9) * rng.uniform(0.1, 0.5)
        noise = rng.standard_normal(length)
        sig += noise / (np.abs(noise).max() + 1e-9) * rng.uniform(0.02, 0.3)
        out[i] = sig
    return out


def magnitude_windows(audio: np.ndarray, n_features: int, context_l: int, emit_t: int):
    """Yield [1, F, L+T] magnitude windows exactly as the host would build them."""
    for i in range(audio.shape[0]):
        X = dsp.stft(audio[i])
        mag = np.abs(X)[:n_features]                       # drop_nyquist keeps bins 0..255
        padded = np.concatenate([np.repeat(mag[:, :1], context_l, axis=1), mag], axis=1)
        n_win = max(1, (mag.shape[1] + emit_t - 1) // emit_t)
        for wnd in range(n_win):
            s = wnd * emit_t
            block = padded[:, s: s + context_l + emit_t]
            if block.shape[1] < context_l + emit_t:        # pad the tail window
                block = np.pad(block, ((0, 0), (0, context_l + emit_t - block.shape[1])))
            yield block[None].astype(np.float32)


def export_trunk_fp32(trunk, out_path: Path, emit_t: int) -> Path:
    example = torch.zeros(1, trunk.n_features, trunk.context_l + emit_t)
    torch.onnx.export(
        trunk, (example,), str(out_path),
        input_names=["noisy_mag_window"], output_names=["h"],
        dynamic_axes={"noisy_mag_window": {0: "B"}, "h": {0: "B"}},
        opset_version=OPSET, dynamo=False,
    )
    onnx.checker.check_model(str(out_path))
    return out_path


def _compression_prologue_nodes(model_path: Path) -> list[str]:
    """Nodes between the magnitude input and the first Conv — must stay FP32.

    Upstream's finding: ORT will happily quantize the eps-Add, which drags the
    raw |STFT| onto a coarse int8 grid and destroys exactly the low-energy
    detail the compression was added to preserve.
    """
    model = onnx.load(str(model_path))
    graph_input = model.graph.input[0].name
    producers = {out: n for n in model.graph.node for out in n.output}
    frontier = {graph_input}
    prologue: list[str] = []
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if node.name in prologue or node.op_type == "Conv":
                continue
            if any(i in frontier for i in node.input):
                prologue.append(node.name)
                frontier.update(node.output)
                changed = True
    return prologue


def quantize_trunk(fp32_path: Path, out_path: Path, calib_feeds) -> Path:
    """Static QDQ int8 per the eco8-neaixt / Neural-ART recipe."""
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = out_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
    exclude = _compression_prologue_nodes(pre)

    class _Reader(CalibrationDataReader):
        def __init__(self, feeds):
            self._it = iter(feeds)

        def get_next(self):
            return next(self._it, None)

    quantize_static(
        str(pre), str(out_path), _Reader(calib_feeds),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,      # SIGNED is mandatory on the Neural-ART
        weight_type=QuantType.QInt8,
        per_channel=True,                     # essential for the depthwise dconv
        nodes_to_exclude=exclude,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    pre.unlink(missing_ok=True)
    onnx.checker.check_model(str(out_path))
    print(f"  excluded {len(exclude)} compression-prologue nodes from int8: {exclude}")
    return out_path


def export_head_graphs(n_features: int, head_t: int, out_dir: Path):
    """Head graphs span the whole adaptation window (head_t frames), not the
    trunk's emit_T chunk: the trunk streams, but one weight update consumes a
    full DNSMOS segment, so W/b gradients are accumulated over all of it."""
    F, C, T = n_features, 192, head_t
    h = torch.zeros(1, C, T)
    W = torch.zeros(F, C)
    b = torch.zeros(F)
    fwd = out_dir / "convfsenet_head_fwd.onnx"
    torch.onnx.export(
        HeadForward(), (h, W, b), str(fwd),
        input_names=["h", "W", "b"], output_names=["mask"],
        opset_version=OPSET, dynamo=False,
    )
    onnx.checker.check_model(str(fwd))

    dmask = torch.zeros(1, F, T)
    mask = torch.zeros(1, F, T)
    bwd = out_dir / "convfsenet_head_bwd.onnx"
    torch.onnx.export(
        HeadBackward(), (dmask, mask, h), str(bwd),
        input_names=["dmask", "mask", "h"], output_names=["dW", "db"],
        opset_version=OPSET, dynamo=False,
    )
    onnx.checker.check_model(str(bwd))
    return fwd, bwd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    p.add_argument("--emit-t", type=int, default=64,
                   help="mask columns per trunk call (upstream deploys 1; 64 keeps the demo fast)")
    p.add_argument("--head-t", type=int, default=dsp.num_frames(144160),
                   help="frames per adaptation window for the head graphs (default 564 = 9.01 s)")
    p.add_argument("--n-features", type=int, default=256,
                   help="256 = the deployed drop_nyquist graph")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="eco8-neaixt g_best checkpoint (real weights); random init otherwise")
    p.add_argument("--eco8-repo", type=Path, default=None)
    p.add_argument("--calib-utts", type=int, default=6)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        trunk, head = load_from_eco8_checkpoint(args.checkpoint, args.n_features, args.eco8_repo)
        print(f"loaded real ConvFSENet weights from {args.checkpoint}")
    else:
        trunk, head = build_split(args.n_features, seed=0)
        print("no --checkpoint: random-init trunk (mechanism demo)")
    torch.save({"trunk": trunk.state_dict(), "head": head.state_dict()},
               args.out_dir / "convfsenet_split.pt")

    fp32 = export_trunk_fp32(trunk, args.out_dir / "convfsenet_trunk_fp32.onnx", args.emit_t)
    print(f"wrote {fp32.name}")

    audio = _synthetic_audio(args.calib_utts, seed=100, length=dsp.HOP * 200)
    feeds = [{"noisy_mag_window": w} for w in
             magnitude_windows(audio, args.n_features, trunk.context_l, args.emit_t)]
    print(f"calibrating trunk int8 on {len(feeds)} windows")
    int8 = quantize_trunk(fp32, args.out_dir / "convfsenet_trunk_int8.onnx", feeds)
    print(f"wrote {int8.name}  ({int8.stat().st_size/1e6:.2f} MB vs fp32 {fp32.stat().st_size/1e6:.2f} MB)")

    fwd, bwd = export_head_graphs(args.n_features, args.head_t, args.out_dir)
    print(f"wrote {fwd.name}, {bwd.name}  (head window T={args.head_t})")

    from dnsmos_trainable.verify import check_device_constraints, software_epoch_ops
    # The trunk keeps a dynamic batch axis (pinned by ST's
    # --fix-parametric-shapes "{'B':1}"); the head graphs take weight matrices
    # as runtime inputs, whose leading dim is F, not a batch.
    checks = {
        fp32: dict(allow_dynamic_batch=True, batched_io={"noisy_mag_window", "h"}),
        int8: dict(allow_dynamic_batch=True, batched_io={"noisy_mag_window", "h"}),
        fwd: dict(batched_io={"h", "mask"}),
        bwd: dict(batched_io={"dmask", "mask", "h"}),
    }
    print("\nSTM32N6 constraint lint (ST Edge AI Core front end):")
    for art, kw in checks.items():
        problems = check_device_constraints(art, max_opset=OPSET, **kw)
        sw = software_epoch_ops(art)
        note = f"  [M55 software epochs: {sorted(sw)}]" if sw else ""
        print(f"  {art.name}: {'OK' if not problems else problems}{note}")


if __name__ == "__main__":
    main()
