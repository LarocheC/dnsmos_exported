#!/usr/bin/env python3
"""Build the int8 QDQ loss-graph artifacts (fwd + hand-written bwd quantized).

Quantizes Conv/Gemm/MatMul in the device-mode loss graph. Calibration runs
full (wav, w) pairs so activation AND gradient tensor ranges are observed.
MinMax calibration is mandatory: percentile clipping saturates activation
ceilings, and in a max-pooling network the clipped maxima collapse into ties
at the clip value, diffusing the routing masks.

Gates — measured against the fp32 reference (original transplanted weights):

- Functional (hard): optimizing a waveform with the int8 graph's gradients
  for 60 steps must raise the fp32-reference OVRL by >= +0.3 (fp32 gradients
  achieve ~+2.3; int8 gradients are noisier but must remain useful descent
  directions).
- Cosine tripwire (hard): mean cosine(grad_int8, grad_fp32) >= 0.4.
- Reported (not gated): full cosine/norm-ratio stats. Note that a per-segment
  cosine near 1.0 vs the fp32 gradient is PHYSICALLY UNATTAINABLE for an
  int8-quantized forward: quantization changes the function, and even
  PyTorch's own straight-through-estimator gradient of a fake-quant model
  sits at cosine ~0.55-1.0 (mean ~0.77) against the fp32 gradient. Consumers
  needing maximum gradient fidelity should use the fp32 loss graph.

An exclusion ladder walks backward-conv exclusions (deepest first) only if
the fully-quantized graph fails the gates.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dnsmos_trainable import load_transplanted
from dnsmos_trainable.backward import DnsmosLossGraph
from dnsmos_trainable.export import export_loss_graph, make_calibration_batches, quantize_qdq
from dnsmos_trainable.verify import (
    check_device_constraints,
    int8_grad_report,
    make_synthetic_batch,
    ort_session,
)

W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def backward_conv_names(pre_path: Path) -> list[str]:
    """Backward convs of a PREPROCESSED loss graph, deepest (conv1-grad) first.

    Node names must come from the preprocessed graph — quant_pre_process's
    optimization pass renames nodes, so names from the raw export would
    silently exclude nothing.
    """
    m = onnx.load(str(pre_path))
    names = [n.name for n in m.graph.node if n.op_type == "Conv"]
    if len(names) != 14:
        raise RuntimeError(f"expected 14 convs (7 fwd + 7 bwd), got {len(names)}")
    return list(reversed(names[7:]))


def functional_gate(int8_loss_path: Path, fp32_fwd_path: Path, steps: int = 60) -> float:
    """OVRL improvement (per fp32 reference) from optimizing with int8 grads.

    Adapts to the loss artifact's layout (flat [1,144160] or rows [1,901,160])
    so the STM32N6 artifact is gated with the same procedure as the desktop one.
    """
    sess_g = ort_session(int8_loss_path)
    sess_ref = ort_session(fp32_fwd_path)
    # Symbolic dims (dynamic batch) come back as strings; we optimize B=1.
    g_shape = tuple(d if isinstance(d, int) else 1 for d in sess_g.get_inputs()[0].shape)
    wav0 = make_synthetic_batch(1, seed=7)
    x = wav0.copy().astype(np.float64)
    m, v, lr, b1, b2 = 0.0, 0.0, 3e-4, 0.9, 0.999

    def ref_ovrl(arr):
        return float(sess_ref.run(None, {"wav": arr.astype(np.float32)})[1][0, 2])

    start = ref_ovrl(x)
    for t in range(1, steps + 1):
        feed = x.astype(np.float32).reshape(g_shape)
        g = sess_g.run(None, {"wav": feed, "w": W})[2].astype(np.float64).reshape(x.shape)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + 1e-8)
    return ref_ovrl(x) - start


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--calib-segments", type=int, default=8)
    parser.add_argument("--eval-segments", type=int, default=30)
    args = parser.parse_args()

    art = root / "artifacts"
    weights = args.weights
    if weights is None:
        qat = root / "models" / "dnsmos_qat_int8.pt"
        weights = qat if qat.exists() else root / "models" / "dnsmos_transplanted.pt"
    print(f"int8 loss source weights: {weights.name}")
    model = load_transplanted(weights)

    scratch = art / "_int8_src"
    scratch.mkdir(parents=True, exist_ok=True)
    src_flat = export_loss_graph(
        DnsmosLossGraph(model, mode="device", io_layout="flat"), scratch / "loss_src_flat.onnx"
    )
    src_rows = export_loss_graph(
        DnsmosLossGraph(model, mode="device", io_layout="rows"), scratch / "loss_src_rows.onnx"
    )

    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre_flat = scratch / "loss_src_flat.pre.onnx"
    pre_rows = scratch / "loss_src_rows.pre.onnx"
    quant_pre_process(str(src_flat), str(pre_flat), skip_symbolic_shape=True)
    quant_pre_process(str(src_rows), str(pre_rows), skip_symbolic_shape=True)

    calib_flat = make_calibration_batches(make_synthetic_batch(args.calib_segments, seed=100), w=W)
    calib_rows = make_calibration_batches(
        make_synthetic_batch(args.calib_segments, seed=100), w=W, rows=True
    )
    evalb = make_synthetic_batch(args.eval_segments, seed=999)

    bwd_convs = backward_conv_names(pre_flat)
    ladder = [[]] + [bwd_convs[:i] for i in (1, 2, 4, 7)]

    final = None
    for stage, exclude in enumerate(ladder):
        out = quantize_qdq(
            pre_flat, art / "dnsmos_loss_int8_qdq.onnx", calib_flat,
            extra_exclude=exclude, preprocessed=True,
        )
        rep = int8_grad_report(art / "dnsmos_loss_fp32.onnx", out, evalb, W)
        gain = functional_gate(out, art / "dnsmos_fwd_fp32.onnx")
        print(f"ladder[{stage}] exclude={len(exclude)} bwd convs -> "
              f"cosine_mean={rep['cosine_mean']:.3f} cosine_min={rep['cosine_min']:.3f} "
              f"ratio=[{rep['norm_ratio_min']:.2f},{rep['norm_ratio_max']:.2f}] "
              f"functional dOVRL={gain:+.3f}")
        if gain >= 0.3 and rep["cosine_mean"] >= 0.4:
            final = exclude
            print("gate: PASS")
            break
    if final is None:
        raise SystemExit("int8 loss gates FAILED at every ladder stage")

    # The device artifact is a separate quantization run (own calibration,
    # own QDQ placement, own mask repair) — walk its own exclusion ladder and
    # gate its gradients independently of the desktop artifact.
    bwd_rows = backward_conv_names(pre_rows)
    dev_ok = False
    for depth in sorted({len(final), len(final) + 1, 2, 4, 7}):
        if depth > 7:
            break
        out_dev = quantize_qdq(
            pre_rows, art / "dnsmos_loss_int8_qdq_stm32n6.onnx", calib_rows,
            extra_exclude=bwd_rows[:depth], preprocessed=True,
        )
        problems = check_device_constraints(out_dev)
        if problems:
            raise SystemExit(f"device constraint violations: {problems}")
        rep_dev = int8_grad_report(art / "dnsmos_loss_fp32_stm32n6.onnx", out_dev, evalb, W)
        gain_dev = functional_gate(out_dev, art / "dnsmos_fwd_fp32.onnx")
        print(f"stm32n6[exclude={depth}] -> cosine_mean={rep_dev['cosine_mean']:.3f} "
              f"cosine_min={rep_dev['cosine_min']:.3f} functional dOVRL={gain_dev:+.3f}")
        if gain_dev >= 0.3 and rep_dev["cosine_mean"] >= 0.4:
            dev_ok = True
            break
    if not dev_ok:
        raise SystemExit("stm32n6 int8 loss gates FAILED at every ladder depth")
    print(f"device int8 loss artifact ok: {out_dev}")
