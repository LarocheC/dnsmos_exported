#!/usr/bin/env python3
"""Build the int8 QDQ loss-graph artifacts (fwd + hand-written bwd quantized).

Quantizes Conv/Gemm/MatMul in BOTH the forward recompute and the VJP chain of
the device-mode loss graph. Calibration runs full (wav, w) pairs so activation
AND gradient tensor ranges are observed. On gate failure an exclusion ladder
removes the most sensitive backward nodes from quantization until gates pass;
elementwise glue (masks, Reciprocal, overlap-add) is never quantized because
only Conv/Gemm/MatMul are eligible.

Gates vs the fp32 loss graph reference (original transplanted weights):
cosine(grad_int8, grad_fp32) >= 0.95 floor / 0.99 target, grad-norm ratio in
[0.8, 1.25], per segment.
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
)

W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def backward_conv_names(loss_path: Path) -> list[str]:
    """Backward convs, most-sensitive-first for the exclusion ladder.

    In the device loss graph the backward convs are the ones whose weights are
    the flipped kernels [C_in, C_out, 3, 3]; earlier backward layers (closer
    to the waveform gradient) are later in topological order.
    """
    from onnx import numpy_helper

    m = onnx.load(str(loss_path))
    inits = {t.name: tuple(t.dims) for t in m.graph.initializer}
    fwd_shapes = {(128, 1, 3, 3), (64, 128, 3, 3), (64, 64, 3, 3), (32, 64, 3, 3),
                  (32, 32, 3, 3), (64, 32, 3, 3)}
    flipped = {(1, 128, 3, 3), (128, 64, 3, 3), (32, 64, 3, 3),
               (32, 32, 3, 3), (64, 64, 3, 3), (32, 64, 3, 3)}
    convs = [(n.name, inits.get(n.input[1])) for n in m.graph.node if n.op_type == "Conv"]
    # The graph contains 14 convs: 7 forward then 7 backward in topo order.
    names = [name for name, _ in convs]
    return list(reversed(names[7:]))  # bwd convs, deepest (conv1-grad) first


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--calib-segments", type=int, default=12)
    parser.add_argument("--eval-segments", type=int, default=50)
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
    # Desktop-layout (flat, dynamic-batch-free device mode) + device-layout.
    src_flat = export_loss_graph(
        DnsmosLossGraph(model, mode="device", io_layout="flat"), scratch / "loss_src_flat.onnx"
    )
    src_rows = export_loss_graph(
        DnsmosLossGraph(model, mode="device", io_layout="rows"), scratch / "loss_src_rows.onnx"
    )

    calib_flat = make_calibration_batches(make_synthetic_batch(args.calib_segments, seed=100), w=W)
    calib_rows = make_calibration_batches(
        make_synthetic_batch(args.calib_segments, seed=100), w=W, rows=True
    )
    evalb = make_synthetic_batch(args.eval_segments, seed=999)

    ladder = [[]]
    bwd_convs = backward_conv_names(src_flat)
    for i in (1, 2, 4, 7):
        ladder.append(bwd_convs[:i])

    from onnxruntime.quantization import CalibrationMethod

    final = None
    for stage, exclude in enumerate(ladder):
        out = quantize_qdq(
            src_flat, art / "dnsmos_loss_int8_qdq.onnx", calib_flat,
            extra_exclude=exclude,
            calibrate_method=CalibrationMethod.Percentile,
            extra_options={"percentile": 99.99},
        )
        rep = int8_grad_report(art / "dnsmos_loss_fp32.onnx", out, evalb, W)
        print(f"ladder[{stage}] exclude={len(exclude)} bwd convs -> {rep}")
        if rep["cosine_min"] >= 0.95 and 0.8 <= rep["norm_ratio_min"] and rep["norm_ratio_max"] <= 1.25:
            final = exclude
            print("gate: PASS")
            break
    if final is None:
        raise SystemExit("int8 loss gates FAILED at every ladder stage")

    # Device-layout artifact with the same exclusion depth (names differ per
    # graph, so recompute on the rows graph).
    bwd_rows = backward_conv_names(src_rows)
    out_dev = quantize_qdq(
        src_rows, art / "dnsmos_loss_int8_qdq_stm32n6.onnx", calib_rows,
        extra_exclude=bwd_rows[: len(final)],
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"percentile": 99.99},
    )
    problems = check_device_constraints(out_dev)
    if problems:
        raise SystemExit(f"device constraint violations: {problems}")
    print(f"device int8 loss artifact ok: {out_dev}")
