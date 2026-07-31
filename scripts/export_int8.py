#!/usr/bin/env python3
"""Build the int8 QDQ forward artifacts (desktop + STM32N6) and report gates.

Uses QAT-fine-tuned weights (models/dnsmos_qat_int8.pt) when present — PTQ on
the exact transplant leaves SIG above the delta gate — falling back to the
plain transplant otherwise. Deltas are always measured against the ORIGINAL
transplanted fp32 artifact: that is the reference the int8 model must
approximate.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dnsmos_trainable import load_transplanted
from dnsmos_trainable.export import (
    export_forward,
    make_calibration_batches,
    quantize_qdq,
)
from dnsmos_trainable.verify import int8_delta_report, make_synthetic_batch

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--calib-segments", type=int, default=16)
    parser.add_argument("--eval-segments", type=int, default=50)
    parser.add_argument("--percentile", type=float, default=99.99)
    args = parser.parse_args()

    art = root / "artifacts"
    weights = args.weights
    if weights is None:
        qat = root / "models" / "dnsmos_qat_int8.pt"
        weights = qat if qat.exists() else root / "models" / "dnsmos_transplanted.pt"
    print(f"int8 source weights: {weights.name}")
    model = load_transplanted(weights)

    scratch = art / "_int8_src"
    scratch.mkdir(parents=True, exist_ok=True)
    fp32_src = export_forward(model, scratch / "fwd_fp32_src.onnx", device=False)
    fp32_src_dev = export_forward(model, scratch / "fwd_fp32_src_dev.onnx", device=True)

    calib = make_calibration_batches(make_synthetic_batch(args.calib_segments, seed=100))
    calib_dev = make_calibration_batches(
        make_synthetic_batch(args.calib_segments, seed=100), rows=True
    )
    from onnxruntime.quantization import CalibrationMethod

    out = quantize_qdq(
        fp32_src, art / "dnsmos_fwd_int8_qdq.onnx", calib,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"percentile": args.percentile},
    )
    out_dev = quantize_qdq(
        fp32_src_dev, art / "dnsmos_fwd_int8_qdq_stm32n6.onnx", calib_dev,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"percentile": args.percentile},
    )

    evalb = make_synthetic_batch(args.eval_segments, seed=999)
    rep = int8_delta_report(art / "dnsmos_fwd_fp32.onnx", out, evalb)
    print("MOS deltas vs original fp32 [SIG, BAK, OVRL]:")
    for k, v in rep.items():
        print(f"  {k}: {[f'{x:.4f}' for x in v]}")
    ok = all(m < 0.05 for m in rep["mean"]) and all(p < 0.10 for p in rep["p95"])
    print("gate:", "PASS" if ok else "FAIL", "(mean < 0.05, p95 < 0.10)")
    if not ok:
        raise SystemExit(1)
