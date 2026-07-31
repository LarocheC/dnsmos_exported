#!/usr/bin/env python3
"""Build the int8 QDQ forward artifacts (desktop + STM32N6) and report gates.

Pipeline: QAT-fine-tuned weights (models/dnsmos_qat_int8.pt; falls back to
the plain transplant) + deterministic QDQ insertion via the sim's own
quantization grids (qdq_writer). quantize_static recalibration is NOT used
here: retraining targets a specific grid, and redeploying on a recalibrated
grid consistently loses accuracy versus the sim.

Deltas are always measured against the ORIGINAL transplanted fp32 artifact —
that is the reference the int8 model must approximate.
Gates: mean |dMOS| < 0.05 and p95 < 0.10 per output on >= 50 segments.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dnsmos_trainable import load_transplanted
from dnsmos_trainable.export import export_forward
from dnsmos_trainable.qat import calibrate_act_ranges
from dnsmos_trainable.qdq_writer import write_qdq_from_sim
from dnsmos_trainable.verify import (
    check_device_constraints,
    int8_delta_report,
    make_synthetic_batch,
)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--calib-segments", type=int, default=16)
    parser.add_argument("--eval-segments", type=int, default=50)
    args = parser.parse_args()

    torch.set_num_threads(4)
    art = root / "artifacts"
    weights = args.weights
    if weights is None:
        qat = root / "models" / "dnsmos_qat_int8.pt"
        weights = qat if qat.exists() else root / "models" / "dnsmos_transplanted.pt"
    print(f"int8 source weights: {weights.name}")

    # Ranges are calibrated on the ORIGINAL model (the QAT run froze them from
    # the same seed/procedure, so this reproduces the grids training targeted).
    torch.manual_seed(0)
    teacher = load_transplanted(root / "models" / "dnsmos_transplanted.pt")
    calib = torch.from_numpy(make_synthetic_batch(args.calib_segments, seed=100))
    ranges = calibrate_act_ranges(teacher, calib)
    model = load_transplanted(weights)

    scratch = art / "_int8_src"
    scratch.mkdir(parents=True, exist_ok=True)
    src = export_forward(model, scratch / "fwd_fp32_src.onnx", device=False)
    src_dev = export_forward(model, scratch / "fwd_fp32_src_dev.onnx", device=True)

    out = write_qdq_from_sim(src, art / "dnsmos_fwd_int8_qdq.onnx", ranges)
    out_dev = write_qdq_from_sim(src_dev, art / "dnsmos_fwd_int8_qdq_stm32n6.onnx", ranges)

    problems = check_device_constraints(out_dev)
    if problems:
        raise SystemExit(f"device constraint violations: {problems}")

    evalb = make_synthetic_batch(args.eval_segments, seed=999)
    rep = int8_delta_report(art / "dnsmos_fwd_fp32.onnx", out, evalb)
    print("MOS deltas vs original fp32 [SIG, BAK, OVRL]:")
    for k, v in rep.items():
        print(f"  {k}: {[f'{x:.4f}' for x in v]}")
    ok = all(m < 0.05 for m in rep["mean"]) and all(p < 0.10 for p in rep["p95"])
    print("gate:", "PASS" if ok else "FAIL", "(mean < 0.05, p95 < 0.10)")
    if not ok:
        raise SystemExit(1)
