#!/usr/bin/env python3
"""Transplant official DNSMOS weights into a PyTorch state dict + parity check."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dnsmos_trainable.model import DnsmosModel
from dnsmos_trainable.transplant import transplant
from dnsmos_trainable.verify import compare_intermediates, compare_raw_scores, make_synthetic_batch, ort_session

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=root / "models" / "sig_bak_ovr.onnx")
    parser.add_argument("--out", type=Path, default=root / "models" / "dnsmos_transplanted.pt")
    args = parser.parse_args()

    sd = transplant(args.official)
    model = DnsmosModel()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("poly.") for k in missing), missing
    model.eval()

    batch = make_synthetic_batch(8)
    diff = compare_raw_scores(model, ort_session(args.official), batch)
    stages = compare_intermediates(args.official, model, batch)
    print("stage-wise max|diff|:", {k: f"{v:.2e}" for k, v in stages.items()})
    print(f"raw-score max|diff| vs official (fp32): {diff:.2e}")

    # Exactness gate: in float64 the only remaining diff is ORT's own fp32
    # rounding. fp32-vs-fp32 across engines carries both engines' noise.
    m64 = DnsmosModel().double()
    m64.load_state_dict({k: v.double() for k, v in sd.items()}, strict=False)
    m64.eval()
    with torch.no_grad():
        raw64, _ = m64(torch.from_numpy(batch).double())
    import numpy as np

    ref = ort_session(args.official).run(None, {"input_1": batch})[0]
    diff64 = float(np.abs(raw64.numpy() - ref).max())
    print(f"raw-score max|diff| vs official (torch fp64): {diff64:.2e}")
    if diff64 > 1e-5 or diff > 5e-5:
        raise SystemExit("parity gate FAILED (fp64>1e-5 or fp32>5e-5)")

    torch.save(sd, args.out)
    print(f"saved {args.out}")
