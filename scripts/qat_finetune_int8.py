#!/usr/bin/env python3
"""QAT self-distillation: fine-tune body weights under fake quantization so the
int8 artifact matches the fp32 reference within the delta gates."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dnsmos_trainable import load_transplanted
from dnsmos_trainable.qat import QatDnsmosModel, calibrate_act_ranges
from dnsmos_trainable.verify import make_synthetic_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--calib-segments", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    root = Path(__file__).resolve().parents[1]
    out = args.out or root / "models" / "dnsmos_qat_int8.pt"

    teacher = load_transplanted(root / "models" / "dnsmos_transplanted.pt")
    for p in teacher.parameters():
        p.requires_grad_(False)

    calib = torch.from_numpy(make_synthetic_batch(args.calib_segments, seed=100))
    ranges = calibrate_act_ranges(teacher, calib)
    student = QatDnsmosModel(teacher, ranges)

    with torch.no_grad():
        base = torch.nn.functional.mse_loss(
            student(calib[:4])[0], teacher(calib[:4])[0]
        ).item()
    print(f"initial fake-quant MSE on calib: {base:.5f}")

    opt = torch.optim.Adam(student.trainable_parameters(), lr=args.lr)
    for step in range(args.steps):
        wav = torch.from_numpy(make_synthetic_batch(args.batch, seed=2000 + step))
        with torch.no_grad():
            target, _ = teacher(wav)
        raw, _ = student(wav)
        loss = torch.nn.functional.mse_loss(raw, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss {loss.item():.5f}")

    torch.save(student.model.state_dict(), out)
    print(f"saved QAT weights to {out}")
