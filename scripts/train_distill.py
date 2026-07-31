#!/usr/bin/env python3
"""Distill the official DNSMOS teacher into the PyTorch student.

The transplant is already exact, so run this only for: fine-tuning on a target
domain, retraining a modified/compact architecture (e.g. configs.student_small),
or producing weights for quantization-aware experiments.

Caveat (EUSIPCO 2024, "Hallucination in Perceptual Metric-Driven Speech
Enhancement"): optimizing a no-reference metric invites metric-hacking. When
using DNSMOS as a training loss downstream, pair it with an anchor loss
(SI-SNR/L1 to the input or a reference).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnsmos_trainable import load_transplanted
from dnsmos_trainable.data import FileListAudioDataset
from dnsmos_trainable.model import DnsmosModel
from dnsmos_trainable.teacher import OnnxTeacher


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, default=None, help="official sig_bak_ovr.onnx")
    parser.add_argument("--init", choices=["transplant", "scratch"], default="transplant")
    parser.add_argument("--student", choices=["full", "small"], default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    teacher = OnnxTeacher(args.teacher or root / "models" / "sig_bak_ovr.onnx")
    out_dir = args.out_dir or root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.student == "small":
        from configs.student_small import StudentSmall

        student = StudentSmall()
        # The frontend is frozen: it must carry the exact transplanted stft
        # weights, not random init.
        student.load_frontend_from_transplant(
            torch.load(root / "models" / "dnsmos_transplanted.pt", map_location="cpu",
                       weights_only=True)
        )
        lr = args.lr or 3e-4
    elif args.init == "transplant":
        student = load_transplanted(root / "models" / "dnsmos_transplanted.pt")
        lr = args.lr or 1e-4
    else:
        student = DnsmosModel()
        lr = args.lr or 3e-4
    for name, p in student.named_parameters():
        # The small student's frontend stays frozen (exact transplanted stft).
        p.requires_grad_(not (args.student == "small" and name.startswith("frontend.")))

    train_ds = FileListAudioDataset(args.train_list, train=True)
    val_ds = FileListAudioDataset(args.val_list, train=False)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = float("inf")

    for epoch in range(args.epochs):
        student.train()
        tot, n = 0.0, 0
        for wav in train_dl:
            with torch.no_grad():
                target = torch.from_numpy(teacher.raw_scores(wav.numpy()))
            raw, _ = student(wav)
            loss = torch.nn.functional.mse_loss(raw, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            tot += loss.item() * wav.shape[0]
            n += wav.shape[0]
        sched.step()

        student.eval()
        preds, targets = [], []
        with torch.no_grad():
            for wav in val_dl:
                targets.append(teacher.raw_scores(wav.numpy()))
                preds.append(student(wav)[0].numpy())
        preds, targets = np.concatenate(preds), np.concatenate(targets)
        val_mse = float(((preds - targets) ** 2).mean())
        r_ovr = pearson(preds[:, 2], targets[:, 2])
        print(f"epoch {epoch}: train {tot/n:.4f} val {val_mse:.4f} r_ovr {r_ovr:.3f}")
        if val_mse < best:
            best = val_mse
            torch.save(student.state_dict(), out_dir / f"distill_{args.student}_best.pt")
    print(f"best val MSE {best:.4f}")


if __name__ == "__main__":
    main()
