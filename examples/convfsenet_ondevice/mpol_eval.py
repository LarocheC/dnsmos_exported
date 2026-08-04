#!/usr/bin/env python3
"""Mask Polarization (MPol) against the null baselines, on the mixed testbed.

The paper's protocol comparison needs a *published* TTA method, not just our
own gradient loop. MPol (arXiv 2601.14770, IEEE OJ-SP) is the natural pick:
lightweight, source-free, and its objective is fully specified:

    L_W (M, M_P) = mean | sorted(vec M) - sorted(vec M_P) |     (1-Wasserstein)
    M_P = X_hat / (X_hat + N_hat)
    N_hat = mean of the k=32 lowest-power time bins of |Y|      (noise estimate)
    L_S = penalty on negative mask entries
    L = L_W + 0.1 * L_S

Faithfulness notes, stated up front:

* Our mask head is sigmoid-bounded, so `L_S` (their guard against the
  negative mask values their unbounded architectures emit) is identically
  zero here. The Wasserstein term is the whole objective.
* We adapt the same parameters as every other method in this study (the mask
  head's W, b) — their prescription is "normalization and output layers", and
  the mask head IS the output layer of our split. Their optimizer settings
  are kept: AdamW, lr 5e-4, and continual weight ensembling
  theta <- 0.8*theta + 0.2*theta_0 applied after each step.
* Their setting is online over a test stream; ours is per-clip episodic. To
  be generous, gains are recorded at several step counts and the BEST fixed
  step count is reported alongside the per-clip oracle over steps.

Same 150 mixed-shift clips (identical rng draw order as `mixed_adapt_eval`),
so every number is paired against the saved nulls in `mixed_shift.npz`.

    python mpol_eval.py --clips 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import dsp  # noqa: E402
from convfsenet_arch import build_split  # noqa: E402
from mixed_adapt_eval import SHIFTS, apply_shift  # noqa: E402
from pesq_adapt_eval import load_test_pairs  # noqa: E402
from pesq_predictor import SR  # noqa: E402

CPS = [0, 1, 2, 5, 10, 20]
K_NOISE = 32          # lowest-power time bins used for the noise estimate
LR = 5e-4             # theirs
BETA = 0.8            # weight-ensembling factor, theirs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=HERE / "artifacts" / "mixed_shift.npz")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--clips", type=int, default=150)
    ap.add_argument("--seconds", type=int, default=3)
    ap.add_argument("--rir-dir", type=Path, default=None,
                    help="measured RIR bank; must match how the npz was built")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--no-ensemble", action="store_true",
                    help="drop the beta=0.8 pull-back toward theta_0 — a "
                         "GENEROUS variant (moves further than the paper "
                         "prescribes) so the comparison cannot be a strawman")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "mpol.npz")
    args = ap.parse_args()

    torch.manual_seed(0); torch.set_num_threads(8)
    if args.rir_dir is not None:
        import mixed_adapt_eval as _mx
        from real_rir import load_rirs
        _mx.RIR_BANK = load_rirs(args.rir_dir)
        print(f"measured RIR bank: {len(_mx.RIR_BANK)} impulse responses")
    from pesq import pesq as pesq_fn

    trunk, head = build_split()
    sp = torch.load(args.split, map_location="cpu", weights_only=False)
    trunk.load_state_dict(sp["trunk"]); head.load_state_dict(sp["head"]); trunk.eval()

    z = np.load(args.npz, allow_pickle=True)
    kinds_ref, g_grid = z["kinds"], z["g_grid"]

    pairs = load_test_pairs(args.clips, args.seconds)
    rng = np.random.default_rng(args.seed)
    win = torch.from_numpy(dsp.hann_periodic()).float()

    n, nc = len(pairs), len(CPS)
    gains = np.full((n, nc), np.nan)
    kinds = np.empty(n, dtype=object)

    print(f"MPol on the mixed testbed: {n} clips, AdamW lr {args.lr}, "
          f"ensembling {'OFF' if args.no_ensemble else f'beta {BETA}'}\n")

    t0 = time.time()
    for i, (cl, no) in enumerate(pairs):
        kind = SHIFTS[i % len(SHIFTS)]
        kinds[i] = kind
        assert kind == str(kinds_ref[i]), f"clip order mismatch at {i}"
        ref_np, inp = apply_shift(cl, no, kind, rng)
        ref = ref_np.astype(np.float32)
        no_s, nf = dsp.rms_normalize(inp)
        X = dsp.stft(no_s)
        Xt = torch.from_numpy(np.abs(X)).float()
        Xc = torch.from_numpy(X).to(torch.complex64)
        mag = Xt[: trunk.n_features][None]
        with torch.no_grad():
            pad = mag[..., :1].repeat(1, 1, trunk.context_l)
            h = trunk(torch.cat([pad, mag], dim=-1))

        # noise spectrum from the k lowest-power frames of the input
        Ymag = Xt[: trunk.n_features]                       # [F, T]
        power = (Ymag ** 2).sum(dim=0)
        lowest = torch.argsort(power)[:K_NOISE]
        N_hat = Ymag[:, lowest].mean(dim=1, keepdim=True)   # [F, 1]

        def synth(full):
            y = torch.istft((Xc * full)[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                            win_length=dsp.WIN_LENGTH, window=win,
                            length=len(inp), center=True)[0]
            return y / nf

        def score(est) -> float:
            try:
                return pesq_fn(SR, ref, est.numpy().astype(np.float32), "wb")
            except Exception:
                return np.nan

        W0 = head.conv.weight.detach()[:, :, 0].clone()
        b0 = head.conv.bias.detach().clone()
        W = W0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        opt = torch.optim.AdamW([W, b], lr=args.lr)

        def mask_now():
            m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
            return torch.cat([m, torch.zeros(1, m.shape[1])], 0)

        base = score(synth(mask_now().detach()))
        if np.isnan(base):
            continue
        gains[i, 0] = 0.0

        for step in range(1, CPS[-1] + 1):
            m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])   # [F, T]
            X_hat = Ymag * m
            M_P = (X_hat / (X_hat + N_hat + 1e-9)).detach()
            loss = (torch.sort(m.flatten())[0]
                    - torch.sort(M_P.flatten())[0]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if not args.no_ensemble:
                with torch.no_grad():                   # continual ensembling
                    W.mul_(BETA).add_((1 - BETA) * W0)
                    b.mul_(BETA).add_((1 - BETA) * b0)
            if step in CPS:
                with torch.no_grad():
                    gains[i, CPS.index(step)] = score(synth(mask_now())) - base

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1:3d}/{n}  ({el:.0f}s, eta {el/(i+1)*(n-i-1)/60:.1f} min)",
                  flush=True)

    ok = ~np.isnan(gains).any(axis=1)
    gains, kinds = gains[ok], kinds[ok]
    g_grid = g_grid[: len(ok)][ok]           # paired rows; supports --clips < 150
    n = int(ok.sum())

    def stat(g):
        se = g.std(ddof=1) / np.sqrt(len(g))
        return f"{g.mean():+8.3f} {1.96*se:8.3f} {(g>0).mean()*100:8.0f}%"

    jbest = int(np.nanmean(g_grid, axis=0).argmax())
    null = g_grid[:, jbest]

    print(f"\n{n} clips (paired with mixed_shift.npz)")
    print(f"{'':>30} {'gain':>8} {'95% CI':>8} {'improved':>9}")
    for k, s in enumerate(CPS):
        print(f"{f'MPol, {s} steps':>30} {stat(gains[:, k])}")
    kb = int(np.nanmean(gains, axis=0).argmax())
    print(f"{'MPol, best fixed steps':>30} {stat(gains[:, kb])}   (= {CPS[kb]} steps)")
    print(f"{'MPol, per-clip oracle steps':>30} {stat(gains.max(1))}")
    print(f"{'global preset null':>30} {stat(null)}")
    d = gains[:, kb] - null
    se_d = d.std(ddof=1) / np.sqrt(n)
    print(f"\nMPol(best) vs null: {d.mean():+.3f} "
          f"({'resolves' if abs(d.mean()) > 1.96*se_d else 'does not resolve'} at 95%)")
    print("\nper-kind, MPol at best fixed steps:")
    for k in SHIFTS:
        s = kinds == k
        print(f"  {k:>9}: {stat(gains[s, kb])}")

    np.savez_compressed(args.out, gains=gains, kinds=kinds, steps=np.array(CPS))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
