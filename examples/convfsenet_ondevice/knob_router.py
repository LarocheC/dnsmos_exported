#!/usr/bin/env python3
"""Honest routers: can device-visible features pick the preset per clip?

The mixed-shift study left classify-then-preset as the only working method
(+0.166), but with the 4-way domain label assumed known. This closes that gap
and tests its upgrade, using the per-preset gains already measured and saved
in `mixed_shift.npz` as training labels:

  * **classify -> table**: multinomial logistic over acoustic features
    predicts the shift kind; the per-kind preset table (presets fit on train
    folds) is applied to the predicted label. The fully honest version of the
    +0.166 result — a hard-gated mixture of experts with a learned router.
  * **gain regression -> argmax**: ridge regression predicts each of the 21
    presets' gains directly from the features; per clip, apply the argmax.
    No discrete label at all — this can express within-kind variation
    (deeper reverb -> deeper softening) that no 4-row table can, and its
    ceiling is the per-clip oracle (+0.221) rather than the table (+0.176).

Everything the router sees is computable on the device from the input and
the enhancer's own mask — no reference, no PESQ, no learned critic:

    decay      lag-1 autocorrelation of the frame-energy envelope (reverb
               smears energy across frames)
    snr_est    total energy vs a minimum-statistics noise floor
    tilt       slope of the long-term log spectrum
    highband   energy fraction above 3 kHz (bandlimit collapses it)
    hf_floor   noise-floor flatness in the top octave
    m_mean     the enhancer's own mask mean (how hard it already suppresses)
    m_std      mask spread
    sisnr0     SI-SNR of the shipped output vs its input

Both routers are 5-fold cross-validated: router weights AND presets are fit
on train folds only. Requires `mixed_shift.npz` from `mixed_adapt_eval.py`
(the clip order is reproduced exactly; features are recomputed).

    python knob_router.py
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
from adapter_adapt_eval import si_snr  # noqa: E402
from convfsenet_arch import build_split  # noqa: E402
from mixed_adapt_eval import SHIFTS, apply_shift  # noqa: E402
from pesq_adapt_eval import load_test_pairs  # noqa: E402

FEAT_NAMES = ("decay1", "decay4", "mod_depth", "snr_est", "tilt", "highband",
              "rolloff", "floor_tilt", "floor_level", "hf_floor",
              "m_mean", "m_std", "sisnr0")


def features(inp: np.ndarray, m0: torch.Tensor, sisnr0: float) -> np.ndarray:
    X = dsp.stft(inp / (np.sqrt(np.mean(inp ** 2)) + 1e-12))
    P = np.abs(X) ** 2                                    # [F, T]
    F_, T = P.shape

    # envelope dynamics: reverb smears energy across frames (autocorrelation
    # up) and fills the gaps between words (modulation depth down)
    lenv = 10 * np.log10(P.sum(axis=0) + 1e-12)
    lenv = lenv - lenv.mean()
    def ac(lag):
        return float(np.corrcoef(lenv[:-lag], lenv[lag:])[0, 1])
    decay1, decay4 = ac(1), ac(4)
    mod_depth = float(lenv.std())

    # minimum-statistics noise floor, kept per band: its LEVEL separates the
    # noise kind, its SLOPE separates the tilt kind (tilt recolors the noise,
    # not the speech)
    S = np.empty_like(P)
    S[:, 0] = P[:, 0]
    for t in range(1, T):
        S[:, t] = 0.8 * S[:, t - 1] + 0.2 * P[:, t]
    w = 47
    floor = np.stack([S[:, max(0, t - w + 1): t + 1].min(axis=1)
                      for t in range(T)], axis=1)
    fmean = floor.mean(axis=1)                            # [F]
    snr_est = float(10 * np.log10(P.mean() / max(fmean.mean(), 1e-12)))
    floor_level = float(10 * np.log10(fmean.mean() + 1e-12)
                        - 10 * np.log10(P.mean() + 1e-12))
    fr = np.arange(F_) / F_
    floor_tilt = float(np.polyfit(fr, 10 * np.log10(fmean + 1e-12), 1)[0])

    spec = 10 * np.log10(P.mean(axis=1) + 1e-12)
    tilt = float(np.polyfit(fr, spec, 1)[0])
    k3 = int(3000 / 8000 * F_)
    highband = float(P[k3:].mean() / (P.mean() + 1e-12))
    cum = np.cumsum(P.mean(axis=1))
    rolloff = float(np.searchsorted(cum, 0.95 * cum[-1]) / F_)   # bandlimit
    top = fmean[int(0.5 * F_):]
    hf_floor = float(10 * np.log10(top.mean() + 1e-12)
                     - 10 * np.log10(fmean.mean() + 1e-12))

    return np.array([decay1, decay4, mod_depth, snr_est, tilt, highband,
                     rolloff, floor_tilt, floor_level, hf_floor,
                     float(m0.mean()), float(m0.std()), sisnr0])


def logistic_fit(X, y, classes, iters=400, lr=0.5):
    """Multinomial logistic regression, plain gradient descent."""
    n, d = X.shape
    K = len(classes)
    Y = np.zeros((n, K))
    for k, c in enumerate(classes):
        Y[y == c, k] = 1.0
    W = np.zeros((d + 1, K))
    Xb = np.hstack([X, np.ones((n, 1))])
    for _ in range(iters):
        Z = Xb @ W
        Z -= Z.max(axis=1, keepdims=True)
        Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
        W -= lr * (Xb.T @ (Pr - Y) / n + 1e-3 * W)
    return W


def logistic_predict(W, X, classes):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return np.array([classes[j] for j in (Xb @ W).argmax(axis=1)])


def logistic_proba(W, X):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    Z = Xb @ W
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z)
    return Pr / Pr.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=HERE / "artifacts" / "mixed_shift.npz")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--clips", type=int, default=150)
    ap.add_argument("--seconds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--rir-dir", type=Path, default=None,
                    help="measured RIR bank; must match how the npz was built")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    kinds, g_grid = z["kinds"], z["g_grid"]
    n = len(kinds)

    torch.manual_seed(0); torch.set_num_threads(8)
    if args.rir_dir is not None:
        import mixed_adapt_eval as _mx
        from real_rir import load_rirs
        _mx.RIR_BANK = load_rirs(args.rir_dir)
        print(f"measured RIR bank: {len(_mx.RIR_BANK)} impulse responses")
    trunk, head = build_split()
    sp = torch.load(args.split, map_location="cpu", weights_only=False)
    trunk.load_state_dict(sp["trunk"]); head.load_state_dict(sp["head"]); trunk.eval()

    # ---- recompute device-visible features, same clips in the same order ----
    pairs = load_test_pairs(args.clips, args.seconds)
    rng = np.random.default_rng(args.seed)
    win = torch.from_numpy(dsp.hann_periodic()).float()
    feats = np.zeros((n, len(FEAT_NAMES)))
    print(f"recomputing features for {n} clips...")
    t0 = time.time()
    for i, (cl, no) in enumerate(pairs[:n]):
        kind = SHIFTS[i % len(SHIFTS)]
        assert kind == str(kinds[i]), f"clip order mismatch at {i}"
        _, inp = apply_shift(cl, no, kind, rng)
        no_s, nf = dsp.rms_normalize(inp)
        X = dsp.stft(no_s)
        Xt = torch.from_numpy(np.abs(X)).float()
        mag = Xt[: trunk.n_features][None]
        with torch.no_grad():
            pad = mag[..., :1].repeat(1, 1, trunk.context_l)
            h = trunk(torch.cat([pad, mag], dim=-1))
            m0 = torch.sigmoid(head.conv(h))[0]
            full = torch.cat([m0, torch.zeros(1, m0.shape[1])], 0)
            y = torch.istft((torch.from_numpy(X).to(torch.complex64) * full)[None],
                            n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                            win_length=dsp.WIN_LENGTH, window=win,
                            length=len(inp), center=True)[0] / nf
            s0 = float(si_snr(y, torch.from_numpy(np.asarray(inp)).float()))
        feats[i] = features(np.asarray(inp), m0, s0)
    print(f"  done in {time.time()-t0:.0f}s")

    # ---- 5-fold CV, router and presets both fit on train only ----
    rng2 = np.random.default_rng(args.seed)
    order = rng2.permutation(n)
    g_cls = np.zeros(n)          # classify -> per-kind table
    g_conf = np.zeros(n)         # same, with confidence fallback
    taus = []
    g_reg = np.zeros(n)          # ridge gains -> argmax
    pred_kinds = np.empty(n, dtype=object)
    for f in range(args.folds):
        te = order[f::args.folds]
        tr = np.setdiff1d(order, te)
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
        Xtr, Xte = (feats[tr] - mu) / sd, (feats[te] - mu) / sd

        W = logistic_fit(Xtr, kinds[tr], SHIFTS)
        pk = logistic_predict(W, Xte, SHIFTS)
        pred_kinds[te] = pk
        table = {k: int(np.nanmean(g_grid[tr[kinds[tr] == k]], axis=0).argmax())
                 for k in SHIFTS}
        g_cls[te] = g_grid[te, [table[k] for k in pk]]

        # confidence-aware variant: commit to the kind preset only when the
        # router is sure; otherwise fall back to the safe global preset. The
        # bounded-failure principle applied to the router itself. tau is fit
        # on the train fold like every other parameter.
        jg_tr = int(np.nanmean(g_grid[tr], axis=0).argmax())
        pr_tr = logistic_proba(W, Xtr).max(axis=1)
        pk_tr = logistic_predict(W, Xtr, SHIFTS)
        best_tau, best_v = 0.0, -np.inf
        for tau in np.arange(0.3, 0.95, 0.05):
            v = np.where(pr_tr >= tau,
                         g_grid[tr, [table[k] for k in pk_tr]],
                         g_grid[tr, jg_tr]).mean()
            if v > best_v:
                best_tau, best_v = tau, v
        pr_te = logistic_proba(W, Xte).max(axis=1)
        g_conf[te] = np.where(pr_te >= best_tau,
                              g_grid[te, [table[k] for k in pk]],
                              g_grid[te, jg_tr])
        taus.append(best_tau)

        Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        A = Xb.T @ Xb + args.ridge * np.eye(Xb.shape[1])
        B = np.linalg.solve(A, Xb.T @ g_grid[tr])
        pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ B
        g_reg[te] = g_grid[te, pred.argmax(axis=1)]

    acc = (pred_kinds == kinds).mean()
    print(f"\nrouter accuracy (4-way, CV): {acc*100:.0f}%")
    conf = {k: (pred_kinds[kinds == k] == k).mean() for k in SHIFTS}
    print("  per kind: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in conf.items()))

    def stat(g):
        se = g.std(ddof=1) / np.sqrt(len(g))
        return f"{g.mean():+8.3f} {1.96*se:8.3f} {(g>0).mean()*100:8.0f}%"

    jbest = int(np.nanmean(g_grid, axis=0).argmax())
    null = g_grid[:, jbest]

    def vs_null(g):
        d = g - null
        se = d.std(ddof=1) / np.sqrt(len(d))
        star = "*" if abs(d.mean()) > 1.96 * se else ""
        return f"{d.mean():+8.3f}{star}"

    print(f"\n{'method':>34} {'gain':>8} {'95% CI':>8} {'improved':>9} {'vs null':>9}")
    print(f"{'global preset (null)':>34} {stat(null)} {'—':>9}")
    print(f"{'classify -> table (honest, CV)':>34} {stat(g_cls)} {vs_null(g_cls)}")
    print(f"{'  + confidence fallback (CV)':>34} {stat(g_conf)} {vs_null(g_conf)}"
          f"   tau {', '.join(f'{t:.2f}' for t in taus)}")
    print(f"{'gain regression -> argmax (CV)':>34} {stat(g_reg)} {vs_null(g_reg)}")
    ptab = np.array([g_grid[i, int(np.nanmean(g_grid[kinds == kinds[i]],
                                              axis=0).argmax())] for i in range(n)])
    print(f"{'per-kind table, oracle labels':>34} {stat(ptab)} {vs_null(ptab)}")
    print(f"{'per-clip grid oracle':>34} {stat(g_grid.max(1))} {vs_null(g_grid.max(1))}")

    # what the regressor uses
    mu, sd = feats.mean(0), feats.std(0) + 1e-9
    Xb = np.hstack([(feats - mu) / sd, np.ones((n, 1))])
    B = np.linalg.solve(Xb.T @ Xb + args.ridge * np.eye(Xb.shape[1]),
                        Xb.T @ g_grid)
    imp = np.abs(B[:-1]).mean(axis=1)
    top = np.argsort(imp)[::-1]
    print("\nregressor feature weight (mean |coef| across presets): "
          + ", ".join(f"{FEAT_NAMES[j]} {imp[j]:.3f}" for j in top))


if __name__ == "__main__":
    main()
