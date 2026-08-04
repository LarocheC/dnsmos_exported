#!/usr/bin/env python3
"""Gate 0: can forty-year-old adaptive DSP match the learned-critic loop?

The budget curve put gradient adaptation against the learned critic at
**+0.083 PESQ** (fixed 10-step budget, 150 reverberant clips). Before building
anything more on that number, the killer question: speech enhancement has been
*adaptive* since the 1970s — running noise estimates, Wiener gains — with no
learning, no critic, and no exploitation surface at all. If any of that
matches +0.083 on the same protocol, the learned-critic premise is dead.

Same 150 clips, same RIRs (identical rng draw order as `adapt_budget.py`, so
every per-clip number is paired against the committed run), same reverberant
clean reference. Candidates, all starting from the enhancer's shipped output:

1. **Decision-directed Wiener post-filter** — minimum-statistics noise PSD,
   decision-directed a-priori SNR, gain floor. Zero parameters fit per clip;
   the textbook adaptive method.
2. **Mask rescale grid** `max(m^a, f)` over 21 (a, f) combos — the cheapest
   conceivable per-clip "adaptation", with four selection rules:
   * `oracle`  — best true PESQ (upper bound; compare to oracle stopping +0.178)
   * `v1-pick` / `v2-pick` — argmax of a predictor's belief over the grid.
     This is *zeroth-order* use of the learned critic: it only ever scores 21
     near-manifold candidates and is never differentiated through, so the
     saturation mechanism that caps the gradient loop cannot fire. A cheap
     preview of the don't-differentiate-through-the-judge idea.
   * `sisnr-floor` — most suppressive candidate keeping SI-SNR(vs noisy)
     >= 6 dB; the trust-region prior with no critic at all.

Decision rule stated up front: if 1. or a device-pickable variant of 2. reaches
+0.083, stop the learned-critic line.

    python classical_baseline.py --clips 150
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
from pesq_adapt_eval import load_test_pairs  # noqa: E402
from pesq_predictor import PesqPredictor, compress, SR  # noqa: E402
from reverb_adapt_eval import make_rir, reverberate  # noqa: E402

# the rescale grid: exponent bends the mask, floor caps the suppression depth
GRID = [(a, f) for a in (0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0)
        for f in (0.0, 0.05, 0.10)]
IDENTITY = GRID.index((1.0, 0.0))


def wiener_dd(E: np.ndarray, hop_s: float = 0.016, alpha_dd: float = 0.98,
              g_floor: float = 0.15, min_win_s: float = 0.75,
              bias: float = 1.5) -> np.ndarray:
    """Decision-directed Wiener gain on a complex STFT [F, T].

    Noise PSD by minimum statistics: smoothed periodogram, running minimum
    over `min_win_s`, times a bias compensation. A-priori SNR by the
    Ephraim-Malah decision-directed rule.
    """
    F_, T = E.shape
    P = np.abs(E) ** 2
    S = np.empty_like(P)
    S[:, 0] = P[:, 0]
    for t in range(1, T):
        S[:, t] = 0.8 * S[:, t - 1] + 0.2 * P[:, t]
    w = max(2, int(min_win_s / hop_s))
    N = np.empty_like(P)
    for t in range(T):
        N[:, t] = S[:, max(0, t - w + 1): t + 1].min(axis=1)
    N = bias * np.maximum(N, 1e-12)

    G = np.ones_like(P)
    xi_prev = np.ones(F_)
    for t in range(T):
        gamma = P[:, t] / N[:, t]
        xi = alpha_dd * xi_prev + (1.0 - alpha_dd) * np.maximum(gamma - 1.0, 0.0)
        G[:, t] = np.maximum(xi / (1.0 + xi), g_floor)
        xi_prev = (G[:, t] ** 2) * gamma
    return E * G


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=Path, default=HERE / "artifacts" / "pesq_predictor_sig.pt")
    ap.add_argument("--v2", type=Path, default=HERE / "artifacts" / "pesq_predictor_adv.pt")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--clips", type=int, default=150)
    ap.add_argument("--seconds", type=int, default=3)
    ap.add_argument("--rt60-lo", type=float, default=0.2)
    ap.add_argument("--rt60-hi", type=float, default=0.6)
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "classical_baseline.npz")
    args = ap.parse_args()

    torch.manual_seed(0); torch.set_num_threads(8)
    from pesq import pesq as pesq_fn

    metrics = {}
    for tag, p in (("v1", args.v1), ("v2", args.v2)):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = PesqPredictor(ck["dim"], ck.get("head", "lsig")).eval()
        m.load_state_dict(ck["model"])
        for q in m.parameters():
            q.requires_grad_(False)
        metrics[tag] = m

    trunk, head = build_split()
    sp = torch.load(args.split, map_location="cpu", weights_only=False)
    trunk.load_state_dict(sp["trunk"]); head.load_state_dict(sp["head"]); trunk.eval()

    pairs = load_test_pairs(args.clips, args.seconds)
    rng = np.random.default_rng(args.seed)          # identical draw order to adapt_budget
    win = torch.from_numpy(dsp.hann_periodic()).float()

    n = len(pairs)
    METHODS = ["wiener", "grid-oracle", "grid-v1", "grid-v2", "grid-sisnr"]
    gains = {k: np.full(n, np.nan) for k in METHODS}
    base_all = np.full(n, np.nan)
    pick_counts = {k: np.zeros(len(GRID), int) for k in ("grid-oracle", "grid-v1", "grid-v2")}

    print(f"classical baselines: {n} clips x {args.seconds}s, "
          f"rt60 U[{args.rt60_lo},{args.rt60_hi}] (paired with adapt_budget seed {args.seed})\n")

    t0 = time.time()
    for i, (cl, no) in enumerate(pairs):
        rt60 = float(rng.uniform(args.rt60_lo, args.rt60_hi))
        rir = make_rir(rt60, rng)
        cl_rev = reverberate(cl, rir)
        no_rev = cl_rev + (no - cl)
        no_s, nf = dsp.rms_normalize(no_rev)

        X = dsp.stft(no_s)
        Xt = torch.from_numpy(np.abs(X)).float()
        Xc = torch.from_numpy(X).to(torch.complex64)
        nmag_c = compress(Xt)
        mag = Xt[: trunk.n_features][None]
        with torch.no_grad():
            pad = mag[..., :1].repeat(1, 1, trunk.context_l)
            h = trunk(torch.cat([pad, mag], dim=-1))
            m0 = torch.sigmoid(head.conv(h))[0]

        def synth(full):
            y = torch.istft((Xc * full)[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                            win_length=dsp.WIN_LENGTH, window=win,
                            length=len(no_rev), center=True)[0]
            return y / nf

        ref = cl_rev.astype(np.float32)
        noisy_t = torch.from_numpy(no_rev).float()

        def score(est) -> float:
            try:
                return pesq_fn(SR, ref, est.numpy().astype(np.float32), "wb")
            except Exception:
                return np.nan

        full0 = torch.cat([m0, torch.zeros(1, m0.shape[1])], 0)
        base = score(synth(full0))
        if np.isnan(base):
            continue
        base_all[i] = base

        # ---- 1. Wiener post-filter on the enhanced spectrum ----
        E = (X * full0.numpy())
        Ew = wiener_dd(E)
        yw = torch.istft(torch.from_numpy(Ew).to(torch.complex64)[None],
                         n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                         win_length=dsp.WIN_LENGTH, window=win,
                         length=len(no_rev), center=True)[0] / nf
        gains["wiener"][i] = score(yw) - base

        # ---- 2. the rescale grid ----
        g_true = np.full(len(GRID), np.nan)
        g_bel = {"v1": np.zeros(len(GRID)), "v2": np.zeros(len(GRID))}
        g_snr = np.zeros(len(GRID))
        g_sup = np.zeros(len(GRID))
        for j, (a, f) in enumerate(GRID):
            mj = torch.clamp(torch.pow(m0, a), min=f)
            fullj = torch.cat([mj, torch.zeros(1, mj.shape[1])], 0)
            est = synth(fullj)
            g_true[j] = score(est) - base
            g_snr[j] = float(si_snr(est, noisy_t))
            g_sup[j] = float(fullj.mean())
            with torch.no_grad():
                emag = compress(Xt * fullj)[None]
                for tag, m_ in metrics.items():
                    g_bel[tag][j] = float(m_(nmag_c[None], emag)[0])

        if np.isnan(g_true).all():
            continue
        jo = int(np.nanargmax(g_true))
        gains["grid-oracle"][i] = g_true[jo]; pick_counts["grid-oracle"][jo] += 1
        for tag in ("v1", "v2"):
            jp = int(np.argmax(g_bel[tag]))
            gains[f"grid-{tag}"][i] = g_true[jp]; pick_counts[f"grid-{tag}"][jp] += 1
        ok = g_snr >= args.sisnr_floor
        js = int(np.argmin(np.where(ok, g_sup, np.inf))) if ok.any() else IDENTITY
        gains["grid-sisnr"][i] = g_true[js]

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1:3d}/{n}  ({el:.0f}s, eta {el/(i+1)*(n-i-1)/60:.1f} min)", flush=True)

    ok = ~np.isnan(base_all)
    print(f"\nbaseline PESQ {np.nanmean(base_all):.3f} on {int(ok.sum())} clips")
    print(f"reference numbers from the learned-critic loop: "
          f"fixed budget +0.083, oracle stopping +0.178\n")
    print(f"{'method':>14} {'gain':>8} {'95% CI':>8} {'improved':>9}")
    for k in METHODS:
        g = gains[k][ok & ~np.isnan(gains[k])]
        se = g.std(ddof=1) / np.sqrt(len(g))
        print(f"{k:>14} {g.mean():+8.3f} {1.96*se:8.3f} {(g>0).mean()*100:8.0f}%")

    for k, c in pick_counts.items():
        top = np.argsort(c)[::-1][:3]
        picks = ", ".join(f"a={GRID[j][0]:g},f={GRID[j][1]:g} x{c[j]}" for j in top if c[j])
        print(f"  {k} top picks: {picks}")

    np.savez_compressed(args.out, base=base_all,
                        **{k.replace('-', '_'): v for k, v in gains.items()})
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
