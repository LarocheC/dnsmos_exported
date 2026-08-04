#!/usr/bin/env python3
"""Can a device-computable rule capture the oracle stopping headroom?

`adapt_budget.py` measured a best fixed budget of +0.083 PESQ against an oracle
per-clip bound of +0.178. This asks whether any of that +0.094 gap is reachable
using only signals a device actually has.

**The admissibility rule**: true PESQ needs the clean reference and is not
available at the edge. Only these are:

  * `believed[k]` — the predictor's own output, already computed every step
  * `sisnr[k]`    — SI-SNR of the current output against the *noisy input*,
                    which is a reference the device has (it is not clean, but
                    it is free)

Anything derived from those is legal; anything touching `true` is oracle-only
and reported solely as an upper bound.

**Thresholds are cross-validated.** With 150 clips and a one-parameter rule it
is trivially easy to fit the threshold on the same data it is scored on and
report a win that will not replicate. Every rule here picks its threshold on
training folds and is scored on held-out folds, and the fixed-budget baseline
gets the same treatment so the comparison is fair.

    python adapt_gate.py
    python adapt_gate.py --npz artifacts/adapt_budget_nosisnr.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- rules
# Each returns, per clip, the index into `steps` at which to stop.
# Signature: (bel, snr, t) -> int array of checkpoint indices.

def rule_fixed(bel, snr, t):
    """Adapt every clip for the same number of steps. The baseline."""
    return np.full(len(bel), int(t))


def rule_belief_rise(bel, snr, t):
    """Stop once the metric has climbed `t` PESQ above where it started.

    Rationale: the belief rise is the visible part of exploitation. Capping it
    caps how far the optimizer is allowed to talk itself up.
    """
    rise = bel - bel[:, :1]
    return _first_true(rise >= t)


def rule_saturation(bel, snr, t):
    """Stop when the metric stops moving — marginal gain per step below `t`.

    Rationale: the budget curve shows belief pinning near its ceiling almost
    immediately. Once it saturates the gradient carries no more information,
    so further steps are drift.
    """
    d = np.diff(bel, axis=1, prepend=bel[:, :1])
    return _first_true(d < t, skip_first=True)


def rule_headroom_gate(bel, snr, t):
    """Adapt only clips whose starting belief is below `t`; others get 0 steps.

    Rationale: 10% of clips are best left alone. If the metric already thinks a
    clip is good, there is nothing to chase and adaptation can only drift.
    Stops at a fixed 10 steps when it does fire.
    """
    k10 = 8                                    # index of step 10 in the grid
    return np.where(bel[:, 0] < t, k10, 0)


def rule_sisnr_drift(bel, snr, t):
    """Stop once the output has moved `t` dB in SI-SNR from where it started.

    Rationale: a purely signal-domain trust region, needing no reference and
    no metric — the cheapest possible rule.
    """
    return _first_true(np.abs(snr - snr[:, :1]) >= t)


def _first_true(mask, skip_first=False):
    """Index of the first True per row; last index if a row never fires."""
    m = mask.copy()
    m[:, 0] = False                            # never stop before adapting
    if skip_first:
        m[:, 1] = False
    idx = np.argmax(m, axis=1)
    idx[~m.any(axis=1)] = mask.shape[1] - 1
    return idx


RULES = [
    ("fixed budget",        rule_fixed,          None),
    ("belief-rise cap",     rule_belief_rise,    np.arange(0.1, 1.6, 0.05)),
    ("belief saturation",   rule_saturation,     np.arange(0.002, 0.12, 0.002)),
    ("headroom gate",       rule_headroom_gate,  np.arange(2.8, 4.5, 0.05)),
    ("SI-SNR drift cap",    rule_sisnr_drift,    np.arange(0.1, 4.0, 0.1)),
]


def evaluate(rule, gain, bel, snr, t):
    k = rule(bel, snr, t)
    return gain[np.arange(len(gain)), k]


def cross_val(rule, grid, gain, bel, snr, folds, rng):
    """Pick the threshold on training folds, score on the held-out fold."""
    n = len(gain)
    order = rng.permutation(n)
    out = np.zeros(n)
    chosen = []
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        best_t, best_v = grid[0], -np.inf
        for t in grid:
            v = evaluate(rule, gain[tr], bel[tr], snr[tr], t).mean()
            if v > best_v:
                best_t, best_v = t, v
        out[te] = evaluate(rule, gain[te], bel[te], snr[te], best_t)
        chosen.append(best_t)
    return out, chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=HERE / "artifacts" / "adapt_budget.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.npz)
    true, bel, snr, steps = z["true"], z["believed"], z["sisnr"], z["steps"]
    ok = ~np.isnan(true).any(axis=1)
    true, bel, snr = true[ok], bel[ok], snr[ok]
    gain = true - true[:, :1]
    n = len(true)
    rng = np.random.default_rng(args.seed)

    print(f"{args.npz.name}: {n} clips, checkpoints {list(steps)}\n")

    # ---- can anything predict the per-clip optimum at all? ----
    best_k = gain.argmax(axis=1)
    best_step = steps[best_k]
    oracle = gain.max(axis=1)
    feats = {
        "believed at step 0": bel[:, 0],
        "belief rise by step 2": bel[:, 2] - bel[:, 0],
        "belief rise by step 10": bel[:, 8] - bel[:, 0],
        "SI-SNR at step 0": snr[:, 0],
        "SI-SNR drift by step 10": snr[:, 8] - snr[:, 0],
    }
    print("do device-visible signals predict the oracle stopping point?")
    print(f"{'signal':>26} {'r vs best step':>15} {'r vs oracle gain':>18}")
    for k, v in feats.items():
        print(f"{k:>26} {np.corrcoef(v, best_step)[0,1]:>+15.3f} "
              f"{np.corrcoef(v, oracle)[0,1]:>+18.3f}")

    # ---- rules, cross-validated ----
    print(f"\n{args.folds}-fold cross-validated (threshold fit on train, scored on test)")
    print(f"{'rule':>20} {'gain':>9} {'95% CI':>9} {'improved':>9} {'vs fixed':>10} {'thresholds':>22}")

    fixed_grid = np.arange(len(steps))
    base, base_t = cross_val(rule_fixed, fixed_grid, gain, bel, snr, args.folds, rng)
    rows = []
    for name, fn, grid in RULES:
        if grid is None:
            g, ts = base, [steps[int(t)] for t in base_t]
        else:
            rng2 = np.random.default_rng(args.seed)      # identical folds
            g, ts = cross_val(fn, grid, gain, bel, snr, args.folds, rng2)
        d = g - base
        se = g.std(ddof=1) / np.sqrt(n)
        se_d = d.std(ddof=1) / np.sqrt(n) if grid is not None else float("nan")
        delta = ("—" if grid is None
                 else f"{d.mean():+.3f}" + ("*" if abs(d.mean()) > 1.96 * se_d else ""))
        tt = ", ".join(f"{t:g}" for t in ts[:3]) + ("..." if len(ts) > 3 else "")
        print(f"{name:>20} {g.mean():+9.3f} {1.96*se:9.3f} {(g>0).mean()*100:8.0f}% "
              f"{delta:>10} {tt:>22}")
        rows.append((name, g))

    print(f"{'never adapt':>20} {0.0:+9.3f} {0.0:9.3f} {0:8.0f}% {-base.mean():+10.3f}")
    se_o = oracle.std(ddof=1) / np.sqrt(n)
    print(f"{'ORACLE (not a rule)':>20} {oracle.mean():+9.3f} {1.96*se_o:9.3f} "
          f"{(oracle>0).mean()*100:8.0f}% {oracle.mean()-base.mean():+10.3f}")

    print("\n* = paired difference vs the fixed budget resolves at 95%")
    best = max(rows, key=lambda r: r[1].mean())
    cap = (best[1].mean() - base.mean()) / (oracle.mean() - base.mean()) * 100
    print(f"best rule: {best[0]} — captures {cap:.0f}% of the oracle headroom")


if __name__ == "__main__":
    main()
