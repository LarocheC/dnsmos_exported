#!/usr/bin/env python3
"""Capacity or direction? Reduce adaptation to the two knobs that contain the fix.

The mixed-shift result invites the objection: maybe gradient adaptation failed
because only the 49k-parameter mask head was adapted — too little capacity, or
the wrong parameters. This experiment removes the objection in the strongest
possible way: adapt exactly TWO scalars,

    m = sigmoid(t * z + c)        z = the head's logits, frozen

`t` is a temperature (t<1 softens the mask — for the suppressed regions that
dominate, sigmoid(t*z) ~ m^t) and `c` a bias shift (c>0 raises the floor).
This family *contains* the reverb fix: the winning preset m^0.5/floor-0.1 is
approximately (t=0.5, c>0), worth +0.64 PESQ. Nothing here can be "too small";
there is almost nothing to exploit; the right answer is on the table.

So the critic's steering is measured directly: which way does it turn `t`?

  * If capacity was the binding constraint, the loop should now find t~0.5
    and match the preset.
  * If direction is the failure — the off-distribution critic *prefers* the
    wrong correction — it will drive t UP (harder mask, what its zeroth-order
    ranking already preferred) and lose PESQ with only two knobs in hand.

Protocol identical to `adapt_budget.py` (same 150 reverberant clips, same rng
draw order, paired), trust region on, plus a truth-steered control: the same
two knobs driven by an oracle grid pick, bounding what honest steering of
this family could achieve.

    python two_knob_adapt.py --clips 150
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

CPS = [0, 5, 10, 20, 40]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictor", type=Path,
                    default=HERE / "artifacts" / "pesq_predictor_sig.pt")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--clips", type=int, default=150)
    ap.add_argument("--seconds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05,
                    help="two scalars need a larger step than 49k weights")
    ap.add_argument("--rt60-lo", type=float, default=0.2)
    ap.add_argument("--rt60-hi", type=float, default=0.6)
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--sisnr-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "two_knob.npz")
    args = ap.parse_args()

    torch.manual_seed(0); torch.set_num_threads(8)
    from pesq import pesq as pesq_fn

    ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
    metric = PesqPredictor(ck["dim"], ck.get("head", "lsig")).eval()
    metric.load_state_dict(ck["model"])
    for p in metric.parameters():
        p.requires_grad_(False)

    trunk, head = build_split()
    sp = torch.load(args.split, map_location="cpu", weights_only=False)
    trunk.load_state_dict(sp["trunk"]); head.load_state_dict(sp["head"]); trunk.eval()

    pairs = load_test_pairs(args.clips, args.seconds)
    rng = np.random.default_rng(args.seed)
    win = torch.from_numpy(dsp.hann_periodic()).float()

    n, nc = len(pairs), len(CPS)
    g_crit = np.full((n, nc), np.nan)        # critic-steered gain per checkpoint
    t_traj = np.full((n, nc), np.nan)        # where the critic drives t
    c_traj = np.full((n, nc), np.nan)
    g_preset = np.full(n, np.nan)            # fixed (t=0.5, c=0.5) for reference
    g_oracle = np.full(n, np.nan)            # best of a small (t, c) grid

    TGRID = [(t, c) for t in (0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6)
             for c in (0.0, 0.25, 0.5)]

    print(f"two-knob adaptation: {n} clips, m = sigmoid(t*z + c), "
          f"{args.steps} steps, lr {args.lr}\n")

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
            z = (head.conv(h))[0]                       # frozen logits [F, T]
        noisy_t = torch.from_numpy(no_rev).float()
        ref = cl_rev.astype(np.float32)

        def synth(full):
            y = torch.istft((Xc * full)[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                            win_length=dsp.WIN_LENGTH, window=win,
                            length=len(no_rev), center=True)[0]
            return y / nf

        def score(est) -> float:
            try:
                return pesq_fn(SR, ref, est.numpy().astype(np.float32), "wb")
            except Exception:
                return np.nan

        def mask_of(t, c):
            m = torch.sigmoid(t * z + c)
            return torch.cat([m, torch.zeros(1, m.shape[1])], 0)

        base = score(synth(mask_of(torch.tensor(1.0), torch.tensor(0.0))))
        if np.isnan(base):
            continue

        # references within the same family
        g_preset[i] = score(synth(mask_of(torch.tensor(0.5), torch.tensor(0.5)))) - base
        g_oracle[i] = max(score(synth(mask_of(torch.tensor(t), torch.tensor(c)))) - base
                          for t, c in TGRID)

        # ---- the critic steers the two knobs ----
        t = torch.tensor(1.0, requires_grad=True)
        c = torch.tensor(0.0, requires_grad=True)
        opt = torch.optim.Adam([t, c], lr=args.lr)

        def rec(k):
            with torch.no_grad():
                g_crit[i, k] = score(synth(mask_of(t, c))) - base
                t_traj[i, k] = float(t); c_traj[i, k] = float(c)

        rec(0)
        for step in range(1, args.steps + 1):
            full = mask_of(t, c)
            loss = -metric(nmag_c[None], compress(Xt * full)[None])[0]
            s = si_snr(synth(full), noisy_t)
            if float(s) < args.sisnr_floor:
                loss = loss + args.sisnr_weight * (args.sisnr_floor - s)
            opt.zero_grad(); loss.backward(); opt.step()
            if step in CPS:
                rec(CPS.index(step))

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1:3d}/{n}  ({el:.0f}s, eta {el/(i+1)*(n-i-1)/60:.1f} min)",
                  flush=True)

    ok = ~np.isnan(g_crit).any(1) & ~np.isnan(g_preset) & ~np.isnan(g_oracle)
    g_crit, t_traj, c_traj = g_crit[ok], t_traj[ok], c_traj[ok]
    g_preset, g_oracle = g_preset[ok], g_oracle[ok]
    n = int(ok.sum())

    def stat(g):
        se = g.std(ddof=1) / np.sqrt(len(g))
        return f"{g.mean():+8.3f} {1.96*se:8.3f} {(g>0).mean()*100:8.0f}%"

    print(f"\n{n} clips. Within the SAME 2-parameter family:")
    print(f"{'':>26} {'gain':>8} {'95% CI':>8} {'improved':>9}")
    print(f"{'fixed (t=0.5, c=0.5)':>26} {stat(g_preset)}")
    print(f"{'per-clip (t,c) oracle':>26} {stat(g_oracle)}")
    for k, s in enumerate(CPS):
        print(f"{f'critic-steered, step {s}':>26} {stat(g_crit[:, k])}")

    print(f"\nwhere the critic drives the knobs (truth wants t ~ 0.5, c > 0):")
    print(f"{'step':>5} {'mean t':>8} {'t<1 (%)':>9} {'mean c':>8}")
    for k, s in enumerate(CPS):
        print(f"{s:>5} {t_traj[:,k].mean():8.3f} {(t_traj[:,k]<1).mean()*100:8.0f}% "
              f"{c_traj[:,k].mean():+8.3f}")

    np.savez_compressed(args.out, g_crit=g_crit, t=t_traj, c=c_traj,
                        g_preset=g_preset, g_oracle=g_oracle, steps=np.array(CPS))
    verdict = ("DIRECTION: the critic turns the right knobs the wrong way"
               if t_traj[:, -1].mean() > 1.0 or g_crit[:, -1].mean() < 0
               else "capacity may have mattered after all — inspect the trajectories")
    print(f"\n=> {verdict}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
