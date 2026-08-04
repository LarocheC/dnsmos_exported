#!/usr/bin/env python3
"""How many adaptation steps are safe? The budget curve for a frozen critic.

MetricGAN's premise is that a static learned surrogate "is easily fooled ...
the gradient provided by Quality-Net is only accurate for the first few
learning iterations." Their fix is to retrain the surrogate alternately with
the generator. **On an MCU that fix is unavailable**: the predictor is frozen
in flash, there are no labels at the edge, and there is no room for a second
training loop. So the qualitative warning has to become a quantitative budget.

This script measures, per adaptation step, on reverberant VoiceBank-DEMAND:

  * **true PESQ**   — against the reverberant clean reference, the only number
                      that actually matters
  * **believed**    — what the on-device predictor claims
  * **optimism**    — believed minus true

Optimism is decomposed, which matters for the diagnosis:

  * optimism at step 0 is the predictor's **calibration error** — it is wrong
    about audio it never touched, and no optimizer is responsible for that
  * the *rise* in optimism from step 0 is the **exploitation** — the part the
    adaptation loop actively created

Only the second is Goodhart. Reporting the sum, as the earlier 7-clip runs
did, conflates a fixable calibration problem with an inherent one.

Three things come out of the curve:

1. Whether mean true PESQ peaks and then declines. If it does, a fixed step
   budget is a deployable stopping rule and its value is the argmax.
2. The gap between a **fixed** budget and **oracle** per-clip stopping. That
   gap bounds what any cleverer stopping rule could ever buy.
3. The spread of per-clip optima. A tight spread means one global budget
   serves everyone; a wide one means fixed budgets are leaving most of the
   gain on the table.

Scale note: the earlier runs used 7 clips and one rt60, which cannot resolve
an effect of the size seen (+0.12 +/- 0.08). Default here is 100 clips with
rt60 drawn per clip, and gains are reported paired with 95% CIs.

    python adapt_budget.py --clips 100 --steps 60
    python adapt_budget.py --clips 100 --steps 60 --no-sisnr   # ablate the trust region
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


def checkpoints(max_steps: int) -> list[int]:
    """Dense early (where the action is), sparse late."""
    c = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    return [s for s in c if s <= max_steps] + ([max_steps] if max_steps not in c else [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictor", type=Path,
                    default=HERE / "artifacts" / "pesq_predictor_sig.pt")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--clips", type=int, default=100)
    ap.add_argument("--seconds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rt60-lo", type=float, default=0.2)
    ap.add_argument("--rt60-hi", type=float, default=0.6)
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--sisnr-weight", type=float, default=0.5)
    ap.add_argument("--no-sisnr", action="store_true",
                    help="drop the SI-SNR trust region, isolating the metric's own behaviour")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "adapt_budget.npz")
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
    cps = checkpoints(args.steps)
    idx = {s: k for k, s in enumerate(cps)}

    n, nc = len(pairs), len(cps)
    true = np.full((n, nc), np.nan)          # real PESQ at each checkpoint
    bel = np.full((n, nc), np.nan)           # what the predictor claimed
    snr = np.full((n, nc), np.nan)           # SI-SNR vs the reverberant input
    rt60s = np.zeros(n)

    print(f"budget curve: {n} clips x {args.seconds}s, up to {args.steps} steps, "
          f"rt60 U[{args.rt60_lo},{args.rt60_hi}], "
          f"trust region {'OFF' if args.no_sisnr else f'SI-SNR floor {args.sisnr_floor} dB'}")
    print(f"head={ck.get('head','lsig')}  lr={args.lr}\n")

    t0 = time.time()
    for i, (cl, no) in enumerate(pairs):
        rt60 = float(rng.uniform(args.rt60_lo, args.rt60_hi))
        rt60s[i] = rt60
        rir = make_rir(rt60, rng)
        cl_rev = reverberate(cl, rir)
        no_rev = cl_rev + (no - cl)                       # VBD's own noise, undereverbed
        no_s, nf = dsp.rms_normalize(no_rev)

        X = dsp.stft(no_s)
        Xt = torch.from_numpy(np.abs(X)).float()
        Xc = torch.from_numpy(X).to(torch.complex64)
        nmag_c = compress(Xt)
        mag = Xt[: trunk.n_features][None]
        with torch.no_grad():
            pad = mag[..., :1].repeat(1, 1, trunk.context_l)
            h = trunk(torch.cat([pad, mag], dim=-1))

        def synth(full):
            y = torch.istft((Xc * full)[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                            win_length=dsp.WIN_LENGTH, window=win,
                            length=len(no_rev), center=True)[0]
            return y / nf

        ref = cl_rev.astype(np.float32)
        noisy_t = torch.from_numpy(no_rev).float()

        W = head.conv.weight.detach()[:, :, 0].clone().requires_grad_(True)
        b = head.conv.bias.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=args.lr)

        def record(step: int) -> None:
            with torch.no_grad():
                m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
                full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
                est = synth(full)
                k = idx[step]
                bel[i, k] = float(metric(nmag_c[None], compress(Xt * full)[None])[0]) * 3.5 + 1.0
                snr[i, k] = float(si_snr(est, noisy_t))
                try:
                    true[i, k] = pesq_fn(SR, ref, est.numpy().astype(np.float32), "wb")
                except Exception as e:                    # NoUtterances etc. — keep the run
                    print(f"    clip {i} step {step}: PESQ failed ({type(e).__name__}), "
                          f"left as NaN", flush=True)

        record(0)
        for step in range(1, args.steps + 1):
            m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
            full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
            loss = -metric(nmag_c[None], compress(Xt * full)[None])[0]
            if not args.no_sisnr:
                s = si_snr(synth(full), noisy_t)
                if float(s) < args.sisnr_floor:
                    loss = loss + args.sisnr_weight * (args.sisnr_floor - s)
            opt.zero_grad(); loss.backward(); opt.step()
            if step in idx:
                record(step)

        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t0
            print(f"  {i+1:3d}/{n} clips  ({el:.0f}s, {el/(i+1):.1f}s/clip, "
                  f"eta {el/(i+1)*(n-i-1)/60:.1f} min)", flush=True)

    np.savez_compressed(args.out, true=true, believed=bel, sisnr=snr,
                        steps=np.array(cps), rt60=rt60s)

    # ---------------- aggregate ----------------
    # every statistic here is paired across steps, so a clip is only usable if
    # its whole trajectory scored; partial rows would bias the curve's shape.
    ok = ~np.isnan(true).any(axis=1)
    if not ok.all():
        print(f"\ndropped {int((~ok).sum())} clip(s) with a failed PESQ call; "
              f"{int(ok.sum())} complete trajectories remain")
    true, bel, snr = true[ok], bel[ok], snr[ok]

    base = true[:, 0]
    gain = true - base[:, None]
    opt0 = bel[:, 0] - true[:, 0]                        # calibration error at step 0
    optim = bel - true                                   # total optimism
    excess = optim - opt0[:, None]                       # exploitation created by adapting

    print(f"\nbaseline (reverberant, unadapted): PESQ {base.mean():.3f} "
          f"[{base.min():.2f}..{base.max():.2f}]")
    print(f"predictor calibration error at step 0: {opt0.mean():+.3f} PESQ "
          f"(this is not exploitation)\n")

    print(f"{'step':>5} {'true PESQ':>10} {'gain':>8} {'95% CI':>9} {'impr':>6} "
          f"{'believed':>9} {'optimism':>9} {'excess':>8} {'SI-SNR':>8}")
    for k, s in enumerate(cps):
        g = gain[:, k]
        se = g.std(ddof=1) / np.sqrt(len(g))
        print(f"{s:>5} {true[:,k].mean():10.3f} {g.mean():+8.3f} {1.96*se:9.3f} "
              f"{(g>0).mean()*100:5.0f}% {bel[:,k].mean():9.3f} "
              f"{optim[:,k].mean():+9.3f} {excess[:,k].mean():+8.3f} {snr[:,k].mean():8.2f}")

    # fixed budget vs oracle stopping
    mean_curve = gain.mean(axis=0)
    kbest = int(np.argmax(mean_curve))
    per_clip_best = gain.max(axis=1)
    per_clip_arg = np.array([cps[j] for j in gain.argmax(axis=1)])
    gf = gain[:, kbest]
    se_f = gf.std(ddof=1) / np.sqrt(len(gf))

    print(f"\nbest FIXED budget: {cps[kbest]} steps -> {gf.mean():+.3f} "
          f"+/- {1.96*se_f:.3f} PESQ, improved {(gf>0).mean()*100:.0f}% of clips")
    print(f"ORACLE per-clip stopping: {per_clip_best.mean():+.3f} PESQ "
          f"(upper bound for any stopping rule)")
    print(f"headroom a better rule could buy: "
          f"{per_clip_best.mean()-gf.mean():+.3f} PESQ")
    q = np.percentile(per_clip_arg, [10, 25, 50, 75, 90])
    print(f"per-clip optimal step: median {int(q[2])}, "
          f"IQR {int(q[1])}-{int(q[3])}, 10-90% {int(q[0])}-{int(q[4])}")

    peaked = mean_curve[-1] < mean_curve[kbest] - 1.96 * se_f
    print(f"\n=> mean true PESQ {'PEAKS then declines' if peaked else 'does not resolve a peak'}"
          f"; exploitation at max budget {excess[:,-1].mean():+.3f} PESQ")
    print(f"saved trajectories to {args.out}")


if __name__ == "__main__":
    main()
