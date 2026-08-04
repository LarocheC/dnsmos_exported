#!/usr/bin/env python3
"""Does on-device adaptation pay off under a domain shift the enhancer never saw?

The PESQ steering test on VoiceBank-DEMAND came back negative, and the likely
reason is that it was the wrong test: ConvFSENet is a PESQ metric-GAN trained
on VBD, so it starts near its PESQ optimum (baseline 3.3-3.9) and an optimizer
can only walk downhill. That measures headroom, not adaptation.

The premise of on-device adaptation is the opposite case — conditions the model
was never trained for. This script builds one: **reverberant** noisy speech,
which VBD contains none of.

Evaluation target matters here. A denoiser fed reverberant input cannot be
expected to dereverberate, so scoring against the anechoic clean signal would
charge it for a task it was never given. The reference is therefore the
**reverberant clean** signal (reverb kept, noise removed) — the standard
convention for denoising under reverberation, and the best output the enhancer
could legitimately produce:

    clean_rev = clean * RIR              <- PESQ reference
    noisy_rev = clean_rev + noise        <- enhancer input

RIRs are synthetic (exponentially-decaying noise with a direct path), which is
enough to create the distribution shift; the point is that the enhancer has
never seen reverberant input, not that the reverb is acoustically exact.

Reported, paired per utterance:
  * baseline PESQ  — the enhancer as shipped, on reverberant input
  * adapted PESQ   — after adapting the mask head against the on-device metric
  * optimism       — what the metric believed minus what PESQ delivered

A positive gain here, with low optimism, is the result that would justify the
whole on-device adaptation loop. A positive gain with high optimism means the
metric is being gamed and the win is illusory.

    python reverb_adapt_eval.py --clips 8 --steps 40 --rt60 0.4
"""

from __future__ import annotations

import argparse
import sys
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


def make_rir(rt60: float, rng: np.random.Generator, sr: int = SR,
             direct_gain: float = 1.0) -> np.ndarray:
    """Synthetic room impulse response: direct path + exponentially-decaying tail."""
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    decay = np.exp(-6.908 * t / max(rt60, 1e-3))          # -60 dB at rt60
    tail = rng.standard_normal(n) * decay
    tail[0] = direct_gain
    return tail / np.sqrt(np.sum(tail ** 2))


def reverberate(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    return np.convolve(x, rir)[: len(x)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictor", type=Path, default=HERE / "artifacts" / "pesq_predictor.pt")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rt60", type=float, default=0.4, help="reverberation time (s)")
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--sisnr-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clamp", type=float, default=None, metavar="PESQ",
                    help="cap the metric at this PESQ before differentiating. The "
                         "predictor's LearnableSigmoid(beta=2) can emit up to PESQ 8.0, "
                         "and the optimizer drives it there -- far past the 4.5 the "
                         "metric can actually mean. Clamping makes the gradient vanish "
                         "once the metric claims a score it cannot justify, which is a "
                         "brake on exactly the observed failure. Try --clamp 4.5.")
    args = ap.parse_args()

    torch.manual_seed(0); torch.set_num_threads(8)
    from pesq import pesq as pesq_fn

    ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
    metric = PesqPredictor(ck["dim"]).eval()
    metric.load_state_dict(ck["model"])
    for p in metric.parameters():
        p.requires_grad_(False)

    trunk, head = build_split()
    sp = torch.load(args.split, map_location="cpu", weights_only=False)
    trunk.load_state_dict(sp["trunk"]); head.load_state_dict(sp["head"]); trunk.eval()

    pairs = load_test_pairs(args.clips, args.seconds)
    rng = np.random.default_rng(args.seed)
    win = torch.from_numpy(dsp.hann_periodic()).float()

    print(f"reverberant VBD test (rt60 {args.rt60:.2f} s), reference = reverberant "
          f"CLEAN, {args.steps} steps\n")
    print(f"{'clip':>4} {'anechoic':>9} {'reverb':>8} | {'base':>7} {'adapted':>8} "
          f"{'d':>7} | {'optimism':>9}")

    gains, optim, shifts = [], [], []
    for i, (cl, no) in enumerate(pairs):
        rir = make_rir(args.rt60, rng)
        cl_rev = reverberate(cl, rir)
        noise = no - cl                                   # VBD's own noise
        no_rev = cl_rev + noise                           # reverberant noisy input
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

        ref = cl_rev.astype(np.float32)                   # reverberant clean
        with torch.no_grad():
            m0 = torch.sigmoid(head.conv(h))[0]
            full0 = torch.cat([m0, torch.zeros(1, m0.shape[1])], 0)
            base = pesq_fn(SR, ref, synth(full0).numpy().astype(np.float32), "wb")
        # how far the shift moved the enhancer, for context
        no_a, nfa = dsp.rms_normalize(no)
        Xa = dsp.stft(no_a); Xat = torch.from_numpy(np.abs(Xa)).float()
        with torch.no_grad():
            maga = Xat[: trunk.n_features][None]
            pada = maga[..., :1].repeat(1, 1, trunk.context_l)
            ha = trunk(torch.cat([pada, maga], dim=-1))
            ma = torch.sigmoid(head.conv(ha))[0]
            fa = torch.cat([ma, torch.zeros(1, ma.shape[1])], 0)
            ya = torch.istft((torch.from_numpy(Xa).to(torch.complex64) * fa)[None],
                             n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                             win_length=dsp.WIN_LENGTH, window=win,
                             length=len(no), center=True)[0] / nfa
            anech = pesq_fn(SR, cl.astype(np.float32), ya.numpy().astype(np.float32), "wb")

        W = head.conv.weight.detach()[:, :, 0].clone().requires_grad_(True)
        b = head.conv.bias.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=args.lr)
        noisy_t = torch.from_numpy(no_rev).float()
        for _ in range(args.steps):
            m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
            full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
            score = metric(nmag_c[None], compress(Xt * full)[None])[0]
            if args.clamp is not None:
                score = score.clamp_max((args.clamp - 1.0) / 3.5)
            loss = -score
            est = synth(full)
            s = si_snr(est, noisy_t)
            if float(s) < args.sisnr_floor:
                loss = loss + args.sisnr_weight * (args.sisnr_floor - s)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
            full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
            believed = float(metric(nmag_c[None], compress(Xt * full)[None])[0]) * 3.5 + 1.0
            adapted = pesq_fn(SR, ref, synth(full).numpy().astype(np.float32), "wb")

        gains.append(adapted - base); optim.append(believed - adapted)
        shifts.append(anech - base)
        print(f"{i:>4} {anech:9.3f} {base:8.3f} | {base:7.3f} {adapted:8.3f} "
              f"{gains[-1]:+7.3f} | {optim[-1]:+9.3f}")

    g, o, sh = np.array(gains), np.array(optim), np.array(shifts)
    n = len(g)
    se = g.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"\nthe reverb shift cost the enhancer {sh.mean():.3f} PESQ "
          f"(anechoic vs reverberant baseline) — i.e. headroom now exists")
    print(f"mean adaptation gain: {g.mean():+.3f} +/- {1.96*se:.3f} (95% CI), "
          f"improved on {int((g>0).sum())}/{n}")
    print(f"mean optimism:        {o.mean():+.3f} PESQ")
    verdict = ("adaptation HELPS under domain shift" if g.mean() > 0 and abs(g.mean()) > 1.96*se
               else "no resolved gain")
    print(f"=> {verdict}; metric {'stayed honest' if o.mean() < 0.5 else 'still over-reports'}")


if __name__ == "__main__":
    main()
