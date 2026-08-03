#!/usr/bin/env python3
"""Does the spectral adapter work as a GRADIENT source, not just a correlated scorer?

`spectral_adapter.py` reaches OVRL pearson 0.93 / spearman 0.89 against true
DNSMOS. This repo has been burned three times by exactly that kind of number:
an agreement metric can improve while the thing you care about degrades
(`crop_study.py`'s "cosine does not rank steering"; the int8-coverage recipe;
the adapter's own feature-MSE variant). Correlation is necessary, not
sufficient — the adapter's whole purpose is to *steer* the mask head.

So this script runs the honest test: adapt the enhancer's mask head with each
candidate gradient source, then score the result with the OFFICIAL 9.01 s
DNSMOS. Paired, same clips, same steps, same anchor:

  waveform path (today)     mask -> Y=X*mask -> ISTFT -> DNSMOS loss graph
  spectral path (proposed)  mask -> |Y|=|X|*mask -> adapter -> frozen body

Both are run under torch autograd here. That is deliberate: this measures
whether the *architecture* steers, independent of the hand-written backward
that a deployment would use. The waveform path's hand-written backward is
already verified to <1e-4 against autograd, so autograd is a fair stand-in.

The SI-SNR hinge is applied in both, against the noisy input, exactly as the
deployed loop does — without it any no-reference metric rewards artefacts.

    python adapter_adapt_eval.py --clips 8 --steps 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import dsp  # noqa: E402
from convfsenet_arch import build_split  # noqa: E402
from dnsmos_trainable.constants import OFFICIAL_CONFIG as CFG  # noqa: E402
from dnsmos_trainable.model import DnsmosModel  # noqa: E402
from spectral_adapter import SpectralAdapter, EPS  # noqa: E402

SR = 16000


def si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = (est * ref).sum() / (ref * ref).sum().clamp_min(1e-12)
    tgt = alpha * ref
    noise = est - tgt
    return 10.0 * torch.log10((tgt.pow(2).sum() / noise.pow(2).sum().clamp_min(1e-12)).clamp_min(1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=HERE / "artifacts" / "spectral_adapter_e100.pt")
    ap.add_argument("--weights", type=Path, default=HERE.parents[1] / "models" / "dnsmos_transplanted.pt")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--cache", type=Path, default=HERE / "artifacts" / "vbd_cache.npz")
    ap.add_argument("--clips", type=int, default=8, help="held-out clips (from the tail of the cache)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--sisnr-weight", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(8)

    metric = DnsmosModel().eval()
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    metric.load_state_dict(sd.get("model", sd), strict=False)
    for p in metric.parameters():
        p.requires_grad_(False)

    ck = torch.load(args.adapter, map_location="cpu", weights_only=False)
    adapter = SpectralAdapter(ck["n_frames"], ck["n_bins"], ck["width"]).eval()
    adapter.load_state_dict(ck["adapter"])
    for p in adapter.parameters():
        p.requires_grad_(False)

    trunk, head = build_split()
    split = torch.load(args.split, map_location="cpu", weights_only=False)
    trunk.load_state_dict(split["trunk"]); head.load_state_dict(split["head"])
    trunk.eval()
    F_BINS = trunk.n_features

    clips = np.load(args.cache)["clips"]
    test = clips[-args.clips:]

    def official(wav_t: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return metric(wav_t[None])[1][0].numpy()

    def run(clip: np.ndarray, source: str):
        x = clip[: CFG.input_len].astype(np.float32)
        Xc = dsp.stft(x.astype(np.float64))
        Xt = torch.from_numpy(np.abs(Xc)).float()                 # [257, T]
        mag = Xt[:F_BINS][None]                                   # [1, 256, T]
        with torch.no_grad():
            pad = mag[..., :1].repeat(1, 1, trunk.context_l)
            h = trunk(torch.cat([pad, mag], dim=-1))              # [1, 192, T]
        W = head.conv.weight.detach()[:, :, 0].clone().requires_grad_(True)
        b = head.conv.bias.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=args.lr)
        noisy_t = torch.from_numpy(x)

        for _ in range(args.steps):
            mask = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])   # [256, T]
            full = torch.cat([mask, torch.zeros(1, mask.shape[1])], 0)  # Nyquist gain 0
            if source == "spectral":
                mag_y = (Xt * full).clamp_min(1e-9)
                logp = torch.log10((mag_y ** 2).clamp_min(EPS))[None, None]   # [1,1,257,T]
                score = metric.poly(metric.body(adapter(logp)))[0]
                # SI-SNR still needs the waveform, but only for the anchor —
                # the metric gradient itself never leaves the spectral domain.
                Yc = torch.from_numpy(Xc).to(torch.complex64) * full
                est = torch.istft(Yc[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                                  win_length=dsp.WIN_LENGTH,
                                  window=torch.from_numpy(dsp.hann_periodic()).float(),
                                  length=len(x), center=True)[0]
            else:
                Yc = torch.from_numpy(Xc).to(torch.complex64) * full
                est = torch.istft(Yc[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                                  win_length=dsp.WIN_LENGTH,
                                  window=torch.from_numpy(dsp.hann_periodic()).float(),
                                  length=len(x), center=True)[0]
                score = metric(est[None])[1][0]
            loss = -score[2]
            s = si_snr(est, noisy_t)
            if float(s) < args.sisnr_floor:
                loss = loss + args.sisnr_weight * (args.sisnr_floor - s)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            mask = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
            full = torch.cat([mask, torch.zeros(1, mask.shape[1])], 0)
            Yc = torch.from_numpy(Xc).to(torch.complex64) * full
            est = torch.istft(Yc[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                              win_length=dsp.WIN_LENGTH,
                              window=torch.from_numpy(dsp.hann_periodic()).float(),
                              length=len(x), center=True)[0]
        return official(est), float(si_snr(est, noisy_t))

    print(f"paired adaptation, {args.clips} held-out clips, {args.steps} steps, "
          f"scored with the OFFICIAL 9.01 s DNSMOS\n")
    print(f"{'clip':>4} {'baseline':>9} {'waveform':>9} {'spectral':>9}   "
          f"{'d_wave':>7} {'d_spec':>7}")
    dw, ds = [], []
    for i, clip in enumerate(test):
        x = clip[: CFG.input_len].astype(np.float32)
        Xc = dsp.stft(x.astype(np.float64))
        Xt = torch.from_numpy(np.abs(Xc)).float()
        mag = Xt[:F_BINS][None]
        with torch.no_grad():
            pad = mag[..., :1].repeat(1, 1, trunk.context_l)
            h = trunk(torch.cat([pad, mag], dim=-1))
            m0 = torch.sigmoid(head.conv(h))[0]
            full0 = torch.cat([m0, torch.zeros(1, m0.shape[1])], 0)
            Y0 = torch.from_numpy(Xc).to(torch.complex64) * full0
            est0 = torch.istft(Y0[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                               win_length=dsp.WIN_LENGTH,
                               window=torch.from_numpy(dsp.hann_periodic()).float(),
                               length=len(x), center=True)[0]
        base = official(est0)[2]
        mw, _ = run(clip, "waveform")
        ms, _ = run(clip, "spectral")
        dw.append(mw[2] - base); ds.append(ms[2] - base)
        print(f"{i:>4} {base:9.3f} {mw[2]:9.3f} {ms[2]:9.3f}   "
              f"{dw[-1]:+7.3f} {ds[-1]:+7.3f}")

    dw, ds = np.array(dw), np.array(ds)
    n = len(dw)
    diff = ds - dw
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"\nmean dOVRL   waveform {dw.mean():+.3f}   spectral {ds.mean():+.3f}")
    print(f"paired diff (spectral - waveform): {diff.mean():+.3f} "
          f"+/- {1.96*se:.3f} (95% CI), spectral better on {int((diff>0).sum())}/{n} clips")
    if n > 1 and abs(diff.mean()) < 1.96 * se:
        print("=> difference is NOT resolved by this sample size; "
              "increase --clips before concluding either way.")


if __name__ == "__main__":
    main()
