#!/usr/bin/env python3
"""Does the PESQ predictor survive being used as a gradient source?

This is the same probe that condemned the DNSMOS spectral adapter, applied to
the metric that was designed to resist it. Two numbers matter, in this order:

1. **Optimism.** Adapt the mask head against the predictor, then score the
   result with *real* PESQ against the clean reference. The gap between what
   the predictor believed and what PESQ actually says is the exploitation. The
   DNSMOS adapter scored +2.24 MOS of optimism here.
2. **True gain.** Real PESQ before vs after adaptation. A metric that is honest
   but useless is no better than one that is dishonest.

Unlike the DNSMOS runs this evaluation uses VoiceBank-DEMAND *test* utterances
with their clean references, so the verdict is measured against ground truth
rather than against another model's opinion.

Note on expectations: ConvFSENet was already trained against PESQ via
metric-GAN, so it starts near its PESQ optimum and the headroom is small by
construction. A modest true gain with low optimism is the good outcome here;
a large apparent gain with large optimism is the failure this script exists to
catch.

    python pesq_adapt_eval.py --clips 8 --steps 40
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import dsp  # noqa: E402
from convfsenet_arch import build_split  # noqa: E402
from pesq_predictor import PesqPredictor, VBD, compress, norm_pesq, SR  # noqa: E402

from adapter_adapt_eval import si_snr  # noqa: E402


def load_test_pairs(n: int, seconds: int, seed: int = 0):
    import pyarrow.parquet as pq
    import soundfile as sf

    tbl = pq.ParquetFile(str(VBD / "test-00000-of-00001.parquet")).read()
    rng = np.random.default_rng(seed)
    seg = seconds * SR
    out = []
    for i in rng.permutation(len(tbl)):
        if len(out) >= n:
            break
        row = tbl.slice(int(i), 1).to_pylist()[0]
        cl, _ = sf.read(io.BytesIO(row["clean"]["bytes"]), dtype="float32")
        no, _ = sf.read(io.BytesIO(row["noisy"]["bytes"]), dtype="float32")
        m = min(len(cl), len(no))
        if m < seg:
            continue
        s = (m - seg) // 2
        out.append((cl[s:s + seg].astype(np.float64), no[s:s + seg].astype(np.float64)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictor", type=Path, default=HERE / "artifacts" / "pesq_predictor.pt")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--sisnr-weight", type=float, default=0.5)
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
    win = torch.from_numpy(dsp.hann_periodic()).float()
    print(f"VBD *test* utterances with clean references, {args.steps} steps, "
          f"real PESQ as ground truth\n")
    print(f"{'clip':>4} {'PESQ base':>10} {'PESQ adapt':>11} {'d':>7} "
          f"{'predictor believed':>19} {'optimism':>9}")

    gains, optimism = [], []
    for i, (cl, no) in enumerate(pairs):
        # ConvFSENet is trained on RMS-normalized input; skipping this does not
        # degrade the mask, it destroys it (PESQ 1.10 vs 3.43). Output is
        # divided by the same factor before scoring.
        no_s, nf = dsp.rms_normalize(no)
        X = dsp.stft(no_s)
        Xt = torch.from_numpy(np.abs(X)).float()
        Xc = torch.from_numpy(X).to(torch.complex64)
        nmag_c = compress(Xt)                                     # predictor channel 0
        mag = Xt[: trunk.n_features][None]
        with torch.no_grad():
            pad = mag[..., :1].repeat(1, 1, trunk.context_l)
            h = trunk(torch.cat([pad, mag], dim=-1))

        def synth(full: torch.Tensor) -> torch.Tensor:
            y = torch.istft((Xc * full)[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                            win_length=dsp.WIN_LENGTH, window=win,
                            length=len(no), center=True)[0]
            return y / nf                                   # restore original level

        with torch.no_grad():
            m0 = torch.sigmoid(head.conv(h))[0]
            full0 = torch.cat([m0, torch.zeros(1, m0.shape[1])], 0)
            base_pesq = pesq_fn(SR, cl.astype(np.float32),
                                synth(full0).numpy().astype(np.float32), "wb")

        W = head.conv.weight.detach()[:, :, 0].clone().requires_grad_(True)
        b = head.conv.bias.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=args.lr)
        noisy_t = torch.from_numpy(no).float()   # unscaled, for the SI-SNR anchor
        for _ in range(args.steps):
            m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
            full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
            # the whole point: the metric reads |X|*mask directly, no ISTFT
            score = metric(nmag_c[None], compress(Xt * full)[None])[0]
            loss = -score
            est = synth(full)
            s = si_snr(est, noisy_t)
            if float(s) < args.sisnr_floor:
                loss = loss + args.sisnr_weight * (args.sisnr_floor - s)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
            full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
            believed_n = float(metric(nmag_c[None], compress(Xt * full)[None])[0])
            adapted = synth(full).numpy().astype(np.float32)
            true_pesq = pesq_fn(SR, cl.astype(np.float32), adapted, "wb")
        believed = believed_n * 3.5 + 1.0                          # de-normalize
        gains.append(true_pesq - base_pesq)
        optimism.append(believed - true_pesq)
        print(f"{i:>4} {base_pesq:10.3f} {true_pesq:11.3f} {gains[-1]:+7.3f} "
              f"{believed:19.3f} {optimism[-1]:+9.3f}")

    g, o = np.array(gains), np.array(optimism)
    n = len(g)
    se_g = g.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"\nmean true PESQ gain: {g.mean():+.3f} +/- {1.96*se_g:.3f} (95% CI), "
          f"improved on {int((g>0).sum())}/{n}")
    print(f"mean optimism:       {o.mean():+.3f} PESQ   "
          f"(DNSMOS spectral adapter, same probe: +2.24 MOS)")
    if o.mean() < 0.5:
        print("=> the metric stayed honest under optimization.")
    else:
        print("=> EXPLOITED: the predictor believes far more than it delivered.")


if __name__ == "__main__":
    main()
