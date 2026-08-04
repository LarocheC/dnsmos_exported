#!/usr/bin/env python3
"""Generate the exploited candidates the predictor saturates on, and label them.

`adapt_gate.py` showed why on-device adaptation stalls at +0.083 PESQ: the
predictor collapses to a constant under optimization (std across clips falls
250x, every clip pinned within 0.05 of the 4.50 ceiling by step 25). It cannot
be its own stopping criterion because it has stopped discriminating.

MetricGAN fixes exactly this with a replay buffer — but *online*, alternating
critic and generator updates, which an MCU cannot do. The same augmentation
works **offline**: run the adaptation loop here, during dataset construction,
capture the masks it produces, label them with real PESQ, and let the predictor
learn what they are actually worth. The critic is still frozen at deployment;
only the anti-exploitation training moves to before the flash.

`candidate_masks` in `pesq_predictor.py` already tries to cover this ground with
hand-designed failure modes — over-suppression, spectral holes, band damage.
The premise here is that hand-designed artefacts are not what a gradient
actually finds, so the examples must come from the optimizer itself.

Two design choices worth stating:

* **Anechoic only.** Exploits are generated on VoiceBank-DEMAND *train*
  utterances with no reverberation. Reverb is the held-out domain shift that
  the whole adaptation experiment exists to recover, and filling that hole
  would turn the evaluation into a test of generalization instead. If exploits
  found on anechoic audio inoculate the metric against exploitation under
  reverb, that is a stronger result, not a weaker one.
* **Both trust-region settings.** Deployment runs with the SI-SNR floor, but
  the unconstrained loop finds more extreme artefacts. Including both widens
  the covered region.

Masks come from the normalized pipeline the enhancer requires; features and
PESQ labels are then computed on the *unnormalized* spectrum, exactly as
`pesq_predictor.build` does, so the two halves of the dataset stay homogeneous.

    python pesq_adv_dataset.py --utts 200
    python pesq_predictor.py train --data artifacts/pesq_dataset_adv.npz \\
        --head sigmoid --out artifacts/pesq_predictor_adv.pt
"""

from __future__ import annotations

import argparse
import io
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
from pesq_predictor import PesqPredictor, VBD, compress, norm_pesq, SR  # noqa: E402

CAPTURE = (1, 3, 10, 30, 100)          # where the budget curve says the action is


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictor", type=Path,
                    default=HERE / "artifacts" / "pesq_predictor_sig.pt",
                    help="the metric to exploit; its failures are what we label")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--base", type=Path, default=HERE / "artifacts" / "pesq_dataset.npz",
                    help="original candidates; the adversarial ones are appended")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "pesq_dataset_adv.npz")
    ap.add_argument("--utts", type=int, default=200)
    ap.add_argument("--seconds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--sisnr-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    from pesq import pesq as pesq_fn

    torch.manual_seed(0); torch.set_num_threads(8)

    ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
    metric = PesqPredictor(ck["dim"], ck.get("head", "lsig")).eval()
    metric.load_state_dict(ck["model"])
    for p in metric.parameters():
        p.requires_grad_(False)

    trunk, head = build_split()
    sp = torch.load(args.split, map_location="cpu", weights_only=False)
    trunk.load_state_dict(sp["trunk"]); head.load_state_dict(sp["head"]); trunk.eval()

    win = torch.from_numpy(dsp.hann_periodic()).float()
    seg = args.seconds * SR
    rng = np.random.default_rng(args.seed)
    shards = sorted(VBD.glob("train-*.parquet"))

    feats, labels, believed, meta = [], [], [], []
    kept = skipped = 0
    t0 = time.time()

    for shard in shards:
        tbl = pq.ParquetFile(str(shard)).read()
        idx = rng.permutation(len(tbl))[: args.utts // len(shards) + 2]
        for i in idx:
            if kept >= args.utts:
                break
            row = tbl.slice(int(i), 1).to_pylist()[0]
            cl, _ = sf.read(io.BytesIO(row["clean"]["bytes"]), dtype="float32")
            no, _ = sf.read(io.BytesIO(row["noisy"]["bytes"]), dtype="float32")
            n = min(len(cl), len(no))
            if n < seg:
                continue
            s = int(rng.integers(0, n - seg + 1))
            cl = cl[s:s + seg].astype(np.float64)
            no = no[s:s + seg].astype(np.float64)

            X = dsp.stft(no)                      # unnormalized: features + labels
            nmag = np.abs(X)
            no_s, nf = dsp.rms_normalize(no)
            Xs = dsp.stft(no_s)                   # normalized: what the enhancer needs
            Xst = torch.from_numpy(np.abs(Xs)).float()
            nmag_c = compress(Xst)
            mag = Xst[: trunk.n_features][None]
            with torch.no_grad():
                pad = mag[..., :1].repeat(1, 1, trunk.context_l)
                h = trunk(torch.cat([pad, mag], dim=-1))
            Xsc = torch.from_numpy(Xs).to(torch.complex64)
            noisy_t = torch.from_numpy(no_s).float()

            for use_tr in (True, False):
                W = head.conv.weight.detach()[:, :, 0].clone().requires_grad_(True)
                b = head.conv.bias.detach().clone().requires_grad_(True)
                opt = torch.optim.Adam([W, b], lr=args.lr)
                for step in range(1, max(CAPTURE) + 1):
                    m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
                    full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
                    score = metric(nmag_c[None], compress(Xst * full)[None])[0]
                    loss = -score
                    if use_tr:
                        y = torch.istft((Xsc * full)[None], n_fft=dsp.N_FFT,
                                        hop_length=dsp.HOP, win_length=dsp.WIN_LENGTH,
                                        window=win, length=seg, center=True)[0]
                        sn = si_snr(y, noisy_t)
                        if float(sn) < args.sisnr_floor:
                            loss = loss + args.sisnr_weight * (args.sisnr_floor - sn)
                    opt.zero_grad(); loss.backward(); opt.step()

                    if step in CAPTURE:
                        with torch.no_grad():
                            mm = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
                            fl = torch.cat([mm, torch.zeros(1, mm.shape[1])], 0).numpy()
                            bel = float(metric(nmag_c[None],
                                               compress(Xst * torch.from_numpy(fl))[None])[0])
                        y = dsp.istft(X * fl, length=seg)      # unnormalized, as in build()
                        try:
                            p = pesq_fn(SR, cl.astype(np.float32), y.astype(np.float32), "wb")
                        except Exception:
                            skipped += 1
                            continue
                        feats.append(np.stack([compress(nmag),
                                               compress(nmag * fl)]).astype(np.float16))
                        labels.append(norm_pesq(p))
                        believed.append(bel * 3.5 + 1.0)
                        meta.append((step, int(use_tr)))
            kept += 1
            if kept % 20 == 0:
                el = time.time() - t0
                print(f"  {kept}/{args.utts} utts, {len(labels)} adversarial candidates "
                      f"({el:.0f}s, eta {el/kept*(args.utts-kept)/60:.1f} min)", flush=True)
        if kept >= args.utts:
            break

    Fa = np.stack(feats); La = np.array(labels, np.float32)
    Be = np.array(believed, np.float32)
    Me = np.array(meta, np.int16)
    true_p = La * 3.5 + 1.0

    print(f"\n{len(La)} adversarial candidates, {skipped} skipped")
    print(f"  true PESQ  {true_p.min():.2f}..{true_p.max():.2f} (mean {true_p.mean():.2f})")
    print(f"  v1 believed {Be.min():.2f}..{Be.max():.2f} (mean {Be.mean():.2f})")
    print(f"  => mean exploitation being labelled away: {(Be-true_p).mean():+.2f} PESQ")
    for st in CAPTURE:
        s = Me[:, 0] == st
        if s.any():
            print(f"     step {st:>3}: true {true_p[s].mean():.2f}  "
                  f"believed {Be[s].mean():.2f}  gap {(Be[s]-true_p[s]).mean():+.2f}")

    z = np.load(args.base)
    Fb, Lb = z["feats"], z["labels"]
    print(f"\nbase dataset {Fb.shape[0]} candidates; combined "
          f"{Fb.shape[0] + Fa.shape[0]}")
    np.savez_compressed(args.out,
                        feats=np.concatenate([Fb, Fa]),
                        labels=np.concatenate([Lb, La]),
                        is_adv=np.concatenate([np.zeros(len(Lb), np.int8),
                                               np.ones(len(La), np.int8)]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
