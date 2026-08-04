#!/usr/bin/env python3
"""A PESQ predictor on the ENHANCER's own features — the deployable metric.

The DNSMOS-adapter experiment failed for a measured reason: a no-reference
metric, approximated by a surrogate, is exploited by the optimizer (+2.24 MOS
of optimism on the spectrograms it steers into; see ONBOARD_RESULTS.md). Three
properties of this design attack that failure directly:

1. **It sees the noisy input too.** Input is two channels — `(noisy_mag,
   enhanced_mag)` — so the metric judges a *relationship*, not an absolute. An
   adversarial "enhanced" must still look like a plausible denoising of this
   particular noisy signal, which is a far harder constraint to satisfy than
   looking good on its own.
2. **Its labels are intrusive.** PESQ is computed against the clean reference,
   which VoiceBank-DEMAND provides at training time. The target is therefore
   trustworthy in a way a no-reference metric's own judgments are not — even
   though at adaptation time no clean signal is needed.
3. **It is Lipschitz-bounded by construction.** Every layer carries
   `spectral_norm`, so the gradient's steepness is bounded — there are no
   arbitrarily sharp directions for an optimizer to exploit. This is a
   structural property the DNSMOS body does not have.

And it is cheap: eco8-neaixt's `MetricDiscriminator` is **181,650 parameters**
(8x smaller than the DNSMOS body's 1.44 M) and consumes compressed magnitude at
n_fft 512 / hop 256 / compress 0.3 — *exactly* the features the ConvFSENet
trunk already computes. No STFT, no ISTFT, no ISTFT-adjoint: with
`enhanced_mag = |X| * mask`, the weight gradient is `dL/dmask = dL/d|Y| * |X|`.

**The load-bearing design decision is the training distribution.** A regressor
trained only on one enhancer's natural outputs is accurate on a thin manifold
and will be exploited off it — that is precisely how the DNSMOS adapter failed
(Spearman 0.89 on natural clips, +2.24 MOS optimism on adversarial ones). So
the candidates here deliberately span a wide range of enhancement qualities and
failure modes: under- and over-suppression, band-limited damage, noisy/clean
interpolation, random smooth masks, and spectral holes — the kinds of signals a
gradient-following optimizer actually produces.

    python pesq_predictor.py build --utts 600      # PESQ-labelled candidates
    python pesq_predictor.py train --epochs 40
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dsp  # noqa: E402

SR = 16000
COMPRESS = 0.3
N_BINS = 257
VBD = (Path.home() / ".cache/huggingface/hub/datasets--JacobLinCool--VoiceBank-DEMAND-16k"
       / "snapshots/4497db342d7312978c45690591fda86117831940/data")


# --------------------------------------------------------------------------
# Model — eco8-neaixt's MetricDiscriminator (MIT), used as a PESQ regressor.
# --------------------------------------------------------------------------
class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features: int, beta: float = 2.0) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class PesqPredictor(nn.Module):
    """(noisy_mag, enhanced_mag) [B, 2, F, T] -> normalized PESQ.

    `head="sigmoid"` (default) bounds the output to (0, 1), i.e. PESQ 1..4.5 —
    the range the metric can actually mean. `head="lsig"` reproduces eco8's
    `LearnableSigmoid1d(beta=2)`, which reaches 2.0 == **PESQ 8.0**.

    That headroom is not harmless. Used as a gradient source, the optimizer
    drove the lsig model to 7.8 — chasing a score no PESQ computation can
    return — and the adaptation had to be rescued with an external clamp. A
    plain sigmoid makes the ceiling structural: the gradient vanishes at the
    top of the valid range because there is nowhere further to go.

    (The bound costs nothing real: PESQ tops out at 4.64 for a perfect match,
    so a sigmoid gives up only the last 0.14, which no enhancer reaches.)
    """

    def __init__(self, dim: int = 16, head: str = "sigmoid") -> None:
        super().__init__()
        if head not in ("sigmoid", "lsig"):
            raise ValueError(f"head must be 'sigmoid' or 'lsig', got {head!r}")
        self.head = head
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(2, dim, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True), nn.PReLU(dim),
            spectral_norm(nn.Conv2d(dim, dim * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True), nn.PReLU(dim * 2),
            spectral_norm(nn.Conv2d(dim * 2, dim * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True), nn.PReLU(dim * 4),
            spectral_norm(nn.Conv2d(dim * 4, dim * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 8, affine=True), nn.PReLU(dim * 8),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            spectral_norm(nn.Linear(dim * 8, dim * 4)), nn.Dropout(0.3), nn.PReLU(dim * 4),
            spectral_norm(nn.Linear(dim * 4, 1)),
            LearnableSigmoid1d(1) if head == "lsig" else nn.Sigmoid(),
        )

    def forward(self, noisy_mag: torch.Tensor, enh_mag: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.stack((noisy_mag, enh_mag), dim=1)).flatten()


def compress(mag: torch.Tensor | np.ndarray):
    """The enhancer's own magnitude compression — same kernel, same exponent."""
    if isinstance(mag, np.ndarray):
        return np.power(np.maximum(mag, 0) + 1e-9, COMPRESS)
    return (mag.clamp_min(0) + 1e-9).pow(COMPRESS)


def norm_pesq(p: float) -> float:
    """eco8's normalization: PESQ 1..4.5 -> 0..1."""
    return (p - 1.0) / 3.5


# --------------------------------------------------------------------------
# Candidate generation — the part that decides whether this can be exploited.
# --------------------------------------------------------------------------
def candidate_masks(X: np.ndarray, clean: np.ndarray, noisy: np.ndarray,
                    rng: np.random.Generator) -> list[np.ndarray]:
    """A deliberately wide spread of masks, including bad ones.

    Covering the failure modes a gradient-following optimizer produces is the
    whole point: over-suppression, holes, band damage and near-passthrough all
    have to be labelled so the regressor knows what they are worth.
    """
    F_, T = X.shape
    mag = np.abs(X)
    ideal = np.abs(dsp.stft(clean)) / np.maximum(mag, 1e-9)      # ideal ratio mask
    ideal = np.clip(ideal, 0, 1)
    out = [
        np.ones_like(mag),                                        # passthrough (= noisy)
        ideal,                                                    # near-perfect
        np.clip(ideal * rng.uniform(1.3, 2.2), 0, 1),             # under-suppression
        ideal * rng.uniform(0.25, 0.7),                           # over-suppression
        np.clip(ideal + rng.normal(0, 0.25, ideal.shape), 0, 1),  # noisy mask
    ]
    # spectral holes — the classic metric-hacking artefact
    holes = ideal.copy()
    for _ in range(rng.integers(3, 10)):
        f0 = rng.integers(0, F_ - 20); t0 = rng.integers(0, max(1, T - 20))
        holes[f0:f0 + rng.integers(5, 20), t0:t0 + rng.integers(5, 20)] = 0.0
    out.append(holes)
    # band-limited damage
    band = ideal.copy()
    cut = rng.integers(F_ // 3, F_)
    band[cut:] *= rng.uniform(0.0, 0.3)
    out.append(band)
    # smooth random mask, unrelated to the signal
    r = rng.uniform(0, 1, (F_ // 16 + 1, T // 16 + 1))
    smooth = np.array(torch.nn.functional.interpolate(
        torch.tensor(r)[None, None].float(), size=(F_, T), mode="bilinear",
        align_corners=True)[0, 0])
    out.append(smooth)
    return out


def build(args) -> None:
    import pyarrow.parquet as pq
    import soundfile as sf
    from pesq import pesq as pesq_fn

    shards = sorted(VBD.glob("train-*.parquet"))
    rng = np.random.default_rng(args.seed)
    feats, labels = [], []
    kept = skipped = 0
    seg = args.seconds * SR
    for shard in shards:
        tbl = pq.ParquetFile(str(shard)).read()
        idx = rng.permutation(len(tbl))[: args.utts // len(shards) + 1]
        for i in idx:
            if kept >= args.utts:
                break
            row = tbl.slice(int(i), 1).to_pylist()[0]
            cl, _ = sf.read(io.BytesIO(row["clean"]["bytes"]), dtype="float32")
            no, _ = sf.read(io.BytesIO(row["noisy"]["bytes"]), dtype="float32")
            n = min(len(cl), len(no))
            if n < seg:
                continue
            s = rng.integers(0, n - seg + 1)
            cl, no = cl[s:s + seg].astype(np.float64), no[s:s + seg].astype(np.float64)
            X = dsp.stft(no)
            nmag = np.abs(X)
            for m in candidate_masks(X, cl, no, rng):
                y = dsp.istft(X * m, length=seg)
                try:
                    p = pesq_fn(SR, cl.astype(np.float32), y.astype(np.float32), "wb")
                except Exception:
                    skipped += 1
                    continue
                feats.append(np.stack([compress(nmag), compress(nmag * m)]).astype(np.float16))
                labels.append(norm_pesq(p))
            kept += 1
            if kept % 25 == 0:
                print(f"  {kept}/{args.utts} utts, {len(labels)} labelled candidates", flush=True)
        if kept >= args.utts:
            break
    Fa = np.stack(feats); La = np.array(labels, np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, feats=Fa, labels=La)
    print(f"\nwrote {args.out}: {Fa.shape} feats, PESQ range "
          f"{La.min()*3.5+1:.2f}..{La.max()*3.5+1:.2f} (mean {La.mean()*3.5+1:.2f}), "
          f"{skipped} skipped")


def train(args) -> None:
    z = np.load(args.data)
    Fa, La = z["feats"], z["labels"]
    n = len(La)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_ev = max(1, int(n * 0.15))
    ev, tr = perm[:n_ev], perm[n_ev:]
    print(f"{n} candidates: {len(tr)} train / {len(ev)} eval, "
          f"PESQ {La.min()*3.5+1:.2f}..{La.max()*3.5+1:.2f}")

    model = PesqPredictor(args.dim, args.head)
    ceiling = (2.0 if args.head == "lsig" else 1.0) * 3.5 + 1.0
    print(f"PesqPredictor({args.head}): {sum(p.numel() for p in model.parameters()):,} "
          f"params, output ceiling PESQ {ceiling:.1f}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xe = torch.from_numpy(Fa[ev]).float(); Ye = torch.from_numpy(La[ev])

    def ev_stats():
        model.eval()
        with torch.no_grad():
            pr = torch.cat([model(Xe[k:k+16, 0], Xe[k:k+16, 1]) for k in range(0, len(Xe), 16)])
        model.train()
        p, t = pr.numpy(), Ye.numpy()
        ra, rb = np.argsort(np.argsort(p)), np.argsort(np.argsort(t))
        return (float(np.abs(p - t).mean() * 3.5), float(np.corrcoef(p, t)[0, 1]),
                float(np.corrcoef(ra, rb)[0, 1]))

    for ep in range(args.epochs):
        order = tr[rng.permutation(len(tr))]
        tot = 0.0
        for k in range(0, len(order), args.batch):
            b = order[k:k + args.batch]
            xb = torch.from_numpy(Fa[b]).float(); yb = torch.from_numpy(La[b])
            loss = F.mse_loss(model(xb[:, 0], xb[:, 1]), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        if ep % 5 == 4 or ep == 0:
            mae, r, rs = ev_stats()
            print(f"  epoch {ep+1:3d}  train MSE {tot/len(order):.5f}  "
                  f"eval MAE {mae:.3f} PESQ  pearson {r:+.3f}  spearman {rs:+.3f}")
    torch.save({"model": model.state_dict(), "dim": args.dim, "head": args.head}, args.out)
    print(f"saved {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate PESQ-labelled candidates from VBD")
    b.add_argument("--utts", type=int, default=600)
    b.add_argument("--seconds", type=int, default=3)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", type=Path, default=HERE / "artifacts" / "pesq_dataset.npz")
    b.set_defaults(func=build)
    t = sub.add_parser("train", help="fit the predictor")
    t.add_argument("--data", type=Path, default=HERE / "artifacts" / "pesq_dataset.npz")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--dim", type=int, default=16)
    t.add_argument("--head", choices=("sigmoid", "lsig"), default="sigmoid",
                   help="output nonlinearity; see PesqPredictor")
    t.add_argument("--out", type=Path, default=HERE / "artifacts" / "pesq_predictor.pt")
    t.set_defaults(func=train)
    args = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(8)
    args.func(args)


if __name__ == "__main__":
    main()
