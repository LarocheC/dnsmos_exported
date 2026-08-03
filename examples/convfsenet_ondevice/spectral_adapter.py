#!/usr/bin/env python3
"""Drive DNSMOS from the ENHANCER's spectrogram — no STFT, no ISTFT, no adjoint.

The enhancer applies a **real gain** to the noisy complex STFT, so the enhanced
magnitude is exactly `|Y| = |X| * mask` in the enhancer's own domain — it never
has to be synthesized to a waveform. If the metric can be fed from there, the
whole DSP sandwich between enhancer and metric disappears:

    now:  mask -> Y=X*mask -> ISTFT -> wav -> [DNSMOS: framing, trained STFT,
          log-power] -> body -> scores
          ...and backwards: dL/dwav -> ISTFT-adjoint -> dL/dY -> mask VJP

    with an adapter:  mask -> |Y|=|X|*mask -> adapter -> body -> scores
          ...and backwards: dL/dmask = dL/d|Y| * |X|          (one multiply)

Measured motivation (see ONBOARD_RESULTS.md): 46x less host DSP per adaptation
window, `istft_adjoint` (the hardest-verified function in dsp.py) drops out of
the backward entirely, 18% of the loss graph's nodes disappear, and 2 of the 6
irreducible-float ops go with them. Two facts make it sound rather than
wishful: `|STFT(ISTFT(X*mask))|` matches `|X*mask|` at cosine 0.9999, and a
real gain cannot alter phase, so a magnitude-domain metric loses no information
*for this loop* (it would for a phase-modifying enhancer).

**What this script does NOT do is re-learn the metric.** The official body and
polynomial stay frozen and bit-exact; only a thin adapter maps the enhancer's
257-bin / 62.5 fps log-power spectrogram onto the grid the body expects
(161 bins / 100 fps).

Two measured surprises shape the design:

1. **A plain bilinear resample already works surprisingly well** — no training
   at all, and the frozen body scores BAK at Spearman 0.86 / OVRL 0.75 against
   true DNSMOS. (An earlier estimate of 0.27-0.57 was wrong: it used
   nearest-neighbour time indexing and a stray x10 on the log scale.) The
   trained STFT kernels are close enough to a DFT that a resample transfers
   much of the signal.
2. **Regressing the feature map is actively harmful.** Feature MSE onto the
   official `Frontend(wav)` output is densely supervised — ~145k values per
   clip instead of 3 — and training it cuts eval feature MSE 6.4x (6.04 ->
   0.94) while *dropping* BAK rank correlation from 0.86 to 0.15. The body
   reads a global max over the spectrogram; MSE regression smooths exactly the
   peaks that max selects. Being closer on average is worse where it counts.

So `--loss score` is the default: the frozen body is differentiable, so the
adapter is regressed on the 3 mapped scores it must preserve. `--loss feature`
is kept because reproducing that failure is the clearest demonstration of why
a proxy objective needs to be the one you care about.

Risk containment: the learned path is zero-initialized (at step 0 the adapter
*is* the bilinear resample), and nothing downstream of it is trainable, so the
adapter cannot quietly reshape what "quality" means — the failure mode that
sank the capacity-shrunk student in FEASIBILITY.md.

Gate: rank correlation against the true metric on held-out clips. Passing that
is necessary but NOT sufficient — this repo has now found three times that
agreement metrics do not predict steering quality, so an adapter that passes
here must still be run through `adaptation_eval.py`'s paired protocol before it
is believed.

    python spectral_adapter.py --epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import dsp  # noqa: E402
from dnsmos_trainable.constants import OFFICIAL_CONFIG  # noqa: E402
from dnsmos_trainable.model import DnsmosModel  # noqa: E402

EPS = 1e-12


class SpectralAdapter(nn.Module):
    """[B, 1, F_enh, T_enh] enhancer log-power -> [B, 1, n_frames, n_bins] body input.

    Fixed bilinear resample onto the body's grid, plus a zero-initialized
    residual CNN. At initialization the module is exactly the resample, so the
    optimizer starts from the best hand-designed guess rather than from noise.
    """

    def __init__(self, n_frames: int, n_bins: int, width: int = 24) -> None:
        super().__init__()
        self.n_frames = int(n_frames)
        self.n_bins = int(n_bins)
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, logp_enh: torch.Tensor) -> torch.Tensor:
        """logp_enh: [B, 1, F_enh, T_enh] -> [B, 1, n_frames, n_bins]."""
        base = F.interpolate(logp_enh, size=(self.n_bins, self.n_frames),
                             mode="bilinear", align_corners=True)
        base = base.transpose(2, 3)                      # -> [B, 1, n_frames, n_bins]
        return base + self.net(base)


def enhancer_logpower(wav: np.ndarray, n_features: int = 257) -> np.ndarray:
    """The spectrogram the enhancer already computes, as log10-power."""
    X = dsp.stft(wav.astype(np.float64))
    p = (np.abs(X) ** 2)[:n_features]
    return np.log10(np.maximum(p, EPS)).astype(np.float32)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=HERE.parents[1] / "models" / "dnsmos_transplanted.pt")
    ap.add_argument("--cache", type=Path, default=HERE / "artifacts" / "vbd_cache.npz")
    ap.add_argument("--train-clips", type=int, default=112)
    ap.add_argument("--eval-clips", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--loss", choices=("feature", "score", "both"), default="score",
                    help="what to regress. 'feature' (per-bin MSE onto the official "
                         "frontend map) is densely supervised but MEASURED HARMFUL: it "
                         "cuts feature MSE 6.4x while dropping BAK rank correlation "
                         "0.86 -> 0.15, because the body reads a global max and MSE "
                         "smooths exactly the peaks it selects. 'score' regresses the "
                         "3 mapped scores through the frozen body instead.")
    ap.add_argument("--feat-weight", type=float, default=0.02,
                    help="weight on the feature term when --loss both")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "spectral_adapter.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(8)
    cfg = OFFICIAL_CONFIG

    model = DnsmosModel().eval()
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    for p in model.parameters():
        p.requires_grad_(False)

    clips = np.load(args.cache)["clips"]
    n_tr, n_ev = args.train_clips, args.eval_clips
    assert n_tr + n_ev <= len(clips), f"cache has {len(clips)} clips"

    print(f"precomputing targets: official frontend features + scores "
          f"({n_tr} train / {n_ev} eval clips)")
    feats_in, feats_tgt, mos_tgt = [], [], []
    with torch.no_grad():
        for ci in range(n_tr + n_ev):
            x = clips[ci][: cfg.input_len].astype(np.float32)
            feats_in.append(torch.from_numpy(enhancer_logpower(x))[None])   # [1,257,T]
            wt = torch.from_numpy(x)[None]
            feats_tgt.append(model.frontend(wt)[0])                          # [1,900,161]
            mos_tgt.append(model.poly(model.body(model.frontend(wt)))[0])
    X_all = torch.stack(feats_in)                                            # [N,1,257,T]
    Y_all = torch.stack(feats_tgt)                                           # [N,1,900,161]
    M_all = torch.stack(mos_tgt).numpy()
    print(f"  enhancer grid {tuple(X_all.shape[-2:])} -> body grid {tuple(Y_all.shape[-2:])}")

    # Normalize the adapter's input/target to comparable scales: the body's
    # LogPower is ln(p)/ln(10) = log10(p), same as ours, so no rescale needed.
    adapter = SpectralAdapter(cfg.n_frames, cfg.n_bins, args.width)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    tr = torch.arange(n_tr)
    ev = torch.arange(n_tr, n_tr + n_ev)

    def evaluate() -> tuple[float, np.ndarray, np.ndarray]:
        adapter.eval()
        with torch.no_grad():
            pred_feat = adapter(X_all[ev])
            fmse = float(F.mse_loss(pred_feat, Y_all[ev]))
            pred_mos = model.poly(model.body(pred_feat)).numpy()
        adapter.train()
        return fmse, pred_mos, M_all[ev.numpy()]

    base_mse, base_mos, true_mos = evaluate()
    print(f"\nbefore training (pure bilinear resample): feature MSE {base_mse:.4f}")
    for i, nm in enumerate(["SIG", "BAK", "OVRL"]):
        print(f"  {nm:5s} pearson {np.corrcoef(true_mos[:,i], base_mos[:,i])[0,1]:+.3f}  "
              f"spearman {spearman(true_mos[:,i], base_mos[:,i]):+.3f}  "
              f"mean|d| {np.abs(true_mos[:,i]-base_mos[:,i]).mean():.3f}")

    print(f"\ntraining the correction ({sum(p.numel() for p in adapter.parameters()):,} params, "
          f"body frozen, loss={args.loss}):")
    for ep in range(args.epochs):
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for k in range(0, len(perm), args.batch):
            idx = perm[k: k + args.batch]
            pred_feat = adapter(X_all[idx])
            if args.loss == "feature":
                loss = F.mse_loss(pred_feat, Y_all[idx])
            else:
                # The body is frozen but differentiable, so the 3 scores we
                # actually care about can be regressed directly through it.
                pred_mos = model.poly(model.body(pred_feat))
                loss = F.mse_loss(pred_mos, torch.from_numpy(M_all[idx.numpy()]))
                if args.loss == "both":
                    loss = loss + args.feat_weight * F.mse_loss(pred_feat, Y_all[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 10 == 9 or ep == 0:
            fmse, pm, tm = evaluate()
            r = [np.corrcoef(tm[:, i], pm[:, i])[0, 1] for i in range(3)]
            print(f"  epoch {ep+1:3d}  train loss {tot/len(perm):7.4f}  eval feat MSE {fmse:7.4f}  "
                  f"pearson SIG/BAK/OVRL {r[0]:+.3f}/{r[1]:+.3f}/{r[2]:+.3f}")

    fmse, pm, tm = evaluate()
    print(f"\nafter training: feature MSE {base_mse:.4f} -> {fmse:.4f}")
    for i, nm in enumerate(["SIG", "BAK", "OVRL"]):
        print(f"  {nm:5s} pearson {np.corrcoef(tm[:,i], pm[:,i])[0,1]:+.3f}  "
              f"spearman {spearman(tm[:,i], pm[:,i]):+.3f}  "
              f"mean|d| {np.abs(tm[:,i]-pm[:,i]).mean():.3f}   "
              f"(true {tm[:,i].mean():.2f} / adapted {pm[:,i].mean():.2f})")

    torch.save({"adapter": adapter.state_dict(), "n_frames": cfg.n_frames,
                "n_bins": cfg.n_bins, "width": args.width}, args.out)
    print(f"\nsaved {args.out}")
    print("NOTE: rank correlation is necessary, not sufficient. Run the paired "
          "protocol in adaptation_eval.py before trusting this as a gradient source.")


if __name__ == "__main__":
    main()
