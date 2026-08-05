#!/usr/bin/env python3
"""Second-enhancer replication: does the verdict survive an architecture change?

Every number so far is ConvFSENet's. This reruns the mixed-testbed comparison
on **LiSenNet** (arXiv 2409.13285; eco8-neaixt's faithful port, 37k params) —
a genuinely different family: sub-band U-Net encoder/decoder with a dual-path
recurrent bottleneck, magnitude-only 2-channel mask on the *compressed*
spectrum, noisy-phase reconstruction at deployment (no Griffin-Lim).

What transfers directly:

  * the critic: PesqPredictor consumes compressed (|X|^0.3) magnitudes at
    n_fft 512 / hop 256 — exactly LiSenNet's internal representation
  * the preset family: `max(M^a, f)` on LiSenNet's effective mask
    (mask[:,0]+mask[:,1], range (0,2))
  * the RMS input normalization (same convention as ConvFSENet)

The gradient loop adapts LiSenNet's *output layer* — the decoder's mask_conv
stack plus the LearnableSigmoid slopes (~600 params), the analogue of
ConvFSENet's mask head. Encoder/bottleneck activations are precomputed once
per clip and frozen, mirroring the frozen-trunk protocol.

Run with eco8's venv (needs the lisennet package):

    /home/claroche/eco8-neaixt/.venv/bin/python lisennet_mixed_eval.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ECO8 = Path("/home/claroche/eco8-neaixt")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(ECO8))

import dsp  # noqa: E402
from adapter_adapt_eval import si_snr  # noqa: E402
import mixed_adapt_eval as mx  # noqa: E402
from mixed_adapt_eval import SHIFTS, apply_shift, GRID  # noqa: E402
from pesq_adapt_eval import load_test_pairs  # noqa: E402
from pesq_predictor import PesqPredictor, SR  # noqa: E402

from lisennet.export_onnx import _load_from_checkpoint  # noqa: E402

CPS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
K10 = CPS.index(10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=ECO8 / "cp_lisennet" / "g_best")
    ap.add_argument("--predictor", type=Path,
                    default=HERE / "artifacts" / "pesq_predictor_sig.pt")
    ap.add_argument("--clips", type=int, default=150)
    ap.add_argument("--seconds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--sisnr-weight", type=float, default=0.5)
    ap.add_argument("--rir-dir", type=Path, default=Path(
        "/tmp/claude-1000/-home-claroche-dnsmos-exported/"
        "dec8099b-3421-445d-829f-32a9edb3a984/scratchpad/rir/mit/Audio"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "lisennet_mixed.npz")
    args = ap.parse_args()

    torch.manual_seed(0); torch.set_num_threads(8)
    from pesq import pesq as pesq_fn

    from real_rir import load_rirs
    mx.RIR_BANK = load_rirs(args.rir_dir)
    print(f"measured RIR bank: {len(mx.RIR_BANK)} impulse responses")

    net = _load_from_checkpoint(args.ckpt)
    for p in net.parameters():
        p.requires_grad_(False)
    print(f"LiSenNet: {sum(p.numel() for p in net.parameters()):,} params")

    ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
    metric = PesqPredictor(ck["dim"], ck.get("head", "lsig")).eval()
    metric.load_state_dict(ck["model"])
    for p in metric.parameters():
        p.requires_grad_(False)

    # the adapted subset: the decoder's output stack
    def head_params():
        ps = list(net.decoder.mask_conv.parameters())
        ps += list(net.decoder.lsigmoid.parameters())
        return ps
    n_head = sum(p.numel() for p in head_params())
    print(f"adapted output layer: {n_head:,} params")

    pairs = load_test_pairs(args.clips, args.seconds)
    rng = np.random.default_rng(args.seed)

    n = len(pairs)
    kinds = np.array([SHIFTS[i % len(SHIFTS)] for i in range(n)])
    base = np.full(n, np.nan)
    g_grid = np.full((n, len(GRID)), np.nan)
    bel_grid = np.zeros((n, len(GRID)))
    g_step = np.full((n, len(CPS)), np.nan)

    print(f"\nLiSenNet mixed testbed: {n} clips\n")
    t0 = time.time()
    for i, (cl, no) in enumerate(pairs):
        ref_np, inp = apply_shift(cl, no, kinds[i], rng)
        ref = ref_np.astype(np.float32)
        no_s, nf = dsp.rms_normalize(inp)
        src = torch.from_numpy(no_s).float()[None]

        with torch.no_grad():
            spec = net.power_compress(net.apply_stft(src))       # (B, T, F)
            src_mag = spec.abs()
            src_pha = spec.angle()
            feat = net.build_features(src_mag, src_pha)
            enc_list = net.encoder(feat)
            bott = net.blocks(enc_list[-1])

        def decode_mask():
            m2 = net.decoder(bott, list(enc_list))               # (B, 2, T, F)
            return m2[:, 0] + m2[:, 1] + 2e-8                    # effective mask

        def synth_from(eff_mask):
            est_mag = eff_mask * src_mag
            est_spec = torch.complex(est_mag * src_pha.cos(),
                                     est_mag * src_pha.sin())    # noisy phase
            est = net.apply_istft(net.power_uncompress(est_spec),
                                  length=src.shape[-1])[0]
            return est / nf

        def score(est) -> float:
            try:
                return pesq_fn(SR, ref, est.numpy().astype(np.float32), "wb")
            except Exception:
                return np.nan

        with torch.no_grad():
            m0 = decode_mask()
            b = score(synth_from(m0))
        if np.isnan(b):
            continue
        base[i] = b

        # critic features: compressed magnitudes, (B, F, T)
        nmag_c = src_mag.transpose(1, 2)

        with torch.no_grad():
            for j, (a, f) in enumerate(GRID):
                mj = torch.clamp(torch.pow(m0, a), min=f)
                est = synth_from(mj)
                g_grid[i, j] = score(est) - b
                emag = (mj * src_mag).transpose(1, 2)
                bel_grid[i, j] = float(metric(nmag_c, emag)[0])

        # ---- gradient loop on the output layer ----
        saved = [p.detach().clone() for p in head_params()]
        for p in head_params():
            p.requires_grad_(True)
        opt = torch.optim.Adam(head_params(), lr=args.lr)
        noisy_t = torch.from_numpy(np.asarray(inp)).float()

        def rec(k):
            with torch.no_grad():
                g_step[i, k] = score(synth_from(decode_mask())) - b

        rec(0)
        for step in range(1, CPS[-1] + 1):
            m = decode_mask()
            emag = (m * src_mag).transpose(1, 2)
            loss = -metric(nmag_c, emag)[0]
            s = si_snr(synth_from(m), noisy_t)
            if float(s) < args.sisnr_floor:
                loss = loss + args.sisnr_weight * (args.sisnr_floor - s)
            opt.zero_grad(); loss.backward(); opt.step()
            if step in CPS:
                rec(CPS.index(step))

        with torch.no_grad():                     # restore for the next clip
            for p, s0 in zip(head_params(), saved):
                p.copy_(s0); p.requires_grad_(False)

        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1:3d}/{n}  ({el:.0f}s, eta {el/(i+1)*(n-i-1)/60:.1f} min)",
                  flush=True)

    ok = (~np.isnan(base)) & ~np.isnan(g_grid).any(1) & ~np.isnan(g_step).any(1)
    kinds, base = kinds[ok], base[ok]
    g_grid, g_step, bel_grid = g_grid[ok], g_step[ok], bel_grid[ok]
    n = int(ok.sum())

    def stat(g):
        se = g.std(ddof=1) / np.sqrt(len(g))
        return f"{g.mean():+8.3f} {1.96*se:8.3f} {(g>0).mean()*100:8.0f}%"

    jbest = int(np.nanmean(g_grid, axis=0).argmax())
    null = g_grid[:, jbest]
    print(f"\n{n} clips; baseline PESQ {base.mean():.3f}; "
          f"null preset a={GRID[jbest][0]:g}, f={GRID[jbest][1]:g}")
    print(f"{'method':>28} {'gain':>8} {'95% CI':>8} {'improved':>9} {'vs null':>9}")

    def vs_null(g):
        d = g - null
        se = d.std(ddof=1) / np.sqrt(len(d))
        return f"{d.mean():+8.3f}" + ("*" if abs(d.mean()) > 1.96 * se else "")

    perkind = np.zeros(n)
    for k in SHIFTS:
        s = kinds == k
        perkind[s] = g_grid[s, int(np.nanmean(g_grid[s], axis=0).argmax())]
    rows = [
        ("global preset (null)", null),
        ("per-kind preset oracle", perkind),
        ("per-clip grid oracle", g_grid.max(1)),
        ("v1 zeroth-order pick", g_grid[np.arange(n), bel_grid.argmax(1)]),
        ("gradient v1, 10 steps", g_step[:, K10]),
        ("gradient + oracle stop", g_step.max(1)),
    ]
    for name, g in rows:
        print(f"{name:>28} {stat(g)} {vs_null(g) if name != rows[0][0] else '—':>9}")

    print("\nper-kind, gradient at 10 steps:")
    for k in SHIFTS:
        s = kinds == k
        print(f"  {k:>9}: {stat(g_step[s, K10])}")
    print("\nper-kind best presets:")
    for k in SHIFTS:
        s = kinds == k
        jk = int(np.nanmean(g_grid[s], axis=0).argmax())
        print(f"  {k:>9}: a={GRID[jk][0]:g}, f={GRID[jk][1]:g}  {stat(g_grid[s, jk])}")

    np.savez_compressed(args.out, kinds=kinds, base=base, g_grid=g_grid,
                        g_step=g_step, bel_grid=bel_grid, steps=np.array(CPS),
                        grid=np.array(GRID))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
