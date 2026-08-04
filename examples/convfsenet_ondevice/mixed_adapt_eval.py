#!/usr/bin/env python3
"""The testbed adaptation deserves: mixed per-clip shifts with no global fix.

Gate 0 ended the reverb line: the shift's correction was expressible as one
global preset (mask^0.5 floored at 0.1, +0.641), so nothing adaptive could
justify itself. The honest conclusion was a design criterion, not a method:

> test-time adaptation can only justify itself under shifts whose correction
> varies per clip — where the preset that helps one clip hurts another.

This builds that testbed. Each clip draws ONE shift from a family chosen so
the corrections *conflict*:

  reverb     — smearing; the enhancer over-suppresses; wants a SOFTER mask
  noise      — 2-4x the clip's own noise; wants a HARDER mask
  tilt       — the noise re-colored (spectral tilt); frequency-dependent fix
  bandlimit  — telephone-ish channel on both ref and input; wants ~identity

The null hypothesis every method must beat is the **best single global preset
fit on the test set itself** — an adversarially generous null (in deployment
you could not fit it). Validity check: that null should be near zero here,
and per-kind presets far above it, or the family failed to conflict.

Methods scored against the null, all per clip:

  * per-clip grid oracle          — upper bound of the preset family
  * per-kind preset oracle        — what a domain classifier + preset table buys
  * v1 / v2 zeroth-order pick     — critic ranks the 21 candidates, no gradient
  * gradient loop (v1, 10 steps)  — the deployed configuration
  * gradient + oracle stop        — its ceiling
  * judge accept/reject           — v2 gates the gradient result (Exp 1 rule)

    python mixed_adapt_eval.py --clips 150
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

SHIFTS = ("reverb", "noise", "tilt", "bandlimit")
RIR_BANK: list | None = None      # set to a list of measured RIRs to replace
                                  # the synthetic generator (see real_rir.py)
GRID = [(a, f) for a in (0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0)
        for f in (0.0, 0.05, 0.10)]
CPS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]     # gradient checkpoints
K10 = CPS.index(10)


def load_train_pairs(n: int, seconds: int, seed: int = 7):
    """(clean, noisy) pairs from VBD *train* shards — for fitting routers.

    The router must never be fit on test utterances; this mirrors
    `load_test_pairs` but draws from the training shards.
    """
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    from pesq_predictor import VBD

    rng = np.random.default_rng(seed)
    seg = seconds * SR
    out = []
    for shard in sorted(VBD.glob("train-*.parquet")):
        tbl = pq.ParquetFile(str(shard)).read()
        for i in rng.permutation(len(tbl)):
            if len(out) >= n:
                return out
            row = tbl.slice(int(i), 1).to_pylist()[0]
            cl, _ = sf.read(io.BytesIO(row["clean"]["bytes"]), dtype="float32")
            no, _ = sf.read(io.BytesIO(row["noisy"]["bytes"]), dtype="float32")
            m = min(len(cl), len(no))
            if m < seg:
                continue
            st = (m - seg) // 2
            out.append((cl[st:st + seg].astype(np.float64),
                        no[st:st + seg].astype(np.float64)))
    return out


def lowpass_fir(fc_hz: float, sr: int = SR, taps: int = 101) -> np.ndarray:
    """Windowed-sinc lowpass, linear phase."""
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * fc_hz / sr * n) * np.hamming(taps)
    return h / h.sum()


def apply_shift(cl: np.ndarray, no: np.ndarray, kind: str,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (reference, enhancer_input) for one shifted clip.

    Convention as in the reverb study: the reference keeps whatever the
    enhancer cannot be expected to undo (reverb tail, channel), so PESQ
    charges it only for the noise it was actually asked to remove.
    """
    noise = no - cl
    if kind == "reverb":
        if RIR_BANK is not None:
            rir = RIR_BANK[int(rng.integers(len(RIR_BANK)))]
        else:
            rir = make_rir(float(rng.uniform(0.2, 0.6)), rng)
        cl_rev = reverberate(cl, rir)
        return cl_rev, cl_rev + noise
    if kind == "noise":
        k = float(rng.uniform(2.0, 4.0))
        return cl, cl + k * noise
    if kind == "tilt":
        b = float(rng.uniform(0.5, 0.9))
        sign = 1 if rng.random() < 0.5 else -1
        t = np.convolve(noise, np.array([1.0, sign * -b]))[: len(noise)]
        t *= np.sqrt(np.sum(noise ** 2) / max(np.sum(t ** 2), 1e-12))
        return cl, cl + 1.5 * t
    if kind == "bandlimit":
        fc = float(rng.uniform(2500, 4000))
        h = lowpass_fir(fc)
        clb = np.convolve(cl, h)[: len(cl)]
        nob = np.convolve(no, h)[: len(no)]
        return clb, nob
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=Path, default=HERE / "artifacts" / "pesq_predictor_sig.pt")
    ap.add_argument("--v2", type=Path, default=HERE / "artifacts" / "pesq_predictor_adv.pt")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--clips", type=int, default=150)
    ap.add_argument("--seconds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--sisnr-weight", type=float, default=0.5)
    ap.add_argument("--grid-only", action="store_true",
                    help="skip the gradient loop; enough for router labels")
    ap.add_argument("--pool", choices=("test", "train"), default="test",
                    help="which VBD split supplies the utterances; 'train' "
                         "builds a router-fitting pool that shares nothing "
                         "with the evaluation clips")
    ap.add_argument("--rir-dir", type=Path, default=None,
                    help="directory of measured RIR wavs; replaces the "
                         "synthetic reverb generator (see real_rir.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "mixed_shift.npz")
    args = ap.parse_args()

    torch.manual_seed(0); torch.set_num_threads(8)
    from pesq import pesq as pesq_fn

    global RIR_BANK
    if args.rir_dir is not None:
        from real_rir import load_rirs
        RIR_BANK = load_rirs(args.rir_dir)
        print(f"measured RIR bank: {len(RIR_BANK)} impulse responses")

    models = {}
    for tag, p in (("v1", args.v1), ("v2", args.v2)):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = PesqPredictor(ck["dim"], ck.get("head", "lsig")).eval()
        m.load_state_dict(ck["model"])
        for q in m.parameters():
            q.requires_grad_(False)
        models[tag] = m
    v1, v2 = models["v1"], models["v2"]

    trunk, head = build_split()
    sp = torch.load(args.split, map_location="cpu", weights_only=False)
    trunk.load_state_dict(sp["trunk"]); head.load_state_dict(sp["head"]); trunk.eval()

    pairs = (load_test_pairs(args.clips, args.seconds) if args.pool == "test"
             else load_train_pairs(args.clips, args.seconds, seed=args.seed + 7))
    rng = np.random.default_rng(args.seed)
    win = torch.from_numpy(dsp.hann_periodic()).float()

    n = len(pairs)
    kinds = np.array([SHIFTS[i % len(SHIFTS)] for i in range(n)])   # balanced
    base = np.full(n, np.nan)
    g_grid = np.full((n, len(GRID)), np.nan)     # preset-family gains
    bel_grid = {t: np.zeros((n, len(GRID))) for t in ("v1", "v2")}
    g_step = np.full((n, len(CPS)), np.nan)      # gradient-loop gains
    jud_step = np.full((n, len(CPS)), np.nan)    # v2 belief along the loop

    print(f"mixed-shift testbed: {n} clips x {args.seconds}s, "
          f"kinds balanced over {SHIFTS}\n")

    t0 = time.time()
    for i, (cl, no) in enumerate(pairs):
        ref_np, inp = apply_shift(cl, no, kinds[i], rng)
        ref = ref_np.astype(np.float32)
        no_s, nf = dsp.rms_normalize(inp)
        X = dsp.stft(no_s)
        Xt = torch.from_numpy(np.abs(X)).float()
        Xc = torch.from_numpy(X).to(torch.complex64)
        nmag_c = compress(Xt)
        mag = Xt[: trunk.n_features][None]
        with torch.no_grad():
            pad = mag[..., :1].repeat(1, 1, trunk.context_l)
            h = trunk(torch.cat([pad, mag], dim=-1))
            m0 = torch.sigmoid(head.conv(h))[0]
        noisy_t = torch.from_numpy(np.asarray(inp)).float()

        def synth(full):
            y = torch.istft((Xc * full)[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                            win_length=dsp.WIN_LENGTH, window=win,
                            length=len(inp), center=True)[0]
            return y / nf

        def score(est) -> float:
            try:
                return pesq_fn(SR, ref, est.numpy().astype(np.float32), "wb")
            except Exception:
                return np.nan

        full0 = torch.cat([m0, torch.zeros(1, m0.shape[1])], 0)
        b = score(synth(full0))
        if np.isnan(b):
            continue
        base[i] = b

        # ---- the preset family ----
        for j, (a, f) in enumerate(GRID):
            fj = torch.cat([torch.clamp(torch.pow(m0, a), min=f),
                            torch.zeros(1, m0.shape[1])], 0)
            g_grid[i, j] = score(synth(fj)) - b
            with torch.no_grad():
                emag = compress(Xt * fj)[None]
                for tag, m_ in models.items():
                    bel_grid[tag][i, j] = float(m_(nmag_c[None], emag)[0])

        # ---- the gradient loop (v1), judge recorded ----
        if args.grid_only:
            g_step[i] = 0.0; jud_step[i] = 0.0
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  {i+1:3d}/{n}  ({el:.0f}s)", flush=True)
            continue
        W = head.conv.weight.detach()[:, :, 0].clone().requires_grad_(True)
        bb = head.conv.bias.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([W, bb], lr=args.lr)

        def rec(k):
            with torch.no_grad():
                m = torch.sigmoid(torch.matmul(W, h[0]) + bb[:, None])
                full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
                g_step[i, k] = score(synth(full)) - b
                jud_step[i, k] = float(v2(nmag_c[None],
                                          compress(Xt * full)[None])[0]) * 3.5 + 1.0

        rec(0)
        for step in range(1, CPS[-1] + 1):
            m = torch.sigmoid(torch.matmul(W, h[0]) + bb[:, None])
            full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
            loss = -v1(nmag_c[None], compress(Xt * full)[None])[0]
            s = si_snr(synth(full), noisy_t)
            if float(s) < args.sisnr_floor:
                loss = loss + args.sisnr_weight * (args.sisnr_floor - s)
            opt.zero_grad(); loss.backward(); opt.step()
            if step in CPS:
                rec(CPS.index(step))

        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1:3d}/{n}  ({el:.0f}s, eta {el/(i+1)*(n-i-1)/60:.1f} min)",
                  flush=True)

    ok = ~np.isnan(base) & ~np.isnan(g_grid).any(1) & ~np.isnan(g_step).any(1)
    kinds, base = kinds[ok], base[ok]
    g_grid, g_step, jud_step = g_grid[ok], g_step[ok], jud_step[ok]
    for t in bel_grid:
        bel_grid[t] = bel_grid[t][ok]
    n = int(ok.sum())

    def stat(g):
        g = np.asarray(g)
        se = g.std(ddof=1) / np.sqrt(len(g))
        return f"{g.mean():+8.3f} {1.96*se:8.3f} {(g>0).mean()*100:8.0f}%"

    print(f"\n{n} clips; baseline PESQ {base.mean():.3f}; per kind:")
    for k in SHIFTS:
        s = kinds == k
        print(f"  {k:>9}: n={s.sum():3d}  baseline {base[s].mean():.3f}")

    # ---- the null and its validity ----
    jbest = int(np.argmax(g_grid.mean(0)))
    null = g_grid[:, jbest]
    print(f"\nVALIDITY — best single global preset (fit ON the test set): "
          f"a={GRID[jbest][0]:g}, f={GRID[jbest][1]:g}")
    print(f"{'':>26} {'gain':>8} {'95% CI':>8} {'improved':>9}")
    print(f"{'global preset null':>26} {stat(null)}")
    for k in SHIFTS:
        s = kinds == k
        print(f"{'  on ' + k:>26} {stat(null[s])}")

    perkind = np.zeros(n)
    print("\nper-kind preset oracle (domain classifier + preset table):")
    for k in SHIFTS:
        s = kinds == k
        jk = int(np.argmax(g_grid[s].mean(0)))
        perkind[s] = g_grid[s, jk]
        print(f"  {k:>9}: a={GRID[jk][0]:g}, f={GRID[jk][1]:g}  {stat(g_grid[s, jk])}")

    # ---- the contenders ----
    rows = [
        ("global preset (null)", null),
        ("per-kind preset oracle", perkind),
        ("per-clip grid oracle", g_grid.max(1)),
        ("v1 zeroth-order pick", g_grid[np.arange(n), bel_grid["v1"].argmax(1)]),
        ("v2 zeroth-order pick", g_grid[np.arange(n), bel_grid["v2"].argmax(1)]),
        ("gradient v1, 10 steps", g_step[:, K10]),
        ("gradient + oracle stop", g_step.max(1)),
        ("judge accept/reject", np.where(jud_step[:, K10] - jud_step[:, 0] >= -1.0,
                                         g_step[:, K10], 0.0)),
    ]
    print(f"\n{'method':>26} {'gain':>8} {'95% CI':>8} {'improved':>9} {'vs null':>8}")
    for name, g in rows:
        d = g - null
        print(f"{name:>26} {stat(g)} {d.mean():+8.3f}")
    print("\nper-kind breakdown of the gradient loop (10 steps):")
    for k in SHIFTS:
        s = kinds == k
        print(f"  {k:>9}: {stat(g_step[s, K10])}")

    np.savez_compressed(args.out, kinds=kinds, base=base, g_grid=g_grid,
                        g_step=g_step, jud_step=jud_step, steps=np.array(CPS),
                        bel_v1=bel_grid["v1"], bel_v2=bel_grid["v2"],
                        grid=np.array(GRID),
                        meta=np.array([args.pool, str(args.seconds),
                                       str(args.seed), str(args.clips)]))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
