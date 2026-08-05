#!/usr/bin/env python3
"""Multi-metric scoring of the key systems: PESQ, STOI, SI-SDR, DNSMOS.

The protocol paper criticizes single-metric evaluation, so it cannot itself
report only PESQ. This scores the four systems that carry the argument on
the measured-rooms mixed testbed, with reference metrics (PESQ, STOI,
SI-SDR vs the shift-appropriate reference) and the no-reference DNSMOS
P.835 (official ONNX, eco8's wrapper):

    baseline        the enhancer as shipped
    global preset   fit on the 400-clip TRAIN pool (the fair null)
    router          gain-regression argmax, fit on the train pool (held out)
    gradient        v1-critic loop, 10 steps (the deployed configuration)

Run with eco8-neaixt's venv (has pystoi/librosa/joblib on top of ours):

    /home/claroche/eco8-neaixt/.venv/bin/python multimetric_eval.py
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
from adapter_adapt_eval import si_snr as si_snr_t  # noqa: E402
from convfsenet_arch import build_split  # noqa: E402
import knob_router as kr  # noqa: E402
import mixed_adapt_eval as mx  # noqa: E402
from mixed_adapt_eval import SHIFTS, apply_shift  # noqa: E402
from pesq_adapt_eval import load_test_pairs  # noqa: E402
from pesq_predictor import PesqPredictor, compress, SR  # noqa: E402

from lisennet.eval_metrics_ext import DNSMOS, si_sdr, _stoi_one  # noqa: E402

SYSTEMS = ("baseline", "global preset", "router", "gradient")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path,
                    default=HERE / "artifacts" / "mixed_shift_realrir.npz")
    ap.add_argument("--train-npz", type=Path,
                    default=HERE / "artifacts" / "mixed_train_pool_realrir.npz")
    ap.add_argument("--rir-dir", type=Path, default=Path(
        "/tmp/claude-1000/-home-claroche-dnsmos-exported/"
        "dec8099b-3421-445d-829f-32a9edb3a984/scratchpad/rir/mit/Audio"))
    ap.add_argument("--predictor", type=Path,
                    default=HERE / "artifacts" / "pesq_predictor_sig.pt")
    ap.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sisnr-floor", type=float, default=6.0)
    ap.add_argument("--sisnr-weight", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "multimetric.npz")
    args = ap.parse_args()

    torch.manual_seed(0); torch.set_num_threads(8)
    from pesq import pesq as pesq_fn

    from real_rir import load_rirs
    mx.RIR_BANK = load_rirs(args.rir_dir)
    print(f"measured RIR bank: {len(mx.RIR_BANK)} impulse responses")

    ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
    metric = PesqPredictor(ck["dim"], ck.get("head", "lsig")).eval()
    metric.load_state_dict(ck["model"])
    for p in metric.parameters():
        p.requires_grad_(False)

    trunk, head = build_split()
    sp = torch.load(args.split, map_location="cpu", weights_only=False)
    trunk.load_state_dict(sp["trunk"]); head.load_state_dict(sp["head"]); trunk.eval()
    win = torch.from_numpy(dsp.hann_periodic()).float()

    # ---- fit the router + fair global preset on the train pool, once ----
    z = np.load(args.npz, allow_pickle=True)
    kinds, g_grid, grid = z["kinds"], z["g_grid"], z["grid"]
    zt = np.load(args.train_npz, allow_pickle=True)
    g_tr = np.nan_to_num(zt["g_grid"])
    f_te = kr.compute_features(args.npz, trunk, head, win)
    f_tr = kr.compute_features(args.train_npz, trunk, head, win)
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-9
    Xb = np.hstack([(f_tr - mu) / sd, np.ones((len(f_tr), 1))])
    B = np.linalg.solve(Xb.T @ Xb + 1.0 * np.eye(Xb.shape[1]), Xb.T @ g_tr)
    picks = (np.hstack([(f_te - mu) / sd, np.ones((len(f_te), 1))]) @ B).argmax(1)
    j_glob = int(g_tr.mean(0).argmax())
    print(f"train-fit global preset: a={grid[j_glob][0]:g}, f={grid[j_glob][1]:g}")

    dnsmos = DNSMOS()
    n = len(kinds)
    METRICS = ("pesq", "stoi", "sisdr", "ovrl", "bak", "sig")
    res = {s: {m: np.full(n, np.nan) for m in METRICS} for s in SYSTEMS}

    pairs = load_test_pairs(n, 3)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i, (cl, no) in enumerate(pairs):
        kind = SHIFTS[i % len(SHIFTS)]
        assert kind == str(kinds[i])
        ref_np, inp = apply_shift(cl, no, kind, rng)
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

        def synth(m):
            full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
            y = torch.istft((Xc * full)[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                            win_length=dsp.WIN_LENGTH, window=win,
                            length=len(inp), center=True)[0]
            return (y / nf).numpy().astype(np.float32)

        outs = {"baseline": synth(m0)}
        a, f = grid[j_glob]
        outs["global preset"] = synth(torch.clamp(torch.pow(m0, float(a)),
                                                  min=float(f)))
        a, f = grid[picks[i]]
        outs["router"] = synth(torch.clamp(torch.pow(m0, float(a)), min=float(f)))

        W = head.conv.weight.detach()[:, :, 0].clone().requires_grad_(True)
        b = head.conv.bias.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=args.lr)
        for _ in range(args.steps):
            m = torch.sigmoid(torch.matmul(W, h[0]) + b[:, None])
            full = torch.cat([m, torch.zeros(1, m.shape[1])], 0)
            loss = -metric(nmag_c[None], compress(Xt * full)[None])[0]
            est = torch.istft((Xc * full)[None], n_fft=dsp.N_FFT, hop_length=dsp.HOP,
                              win_length=dsp.WIN_LENGTH, window=win,
                              length=len(inp), center=True)[0] / nf
            s = si_snr_t(est, noisy_t)
            if float(s) < args.sisnr_floor:
                loss = loss + args.sisnr_weight * (args.sisnr_floor - s)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            outs["gradient"] = synth(torch.sigmoid(torch.matmul(W, h[0]) + b[:, None]))

        for s, est in outs.items():
            try:
                res[s]["pesq"][i] = pesq_fn(SR, ref, est, "wb")
            except Exception:
                pass
            res[s]["stoi"][i] = _stoi_one(ref, est)
            res[s]["sisdr"][i] = si_sdr(ref, est)
            d = dnsmos(est)
            res[s]["ovrl"][i] = d["OVRL"]; res[s]["bak"][i] = d["BAK"]
            res[s]["sig"][i] = d["SIG"]

        if (i + 1) % 15 == 0:
            el = time.time() - t0
            print(f"  {i+1:3d}/{n}  ({el:.0f}s, eta {el/(i+1)*(n-i-1)/60:.1f} min)",
                  flush=True)

    ok = np.ones(n, bool)
    for s in SYSTEMS:
        ok &= ~np.isnan(res[s]["pesq"])
    print(f"\n{int(ok.sum())} clips scored on all systems\n")

    hdr = f"{'system':>15}" + "".join(f"{m.upper():>10}" for m in METRICS)
    print(hdr)
    for s in SYSTEMS:
        print(f"{s:>15}" + "".join(f"{np.nanmean(res[s][m][ok]):>10.3f}"
                                   for m in METRICS))
    print("\npaired deltas vs baseline (* = 95% resolved):")
    print(hdr)
    for s in SYSTEMS[1:]:
        row = f"{s:>15}"
        for m in METRICS:
            d = res[s][m][ok] - res["baseline"][m][ok]
            d = d[~np.isnan(d)]
            se = d.std(ddof=1) / np.sqrt(len(d))
            row += f"{d.mean():>+9.3f}" + ("*" if abs(d.mean()) > 1.96 * se else " ")
        print(row)

    np.savez_compressed(args.out, kinds=kinds,
                        **{f"{s.replace(' ', '_')}_{m}": res[s][m]
                           for s in SYSTEMS for m in METRICS})
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
