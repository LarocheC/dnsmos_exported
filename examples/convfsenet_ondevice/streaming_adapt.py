#!/usr/bin/env python3
"""Streaming on-device adaptation: enhance frame-by-frame, bank 1 s, score with DNSMOS.

The windowed demo (`ondevice_train.py`) recomputes the trunk over the whole
9.01 s segment per step — batch-mode adaptation. This variant matches how the
chip actually enhances: the per-frame FIFO trunk (eco8's deployed form,
4.40 ms/frame on target) runs in real time anyway, so the adaptation window's
tensors are banked FOR FREE as enhancement happens:

  every 16 ms hop (real-time enhancement path, already paid for):
    M55   |STFT| column                                 -> mag_t [256]
    NPU   trunk_stream(mag_t, 9 FIFO states)            -> h_t [192, 1]
    M55   mask_t = sigmoid(W h_t + b)                   -> gain, applied, ISTFT-OLA out
    M55   bank (X_t, h_t, mask_t)                       <- the only extra work: 3 memcpys

  every 63 banked frames (~1.01 s), opportunistic adaptation:
    M55   ISTFT of the banked window                    -> enhanced [16128]
    NPU   dnsmos_loss_crop1s(enhanced[:16000], w)       -> scores + dL/d(enhanced)
    M55   SI-SNR hinge vs banked noisy input
    M55   ISTFT-adjoint, mask VJP                       -> dmask [1, 256, 63]
    M55   head_bwd(dmask, mask, h)                      -> dW [256,192], db [256]
    M55   Adam step on (W, b)                           -> next frames use new weights

No emit_T=64 trunk recompute anywhere: vs the windowed demo this removes
1,094 MMAC of trunk work per 9 s window and replaces it with zero extra
compute (the banked tensors are the enhancement path's own outputs). The
trunk FIFO states flow straight through window boundaries, so every frame is
enhanced with its full real receptive field — no per-window cold starts.

Banked masks are valid gradients because weights only change at window
boundaries: every banked mask was computed with exactly the (W, b) being
updated. PyTorch-free like ondevice_train.py: ORT inference sessions + numpy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dsp  # noqa: E402
from ondevice_train import NpAdam  # noqa: E402

SR = 16000
N_FEATURES_FULL = 257
DNSMOS_CROP_LEN = 16000          # the 1 s crop artifact's input
W_FRAMES = 63                    # 63 x 256-hop = 16128 samples >= 1 s
OFFICIAL_LEN = 144160            # 9.01 s official DNSMOS segment


def _sess(path: Path, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class StreamingEnhancer:
    """The per-frame path: FIFO trunk (ORT) + numpy head, with banking."""

    def __init__(self, trunk_path: Path, threads: int = 1):
        self.trunk = _sess(trunk_path, threads)
        ins = self.trunk.get_inputs()
        self.state_names = [i.name for i in ins if i.name.startswith("state_")]
        self.state_shapes = [tuple(d if isinstance(d, int) else 1 for d in i.shape)
                             for i in ins if i.name.startswith("state_")]
        self.n_features = int(ins[0].shape[1])

    def reset(self):
        self.states = [np.zeros(s, dtype=np.float32) for s in self.state_shapes]

    def step(self, mag_t: np.ndarray) -> np.ndarray:
        """One hop: mag_t [256] -> h_t [192, 1]; advances the FIFO states."""
        feed = {"noisy_mag": mag_t[None].astype(np.float32)}
        feed.update({n: s for n, s in zip(self.state_names, self.states)})
        out = self.trunk.run(None, feed)
        self.states = out[1:]
        return out[0][0]                                   # [192, 1]


def head_mask(h_t: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The deployed backend conv (k=1) as numpy: h [192,1] -> mask col [256]."""
    return 1.0 / (1.0 + np.exp(-(W @ h_t[:, 0] + b)))


def enhance_stream(eng: StreamingEnhancer, X: np.ndarray, W, b,
                   collect=False):
    """Stream a whole clip's STFT through trunk+head with fixed weights."""
    F, T = X.shape
    eng.reset()
    mask_full = np.zeros((F, T))
    hs = np.zeros((192, T), dtype=np.float32) if collect else None
    for t in range(T):
        mag_t = np.abs(X[: eng.n_features, t]).astype(np.float32)
        h_t = eng.step(mag_t)
        if collect:
            hs[:, t] = h_t[:, 0]
        mask_full[: eng.n_features, t] = head_mask(h_t, W, b)
    return mask_full, hs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    p.add_argument("--trunk", type=Path, default=None,
                   help="default: artifacts/convfsenet_trunk_stream_int8.onnx")
    p.add_argument("--dnsmos-loss", type=Path,
                   default=HERE / "artifacts_crop" / "crop1s_int8_d2.onnx",
                   help="1 s crop loss graph (flat wav [1,16000] I/O)")
    p.add_argument("--head-graphs", type=Path, default=HERE / "artifacts" / "stream_head",
                   help="dir with head_bwd exported at T=63 (auto-built if missing)")
    p.add_argument("--official", type=Path,
                   default=HERE.parents[1] / "models" / "sig_bak_ovr.onnx")
    p.add_argument("--cache", type=Path, default=HERE / "artifacts" / "vbd_cache.npz")
    p.add_argument("--clip", type=int, default=159,
                   help="cache clip index (default = last, crop_study's adaptation clip)")
    p.add_argument("--passes", type=int, default=1,
                   help="passes over the clip (default 1 = the deployment-realistic "
                        "protocol: streaming audio is never replayed; multi-pass "
                        "replays over-optimize the int8 crop proxy — its score "
                        "keeps rising while the official metric falls)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sisnr-floor", type=float, default=6.0)
    p.add_argument("--sisnr-weight", type=float, default=0.5)
    p.add_argument("--energy-gate", type=float, default=0.5,
                   help="skip adaptation on windows whose noisy RMS is below "
                        "this fraction of the clip RMS (speech-presence gate). "
                        "A noise-only window teaches nothing useful: the "
                        "enhancer correctly outputs ~silence there, the "
                        "SI-SNR-to-noisy anchor craters, and the update drags "
                        "the mask toward reproducing noise. 0 disables.")
    p.add_argument("--passthrough-head", action="store_true",
                   help="init head to identity instead of the trained backend")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    trunk_path = args.trunk or (args.artifacts / "convfsenet_trunk_stream_int8.onnx")
    eng = StreamingEnhancer(trunk_path, args.threads)
    F = eng.n_features

    # ---- torch-touching setup, all in one place; the loop below is torch-free.
    bwd_path = args.head_graphs / "convfsenet_head_bwd.onnx"
    need_graphs = not bwd_path.exists()
    if args.passthrough_head:
        W = np.zeros((F, 192), dtype=np.float64)
        b = np.full(F, 4.0, dtype=np.float64)
        print("head: pass-through init (mask ~= 0.982)")
    if need_graphs or not args.passthrough_head:
        import torch  # noqa: PLC0415  (setup only)
        if not args.passthrough_head:
            split = torch.load(args.artifacts / "convfsenet_split.pt", map_location="cpu")
            W = split["head"]["conv.weight"].numpy()[:, :, 0].astype(np.float64)  # [256,192]
            b = split["head"]["conv.bias"].numpy().astype(np.float64)
            print("head: trained backend weights (real enhancer baseline)")
        if need_graphs:
            args.head_graphs.mkdir(parents=True, exist_ok=True)
            import export_demo_artifacts as eda
            eda.export_head_graphs(F, W_FRAMES, args.head_graphs)
            print(f"exported head graphs at T={W_FRAMES} into {args.head_graphs}")
        del sys.modules["torch"]
    head_bwd = _sess(bwd_path, args.threads)

    loss = _sess(args.dnsmos_loss, args.threads)
    official = _sess(args.official, threads=4)
    w_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def official_scores(wav: np.ndarray) -> np.ndarray:
        seg = (np.asarray(wav) / norm_factor).astype(np.float32)   # original level
        while len(seg) < OFFICIAL_LEN:
            seg = np.concatenate([seg, seg])
        return official.run(None, {"input_1": seg[:OFFICIAL_LEN][None]})[0][0]

    clips = np.load(args.cache)["clips"]
    noisy_raw = clips[args.clip].astype(np.float64)
    # ConvFSENet expects RMS-normalized input (see dsp.rms_normalize): the
    # power-law prologue makes the mask level-dependent, and raw audio yields
    # a destroyed mask rather than a degraded one.
    noisy, norm_factor = dsp.rms_normalize(noisy_raw)
    X = dsp.stft(noisy)                                    # [257, T]
    T = X.shape[1]
    n_windows = T // W_FRAMES
    print(f"clip {args.clip}: {len(noisy)/SR:.2f} s, {T} frames, "
          f"{n_windows} adaptation windows/pass, {args.passes} passes "
          f"({n_windows*args.passes} weight updates)")

    print(f"noisy           OVRL {official_scores(noisy)[2]:.3f}")
    mask0, _ = enhance_stream(eng, X, W, b)
    base = dsp.istft(dsp.apply_mask(X, mask0), length=len(noisy))
    mos_base = official_scores(base)
    print(f"baseline head   SIG {mos_base[0]:.3f}  BAK {mos_base[1]:.3f}  OVRL {mos_base[2]:.3f}")

    clip_rms = float(np.sqrt(np.mean(noisy ** 2)))
    opt = NpAdam([W.shape, b.shape], lr=args.lr)
    t0 = time.time()
    n_updates = n_skipped = 0
    best = (-np.inf, W.copy(), b.copy())
    for pss in range(args.passes):
        crop_scores = []
        eng.reset()                                        # new stream, states from zero
        bank_h = np.zeros((192, W_FRAMES), dtype=np.float32)
        bank_mask = np.zeros((F, W_FRAMES))
        widx = 0
        for t in range(n_windows * W_FRAMES):
            j = t % W_FRAMES
            mag_t = np.abs(X[:F, t]).astype(np.float32)
            h_t = eng.step(mag_t)                          # states persist across windows
            m_t = head_mask(h_t, W, b)
            bank_h[:, j] = h_t[:, 0]
            bank_mask[:, j] = m_t
            if j != W_FRAMES - 1:
                continue

            # ---- window boundary: opportunistic adaptation ----
            s0 = widx * W_FRAMES
            noisy_seg = noisy[s0 * dsp.HOP: s0 * dsp.HOP + W_FRAMES * dsp.HOP]
            seg_rms = float(np.sqrt(np.mean(noisy_seg ** 2)))
            if seg_rms < args.energy_gate * clip_rms:
                n_skipped += 1
                widx += 1
                continue
            Xw = X[:, s0: s0 + W_FRAMES]
            mask_full = np.zeros((N_FEATURES_FULL, W_FRAMES))
            mask_full[:F] = bank_mask
            seg_len = W_FRAMES * dsp.HOP
            enhanced = dsp.istft(dsp.apply_mask(Xw, mask_full), length=seg_len)

            raw, mos, grad = loss.run(None, {
                "wav": enhanced[:DNSMOS_CROP_LEN].astype(np.float32)[None],
                "w": w_vec})
            crop_scores.append(float(mos[0][2]))
            g_wav = np.zeros(seg_len)
            g_wav[:DNSMOS_CROP_LEN] = grad.reshape(-1).astype(np.float64)
            sisnr, g_sisnr = dsp.sisnr_and_grad(enhanced, noisy_seg)
            if sisnr < args.sisnr_floor:
                g_wav -= args.sisnr_weight * g_sisnr

            gY = dsp.istft_adjoint(g_wav, T=W_FRAMES)
            gmask_full = dsp.mask_grad(Xw, gY)
            dW, db = head_bwd.run(None, {
                "dmask": gmask_full[:F][None].astype(np.float32),
                "mask": bank_mask[None].astype(np.float32),
                "h": bank_h[None].astype(np.float32)})
            uW, ub = opt.step([dW.astype(np.float64), db.astype(np.float64)])
            W -= uW
            b -= ub
            n_updates += 1
            widx += 1
        maskP, _ = enhance_stream(eng, X, W, b)
        enhP = dsp.istft(dsp.apply_mask(X, maskP), length=len(noisy))
        mosP = official_scores(enhP)
        crop_mean = float(np.mean(crop_scores)) if crop_scores else float("nan")
        # On-device early stopping: the chip cannot run the official 9.01 s
        # model, but it CAN score its own enhanced windows with the same crop
        # metric it optimizes (an inference it computes anyway). Keep the
        # weights from the best-scoring pass.
        if crop_mean > best[0]:
            best = (crop_mean, W.copy(), b.copy())
        print(f"  pass {pss}: official OVRL {mosP[2]:.3f}  on-device crop OVRL "
              f"{crop_mean:.3f}  ({n_updates} updates, {n_skipped} gated)")
    W, b = best[1], best[2]
    print(f"  keeping weights from best on-device pass (crop OVRL {best[0]:.3f})")
    loop_s = time.time() - t0

    maskN, _ = enhance_stream(eng, X, W, b)
    final = dsp.istft(dsp.apply_mask(X, maskN), length=len(noisy))
    mos_final = official_scores(final)
    sisnr_final, _ = dsp.sisnr_and_grad(final, noisy)
    print(f"\nadapted head    SIG {mos_final[0]:.3f}  BAK {mos_final[1]:.3f}  "
          f"OVRL {mos_final[2]:.3f}  SI-SNR {sisnr_final:.1f} dB")
    print(f"official OVRL: baseline {mos_base[2]:.3f} -> adapted {mos_final[2]:.3f} "
          f"({mos_final[2]-mos_base[2]:+.3f})  [{loop_s:.1f} s host loop]")

    assert "torch" not in sys.modules, "the adaptation loop must stay PyTorch-free"
    ok = mos_final[2] > mos_base[2]
    print("PASS" if ok else "FAIL", "(adapted beats the streamed baseline)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
