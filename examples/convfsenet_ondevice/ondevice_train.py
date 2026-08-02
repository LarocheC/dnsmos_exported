#!/usr/bin/env python3
"""On-device training of ConvFSENet's mask head against DNSMOS — ORT + numpy only.

Simulates, step for step, what the STM32N6 would execute. No PyTorch, no
autograd, no training runtime: every learned quantity comes out of an ordinary
ONNX *inference* session, and the optimizer is ~20 lines of numpy.

Per 9.01 s adaptation window:

  M55   STFT(noisy)                        -> X [257, 564] complex
  M55   |X|, drop Nyquist                  -> mag [256, 564]
  NPU   trunk_int8  x ceil(T/emit_T)       -> h [1, 192, 564]        (frozen)
  M55   head_fwd(h, W, b)                  -> mask [1, 256, 564]     (trainable)
  M55   Y = X * mask ; ISTFT               -> enhanced [144160]
  NPU   dnsmos_loss(enhanced, w)           -> scores, dL/d(enhanced)
  M55   + SI-SNR anchor gradient
  M55   ISTFT-adjoint, mask VJP            -> dL/dmask [1, 256, 564]
  M55   head_bwd(dmask, mask, h)           -> dW [256,192], db [256]
  M55   Adam step on (W, b)                                       49,408 params

(head_fwd and head_bwd are ONNX graphs but land on the M55, not the NPU: the
Neural-ART maps MatMul to hardware only when the second operand is constant,
and both of theirs are runtime tensors. At 27.7 MMAC each, once per window,
that is affordable; the trunk and DNSMOS are what need the NPU.)

The gradient never enters the trunk: the head is the last layer and the mask is
a real gain on the noisy spectrum, so no trunk activations are stashed and no
backward pass traverses the 9 TCM blocks. That is what makes this fit on the
chip at all.

Anchor: DNSMOS is a no-reference metric and will happily reward artefacts
(EUSIPCO 2024). On device there is no clean reference, so the regularizer is a
hinge on SI-SNR against the *noisy input* — loose enough to permit real noise
removal, tight enough to forbid muting or hallucinating.
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

SR = 16000
DNSMOS_LEN = 144160  # 9.01 s, the DNSMOS segment
N_FEATURES_FULL = 257


def make_noisy(seed: int = 0, length: int = DNSMOS_LEN):
    """Synthetic noisy speech: harmonic 'voice' + broadband noise. Returns (noisy, voice)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length) / SR
    f0 = 140.0 * (1.0 + 0.02 * np.sin(2 * np.pi * 4.5 * t))
    voice = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    voice *= 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * 3.0 * t), 0.0, None)
    voice = 0.25 * voice / np.abs(voice).max()
    noise = rng.standard_normal(length)
    noise = 0.08 * noise / np.abs(noise).max()
    return (voice + noise).astype(np.float32), voice.astype(np.float32)


class NpAdam:
    """The whole on-device optimizer."""

    def __init__(self, shapes, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros(s, dtype=np.float64) for s in shapes]
        self.v = [np.zeros(s, dtype=np.float64) for s in shapes]
        self.t = 0

    def step(self, grads):
        self.t += 1
        out = []
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            out.append(self.lr * mh / (np.sqrt(vh) + self.eps))
        return out


class OnDeviceEnhancer:
    """Wraps the four ONNX sessions with the host-side DSP, as the M55 would."""

    def __init__(self, art_dir: Path, dnsmos_loss: Path, use_int8_trunk=True, threads=None):
        # `threads=1` makes the loop reproducible. It matters only for int8
        # graphs: fp32 ones are bit-identical at any thread count, while a
        # QDQ graph's output shifts with reduction order, and the backward's
        # `Equal` tie masks amplify that ULP into wholesale gradient
        # rerouting. Both the int8 trunk and an int8 loss graph are affected.
        opts = None
        if threads is not None:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads

        def sess(p):
            return ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])

        trunk_name = "convfsenet_trunk_int8.onnx" if use_int8_trunk else "convfsenet_trunk_fp32.onnx"
        self.trunk = sess(art_dir / trunk_name)
        self.head_fwd = sess(art_dir / "convfsenet_head_fwd.onnx")
        self.head_bwd = sess(art_dir / "convfsenet_head_bwd.onnx")
        self.loss = sess(dnsmos_loss)
        shp = self.trunk.get_inputs()[0].shape
        self.n_features = int(shp[1])
        self.window_w = int(shp[2])
        self.emit_t = int(self.trunk.get_outputs()[0].shape[2])
        self.context_l = self.window_w - self.emit_t
        self.loss_wav_shape = tuple(
            d if isinstance(d, int) else 1 for d in self.loss.get_inputs()[0].shape
        )

    def analyse(self, noisy: np.ndarray):
        X = dsp.stft(noisy)                               # [257, T]
        return X, np.abs(X)[: self.n_features]            # drop Nyquist for the graph

    def run_trunk(self, mag: np.ndarray) -> np.ndarray:
        """Windowed trunk over the whole segment, exactly as the chip's ring buffer does."""
        T = mag.shape[1]
        padded = np.concatenate(
            [np.repeat(mag[:, :1], self.context_l, axis=1), mag], axis=1
        )  # cold start = replicate (upstream's choice, worth ~0.045 PESQ)
        chunks = []
        for start in range(0, T, self.emit_t):
            block = padded[:, start: start + self.window_w]
            if block.shape[1] < self.window_w:
                block = np.pad(block, ((0, 0), (0, self.window_w - block.shape[1])))
            chunks.append(self.trunk.run(None, {"noisy_mag_window": block[None].astype(np.float32)})[0])
        return np.concatenate(chunks, axis=2)[:, :, :T]   # [1, 192, T]

    def mask_of(self, h, W, b):
        return self.head_fwd.run(None, {
            "h": h.astype(np.float32), "W": W.astype(np.float32), "b": b.astype(np.float32)})[0]

    def synthesise(self, X, mask, length):
        full = np.zeros((N_FEATURES_FULL, mask.shape[2]), dtype=np.float64)
        full[: self.n_features] = mask[0]                 # Nyquist gain stays 0
        return dsp.istft(dsp.apply_mask(X, full), length=length), full

    def dnsmos(self, wav: np.ndarray, w: np.ndarray):
        feed = wav.astype(np.float32).reshape(self.loss_wav_shape)
        raw, mos, grad = self.loss.run(None, {"wav": feed, "w": w.astype(np.float32)})
        return mos[0], grad.reshape(-1)

    def head_grads(self, dmask, mask, h):
        dW, db = self.head_bwd.run(None, {
            "dmask": dmask.astype(np.float32), "mask": mask.astype(np.float32),
            "h": h.astype(np.float32)})
        return dW.astype(np.float64), db.astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    p.add_argument("--dnsmos-loss", type=Path,
                   default=HERE.parents[1] / "artifacts" / "dnsmos_loss_fp32.onnx")
    p.add_argument("--dnsmos-int8-fwd", type=Path,
                   default=HERE.parents[1] / "artifacts" / "dnsmos_fwd_int8_qdq.onnx")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--sisnr-floor", type=float, default=6.0,
                   help="hinge: keep SI-SNR(enhanced, noisy) above this (dB)")
    p.add_argument("--sisnr-weight", type=float, default=0.5)
    p.add_argument("--fp32-trunk", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wav", type=Path, default=None)
    args = p.parse_args()

    eng = OnDeviceEnhancer(args.artifacts, args.dnsmos_loss, use_int8_trunk=not args.fp32_trunk)
    print(f"trunk: {'int8 QDQ (NPU)' if not args.fp32_trunk else 'fp32'} "
          f"| F={eng.n_features} emit_T={eng.emit_t} L={eng.context_l}")

    if args.wav is not None:
        import soundfile as sf
        audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        assert sr == SR, "provide a 16 kHz wav"
        while len(audio) < DNSMOS_LEN:
            audio = np.concatenate([audio, audio])
        noisy = audio[:DNSMOS_LEN].astype(np.float32)
    else:
        noisy, _ = make_noisy(args.seed)

    # Head state lives in RAM; pass-through init so every gain is attributable.
    W = np.zeros((eng.n_features, 192), dtype=np.float64)
    b = np.full(eng.n_features, 4.0, dtype=np.float64)     # sigmoid(4) = 0.982
    opt = NpAdam([W.shape, b.shape], lr=args.lr)
    w_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)   # L = -OVRL

    X, mag = eng.analyse(noisy)
    t0 = time.time()
    h = eng.run_trunk(mag)                                  # frozen: compute once
    trunk_ms = (time.time() - t0) * 1000
    T = h.shape[2]
    print(f"trunk: {mag.shape[1]} frames in {int(np.ceil(T/eng.emit_t))} windows, {trunk_ms:.0f} ms\n")

    mos_noisy, _ = eng.dnsmos(noisy, w_vec)
    print(f"noisy input        SIG {mos_noisy[0]:.3f}  BAK {mos_noisy[1]:.3f}  OVRL {mos_noisy[2]:.3f}")

    history = []
    for step in range(args.steps):
        mask = eng.mask_of(h, W, b)
        enhanced, mask_full = eng.synthesise(X, mask, len(noisy))

        mos, g_dnsmos = eng.dnsmos(enhanced, w_vec)         # dL/d(enhanced) for L = -OVRL
        sisnr, g_sisnr = dsp.sisnr_and_grad(enhanced, noisy.astype(np.float64))
        g_wav = g_dnsmos.astype(np.float64)
        if sisnr < args.sisnr_floor:                        # hinge: only pulls when violated
            g_wav -= args.sisnr_weight * g_sisnr

        gY = dsp.istft_adjoint(g_wav, T=X.shape[1])
        gmask_full = dsp.mask_grad(X, gY)
        dmask = gmask_full[: eng.n_features][None]

        dW, db = eng.head_grads(dmask, mask, h)
        uW, ub = opt.step([dW, db])
        W -= uW
        b -= ub

        history.append((mos[2], sisnr))
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  OVRL {mos[2]:.3f}  SIG {mos[0]:.3f}  BAK {mos[1]:.3f}  "
                  f"SI-SNR {sisnr:5.1f} dB  mask[{mask.min():.2f},{mask.max():.2f}]")

    mask = eng.mask_of(h, W, b)
    enhanced, _ = eng.synthesise(X, mask, len(noisy))
    mos, _ = eng.dnsmos(enhanced, w_vec)
    sisnr, _ = dsp.sisnr_and_grad(enhanced, noisy.astype(np.float64))
    print(f"\nenhanced (trained) SIG {mos[0]:.3f}  BAK {mos[1]:.3f}  OVRL {mos[2]:.3f}  "
          f"SI-SNR {sisnr:.1f} dB")
    print(f"OVRL: {mos_noisy[2]:.3f} -> {mos[2]:.3f}  ({mos[2]-mos_noisy[2]:+.3f})")

    if args.dnsmos_int8_fwd.exists():
        s8 = ort.InferenceSession(str(args.dnsmos_int8_fwd), providers=["CPUExecutionProvider"])
        m8 = s8.run(None, {"wav": enhanced.astype(np.float32)[None]})[1]
        print(f"deployed int8 DNSMOS OVRL: {m8[0][2]:.3f}")

    assert "torch" not in sys.modules, "the on-device path must stay PyTorch-free"
    ok = (mos[2] - mos_noisy[2]) >= 0.2 and sisnr >= args.sisnr_floor - 3.0
    print("PASS" if ok else "FAIL",
          f"(OVRL +0.2 over noisy, SI-SNR >= {args.sisnr_floor - 3.0:.0f} dB)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
