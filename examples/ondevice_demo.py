#!/usr/bin/env python3
"""On-device-style training loop with NO PyTorch: onnxruntime + numpy only.

Optimizes an additive waveform perturbation (or, with --fir, a 64-tap FIR
filter) to raise the DNSMOS OVRL score of a noisy signal, using gradients
from the exported fused forward+backward loss graph. An SI-SNR hinge anchors
the result to the input signal — optimizing a no-reference metric without an
anchor invites metric-hacking artifacts (EUSIPCO 2024).

The fp32 loss graph provides gradients; the int8 QDQ forward artifact plays
the role of the deployed metric and reports the "deployed" score at the end.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
INPUT_LEN = 144160
LOG10E = 10.0 / np.log(10.0)


def make_noisy_signal(seed: int = 0) -> np.ndarray:
    """Harmonic 'voice' + noise, generated in numpy."""
    rng = np.random.default_rng(seed)
    t = np.arange(INPUT_LEN) / SR
    f0 = 140.0 * (1.0 + 0.02 * np.sin(2 * np.pi * 4.5 * t))
    sig = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6))
    sig *= 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * 3.0 * t), 0.0, None)
    sig = 0.25 * sig / np.abs(sig).max()
    noise = rng.standard_normal(INPUT_LEN)
    noise = 0.08 * noise / np.abs(noise).max()
    return (sig + noise).astype(np.float32)


def sisnr_and_grad(est: np.ndarray, ref: np.ndarray, eps: float = 1e-10):
    """Scale-invariant SNR (dB) and its analytic gradient w.r.t. est."""
    u = est - est.mean()
    s = ref - ref.mean()
    s_energy = float(np.dot(s, s)) + eps
    alpha = float(np.dot(u, s)) / s_energy
    t = alpha * s
    e = u - t
    t2 = float(np.dot(t, t)) + eps
    e2 = float(np.dot(e, e)) + eps
    sisnr = 10.0 * np.log10(t2 / e2)
    # d sisnr / du = (20/ln10) (t/||t||^2 - e/||e||^2); then re-center.
    g = 2.0 * LOG10E * (t / t2 - e / e2)
    return sisnr, g - g.mean()


class NpAdam:
    def __init__(self, shape, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return self.lr * mhat / (np.sqrt(vhat) + self.eps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-graph", type=Path, default=ROOT / "artifacts" / "dnsmos_loss_fp32.onnx")
    parser.add_argument("--int8-fwd", type=Path, default=ROOT / "artifacts" / "dnsmos_fwd_int8_qdq.onnx")
    parser.add_argument("--wav", type=Path, default=None, help="optional 16 kHz wav input")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sisnr-target", type=float, default=18.0, help="hinge target in dB")
    parser.add_argument("--sisnr-weight", type=float, default=1.0)
    parser.add_argument("--fir", action="store_true", help="learn a 64-tap FIR instead of a free perturbation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.wav is not None:
        import soundfile as sf

        audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        assert sr == SR, "provide a 16 kHz wav"
        while len(audio) < INPUT_LEN:
            audio = np.concatenate([audio, audio])
        x0 = audio[:INPUT_LEN].astype(np.float32)
    else:
        x0 = make_noisy_signal(args.seed)

    loss_sess = ort.InferenceSession(str(args.loss_graph), providers=["CPUExecutionProvider"])
    int8_sess = None
    if args.int8_fwd.exists():
        int8_sess = ort.InferenceSession(str(args.int8_fwd), providers=["CPUExecutionProvider"])
    w = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # L = -OVRL_mos

    def dnsmos(sess, wav):
        raw, mos, *rest = sess.run(None, {"wav": wav[None].astype(np.float32), "w": w}) if len(sess.get_inputs()) == 2 else sess.run(None, {"wav": wav[None].astype(np.float32)})
        return mos[0], rest[0][0] if rest else None

    mos0, _ = dnsmos(loss_sess, x0)
    print(f"start:  SIG {mos0[0]:.3f}  BAK {mos0[1]:.3f}  OVRL {mos0[2]:.3f}")

    if args.fir:
        taps = np.zeros(64, dtype=np.float64)
        taps[0] = 1.0
        opt = NpAdam(taps.shape, lr=1e-3)
    else:
        delta = np.zeros(INPUT_LEN, dtype=np.float64)
        opt = NpAdam(delta.shape, lr=args.lr)

    x_pad = np.concatenate([np.zeros(63, dtype=np.float64), x0.astype(np.float64)])
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, 64)[:, ::-1]  # [n, 64]

    mos = mos0
    for step in range(args.steps):
        if args.fir:
            x = (windows @ taps).astype(np.float32)
        else:
            x = (x0 + delta).astype(np.float32)
        mos, g_dnsmos = dnsmos(loss_sess, x)
        sisnr, g_sisnr = sisnr_and_grad(x.astype(np.float64), x0.astype(np.float64))
        # J = -OVRL + weight * max(0, target - SISNR); hinge keeps the anchor
        # inactive while we are close to the input.
        g_x = g_dnsmos.astype(np.float64)
        if sisnr < args.sisnr_target:
            g_x -= args.sisnr_weight * g_sisnr
        if args.fir:
            g_taps = windows.T @ g_x
            taps -= opt.step(g_taps)
        else:
            delta -= opt.step(g_x)
        if step % 20 == 0 or step == args.steps - 1:
            print(f"step {step:4d}: OVRL {mos[2]:.3f}  SI-SNR {min(sisnr, 99.9):5.1f} dB")

    x_final = (windows @ taps).astype(np.float32) if args.fir else (x0 + delta).astype(np.float32)
    sisnr, _ = sisnr_and_grad(x_final.astype(np.float64), x0.astype(np.float64))
    print(f"final:  SIG {mos[0]:.3f}  BAK {mos[1]:.3f}  OVRL {mos[2]:.3f}  SI-SNR {sisnr:.1f} dB")
    print(f"OVRL improvement: {mos[2] - mos0[2]:+.3f}")
    if int8_sess is not None:
        raw8, mos8 = int8_sess.run(None, {"wav": x_final[None]})
        print(f"deployed int8 OVRL: {mos8[0][2]:.3f}")

    assert "torch" not in sys.modules, "demo must stay PyTorch-free"
    ok = (mos[2] - mos0[2]) >= 0.2 and sisnr >= 15.0
    print("PASS" if ok else "FAIL", "(OVRL +0.2 with SI-SNR >= 15 dB)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
