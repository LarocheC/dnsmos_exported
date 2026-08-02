"""STFT / ISTFT and the exact ISTFT adjoint, in numpy.

Upstream keeps the complex STFT/ISTFT *outside* the ONNX graph — on the STM32N6
they run on the Cortex-M55 (CMSIS-DSP), while the NPU sees only the real-valued
magnitude window. This module is the host-side half of that split, plus the one
piece on-device training adds: the adjoint of the ISTFT, needed to carry
dL/d(waveform) back to dL/d(spectrum).

Conventions match `torch.stft`/`torch.istft` as used upstream
(`convfsenet/model.py:_TorchSpectrogram`): n_fft=512, hop=256, win_length=512,
periodic Hann, center=True, pad_mode='reflect', normalized=False.

The adjoint is NOT the forward STFT. Writing the ISTFT as

    y = Crop( OLA( w * irfft(Y) ) / wsq )

its adjoint is, in reverse,

    gY = (c_k/N) * rfft( w * Frame( ZeroPad(gy) / wsq ) )

with c_k = 1 at k in {0, N/2} and 2 elsewhere — the factor that accounts for
irfft's Hermitian folding. `test_convfsenet_demo.py` checks this against
torch.autograd rather than trusting the derivation.
"""

from __future__ import annotations

import numpy as np

N_FFT = 512
HOP = 256
WIN_LENGTH = 512


def hann_periodic(win_length: int = WIN_LENGTH) -> np.ndarray:
    """torch.hann_window default: periodic (denominator N, not N-1)."""
    n = np.arange(win_length)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_length)).astype(np.float64)


def _pad_center(x: np.ndarray, n_fft: int = N_FFT) -> np.ndarray:
    pad = n_fft // 2
    return np.pad(x, (pad, pad), mode="reflect")


def num_frames(n_samples: int, hop: int = HOP) -> int:
    """Frame count torch.stft(center=True) produces for n_samples."""
    return n_samples // hop + 1


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
         win_length: int = WIN_LENGTH) -> np.ndarray:
    """[S] real -> [F, T] complex, matching torch.stft(center=True).

    Requires len(x) > n_fft//2 and win_length == n_fft, which is the deployed
    geometry. numpy's reflect pad would happily multi-reflect a shorter buffer
    and return numbers torch.stft refuses to produce, so reject it explicitly
    rather than silently diverging from the documented contract.
    """
    if win_length != n_fft:
        raise ValueError(f"win_length must equal n_fft here (got {win_length} vs {n_fft})")
    if len(x) <= n_fft // 2:
        raise ValueError(
            f"need more than n_fft//2 = {n_fft // 2} samples for center padding, got {len(x)}"
        )
    w = hann_periodic(win_length)
    xp = _pad_center(np.asarray(x, dtype=np.float64), n_fft)
    T = num_frames(len(x), hop)
    frames = np.stack([xp[t * hop: t * hop + n_fft] for t in range(T)], axis=0)
    return np.fft.rfft(frames * w, n=n_fft, axis=-1).T  # [F, T]


def _window_envelope(T: int, n_fft: int = N_FFT, hop: int = HOP,
                     win_length: int = WIN_LENGTH) -> np.ndarray:
    """Sum of w^2 over the overlap-add timeline (torch's normalization term)."""
    w = hann_periodic(win_length)
    total = n_fft + (T - 1) * hop
    env = np.zeros(total, dtype=np.float64)
    for t in range(T):
        env[t * hop: t * hop + n_fft] += w * w
    return env


def istft(Y: np.ndarray, length: int, n_fft: int = N_FFT, hop: int = HOP,
          win_length: int = WIN_LENGTH, eps: float = 1e-11) -> np.ndarray:
    """[F, T] complex -> [length] real, matching torch.istft(center=True, length=)."""
    w = hann_periodic(win_length)
    T = Y.shape[1]
    frames = np.fft.irfft(Y.T, n=n_fft, axis=-1) * w         # [T, n_fft]
    total = n_fft + (T - 1) * hop
    acc = np.zeros(total, dtype=np.float64)
    for t in range(T):
        acc[t * hop: t * hop + n_fft] += frames[t]
    env = _window_envelope(T, n_fft, hop, win_length)
    acc = acc / np.maximum(env, eps)
    pad = n_fft // 2
    return acc[pad: pad + length]


def istft_adjoint(gy: np.ndarray, T: int, n_fft: int = N_FFT, hop: int = HOP,
                  win_length: int = WIN_LENGTH, eps: float = 1e-11) -> np.ndarray:
    """Adjoint of `istft`: dL/dy [length] -> dL/dY [F, T] complex.

    Complex convention: the returned array `G` satisfies
    dL/dRe(Y) = Re(G) and dL/dIm(Y) = Im(G), i.e. it is the pair of real
    gradients packed as a complex array (NOT a Wirtinger derivative).
    """
    w = hann_periodic(win_length)
    length = len(gy)
    total = n_fft + (T - 1) * hop
    pad = n_fft // 2

    u = np.zeros(total, dtype=np.float64)          # Crop^H : zero-pad back
    u[pad: pad + length] = gy
    u = u / np.maximum(_window_envelope(T, n_fft, hop, win_length), eps)

    frames = np.stack([u[t * hop: t * hop + n_fft] for t in range(T)], axis=0)  # OLA^H
    frames = frames * w

    G = np.fft.rfft(frames, n=n_fft, axis=-1)      # [T, F]
    c = np.full(G.shape[-1], 2.0)
    c[0] = 1.0
    if n_fft % 2 == 0:
        c[-1] = 1.0
    G = G * (c / n_fft)
    return G.T                                     # [F, T]


def apply_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Y = X * mask, a real gain on the complex spectrum (upstream's masker)."""
    return X * mask


def mask_grad(X: np.ndarray, gY: np.ndarray) -> np.ndarray:
    """VJP of Y = X * mask w.r.t. the real mask: dL/dmask = Re(X) gYr + Im(X) gYi."""
    return X.real * gY.real + X.imag * gY.imag


def sisnr_and_grad(est: np.ndarray, ref: np.ndarray, eps: float = 1e-10):
    """Scale-invariant SNR (dB) and its analytic gradient w.r.t. `est`."""
    u = est - est.mean()
    s = ref - ref.mean()
    s_energy = float(np.dot(s, s)) + eps
    alpha = float(np.dot(u, s)) / s_energy
    target = alpha * s
    noise = u - target
    t2 = float(np.dot(target, target)) + eps
    e2 = float(np.dot(noise, noise)) + eps
    value = 10.0 * np.log10(t2 / e2)
    g = (20.0 / np.log(10.0)) * (target / t2 - noise / e2)
    return value, g - g.mean()
