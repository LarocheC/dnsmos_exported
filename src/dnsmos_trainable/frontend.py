"""Featurizer mirroring the official sig_bak_ovr.onnx graph op-for-op.

The official graph frames the raw waveform with two 160-sample-offset slices,
projects each 320-sample frame through *trained* real/imag matrices (close to
a hann-windowed DFT — per-bin cosine 0.93-0.98 — but trained end-to-end, so
exact parity requires transplanting them verbatim), and takes a log10 power
spectrogram.

The official chain is sqrt(re^2+im^2) -> Pow(2) -> Max(eps) -> Log -> Div(ln10).
sqrt followed by squaring cancels, so we compute log10(clamp(re^2+im^2, eps))
directly: numerically identical forward, and gradient-safe (no sqrt at 0).
"""

import torch
from torch import nn

from dnsmos_trainable.constants import (
    EPS, HOP, INPUT_LEN, LN10, N_BINS, N_FRAMES, OFFICIAL_CONFIG, WIN,
)


class Framing(nn.Module):
    """[B, 144160] -> [B, 900, 320] via the official two-slice layout.

    frames[:, t, :160] = x[160t : 160t+160]  (slice A, reshaped)
    frames[:, t, 160:] = x[160t+160 : 160t+320]  (slice B, reshaped)
    which together give overlapping 320-sample windows at hop 160.
    """

    def __init__(self, cfg=OFFICIAL_CONFIG) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cfg = self.cfg
        a = x[:, : cfg.input_len - cfg.hop].reshape(b, cfg.n_frames, cfg.hop)
        c = x[:, cfg.hop:].reshape(b, cfg.n_frames, cfg.hop)
        return torch.cat([a, c], dim=2)


class TrainedStft(nn.Module):
    """[B, 900, 320] -> re, im each [B, 900, 161] via the trained projections."""

    def __init__(self, cfg=OFFICIAL_CONFIG) -> None:
        super().__init__()
        self.w_re = nn.Parameter(torch.zeros(cfg.n_bins, cfg.win))
        self.w_im = nn.Parameter(torch.zeros(cfg.n_bins, cfg.win))

    def forward(self, frames: torch.Tensor):
        re = frames @ self.w_re.T
        im = frames @ self.w_im.T
        return re, im


class LogPower(nn.Module):
    """re, im -> log10 power spectrogram [B, 900, 161]."""

    def forward(self, re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
        p = re * re + im * im
        return torch.log(torch.clamp(p, min=EPS)) * (1.0 / LN10)


class Frontend(nn.Module):
    """[B, 144160] -> [B, 1, 900, 161] feature map (NCHW)."""

    def __init__(self, cfg=OFFICIAL_CONFIG) -> None:
        super().__init__()
        self.cfg = cfg
        self.framing = Framing(cfg)
        self.stft = TrainedStft(cfg)
        self.logpower = LogPower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.framing(x)
        re, im = self.stft(frames)
        feat = self.logpower(re, im)
        return feat.unsqueeze(1)
