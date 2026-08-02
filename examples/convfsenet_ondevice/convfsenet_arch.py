"""ConvFSENet windowed architecture, split into a frozen trunk and a trainable mask head.

Vendored from github.com/LarocheC/eco8-neaixt (MIT, (c) 2025-2026 Clement Laroche;
(c) 2023 Yexin Lu for the MP-SENet lineage) — specifically `convfsenet/model.py`
and the stateless-windowed STM32N6 view in `convfsenet/streaming.py`. Kept here so
this demo is self-contained; `tests/test_convfsenet_demo.py` cross-checks it
against the upstream implementation whenever a clone is available.

The split is the whole point of the demo:

    noisy |STFT| window [1, F, L+T]
        -> compress (|m|+1e-9)**0.3        FP32, stays out of int8 (upstream's
        -> frontend Conv1d(F->192,k=1)+ReLU     "compression prologue" exclusion)
        -> 9 x windowed TCM block          <-- TRUNK: frozen, int8, Neural-ART NPU
        -> h [1, 192, T]
        ------------------------------------------------------------------
        -> backend Conv1d(192->F,k=1)      <-- HEAD: float32, TRAINABLE on device
        -> sigmoid -> mask [1, F, T]            (49,408 params at F=256)

Because the mask is a real gain applied to the *noisy complex STFT*, and the head
is the last layer, a gradient arriving at the waveform reaches the head's weights
without ever entering the trunk. That is what makes on-device training tractable:
no trunk activations to stash, no backward through 9 TCM blocks.
"""

from __future__ import annotations

import torch
from torch import nn

# Geometry of the deployed 192/384 causal model (configs/convfsenet.json upstream).
N_FFT = 512
HOP = 256
WIN_LENGTH = 512
N_FEATURES_FULL = 257
N_CHANNELS_RES = 192
N_CHANNELS_CONV = 384
KERNEL_SIZE = 3
N_BLOCKS = 3
N_STACKS = 3
COMPRESS_FACTOR = 0.3
MAG_EPS = 1e-9
# L = sum over blocks of (K-1)*D = 3 stacks x (2+4+8) = 42; receptive field L+1 = 43.
CONTEXT_L = 42


def compress_magnitude(mag: torch.Tensor, compress_factor: float = COMPRESS_FACTOR) -> torch.Tensor:
    """Power-law magnitude compression, upstream's int8-friendliness trick."""
    return (mag + MAG_EPS).pow(compress_factor)


class WindowedTCMBlock(nn.Module):
    """One BN-folded TCM block run with VALID convs over a context window.

    Shrinks the time axis by trim=(K-1)*D and crops the residual to match, so a
    window of L+T columns collapses to T with no padding and no FIFO state —
    upstream's Track 1 rework, which removes the Slice/Concat/Gather ops that
    always fall back to M55-Hybrid epochs on the Neural-ART.
    """

    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.trim = (KERNEL_SIZE - 1) * self.dilation
        # BN is folded into conv1x1 and dconv at export time upstream; we carry
        # the folded (bias-bearing) form directly.
        self.conv1x1 = nn.Conv1d(N_CHANNELS_RES, N_CHANNELS_CONV, 1)
        self.dconv = nn.Conv1d(
            N_CHANNELS_CONV, N_CHANNELS_CONV, KERNEL_SIZE,
            stride=1, padding=0, dilation=self.dilation, groups=N_CHANNELS_CONV,
        )
        self.conv1x1_out = nn.Conv1d(N_CHANNELS_CONV, N_CHANNELS_RES, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv1x1(x))
        y = torch.relu(self.dconv(y))
        z = self.conv1x1_out(y)
        res = x[..., self.trim:] if self.trim > 0 else x
        return z + res


class ConvFSENetTrunk(nn.Module):
    """Frozen feature trunk: |STFT| window -> h [B, 192, T]. Runs int8 on the NPU."""

    def __init__(self, n_features: int = N_FEATURES_FULL - 1) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.frontend = nn.Conv1d(self.n_features, N_CHANNELS_RES, 1)
        dilations = [2 ** b for _ in range(N_STACKS) for b in range(N_BLOCKS)]
        self.blocks = nn.ModuleList(WindowedTCMBlock(d) for d in dilations)
        self.context_l = sum(b.trim for b in self.blocks)

    def forward(self, noisy_mag_window: torch.Tensor) -> torch.Tensor:
        """[B, F, L+T] plain |STFT| -> [B, 192, T]."""
        x = compress_magnitude(noisy_mag_window)
        x = torch.relu(self.frontend(x))
        for blk in self.blocks:
            x = blk(x)
        return x


class MaskHead(nn.Module):
    """Trainable mask head: h [B, 192, T] -> mask [B, F, T] in (0, 1).

    Deliberately tiny (F*192 + F params) and deliberately last: it is the only
    thing the on-device optimizer touches.
    """

    def __init__(self, n_features: int = N_FEATURES_FULL - 1) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.conv = nn.Conv1d(N_CHANNELS_RES, self.n_features, 1)

    @property
    def n_params(self) -> int:
        return self.conv.weight.numel() + self.conv.bias.numel()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(h))

    def init_passthrough(self, logit: float = 4.0) -> None:
        """Start as a near-unity mask (output ~= noisy input).

        Gives the demo an unambiguous baseline: every subsequent quality gain is
        attributable to the DNSMOS-driven updates rather than to pretraining.
        sigmoid(4) = 0.982.
        """
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.fill_(float(logit))


def build_split(n_features: int = N_FEATURES_FULL - 1, seed: int | None = 0):
    """Construct (trunk, head). n_features=256 matches the deployed drop_nyquist graph."""
    if seed is not None:
        torch.manual_seed(seed)
    trunk = ConvFSENetTrunk(n_features).eval()
    head = MaskHead(n_features).eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk, head


def load_from_eco8_checkpoint(ckpt_path, n_features: int = N_FEATURES_FULL - 1,
                              eco8_repo=None):
    """Load real ConvFSENet weights from an eco8-neaixt training checkpoint.

    Requires the upstream repo on sys.path (for `convfsenet.model` /
    `convfsenet.streaming`), because BatchNorm folding and the Nyquist slice are
    upstream's transforms and must not be re-derived here.
    """
    import sys

    if eco8_repo is not None:
        sys.path.insert(0, str(eco8_repo))
    import json
    from pathlib import Path

    from common.env import AttrDict
    from convfsenet.model import build_causal_model
    from convfsenet.streaming import ConvFSENetWindowedONNX

    ckpt_path = Path(ckpt_path)
    cfg_path = ckpt_path.parent / "config.json"
    h = AttrDict(json.loads(cfg_path.read_text()))
    base = build_causal_model(h)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base.load_state_dict(state["generator"], strict=True)
    base.eval()

    drop_nyquist = n_features == N_FEATURES_FULL - 1
    win = ConvFSENetWindowedONNX(base, T=1, drop_nyquist=drop_nyquist).eval()

    trunk = ConvFSENetTrunk(n_features).eval()
    head = MaskHead(n_features).eval()
    with torch.no_grad():
        trunk.frontend.weight.copy_(win.frontend_conv.weight)
        trunk.frontend.bias.copy_(win.frontend_conv.bias)
        for dst, src in zip(trunk.blocks, win.blocks):
            dst.conv1x1.weight.copy_(src.conv1x1_folded.weight)
            dst.conv1x1.bias.copy_(src.conv1x1_folded.bias)
            dst.dconv.weight.copy_(src.dconv_folded.weight)
            dst.dconv.bias.copy_(src.dconv_folded.bias)
            dst.conv1x1_out.weight.copy_(src.conv1x1_out.weight)
        head.conv.weight.copy_(win.backend_conv.weight)
        head.conv.bias.copy_(win.backend_conv.bias)
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk, head
