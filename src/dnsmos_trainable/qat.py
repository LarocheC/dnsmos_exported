"""Lightweight quantization-aware fine-tuning (self-distillation).

PTQ leaves the SIG output above the delta gate: quantization noise is spread
evenly across the conv stack and the global-max pooling propagates worst-case
(not average) error. This module inserts fake-quant ops mirroring the ORT QDQ
placement (per-channel symmetric int8 weights, per-tensor asymmetric int8
activations after each ReLU — MaxPool/ReduceMax are quantization-transparent,
so they need no extra sites), freezes the activation ranges from a percentile
calibration, and fine-tunes the body weights so the *quantized* outputs match
the fp32 teacher (the transplanted model itself).

The QAT weights are used ONLY for int8 artifacts; fp32 artifacts keep the
exact transplanted weights.
"""

import copy

import torch
from torch import nn

from dnsmos_trainable.model import DnsmosModel

ACT_SITES = ["feat", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "b1", "b2", "raw"]


def _act_qparams(rmin: float, rmax: float) -> tuple[float, int]:
    rmin, rmax = min(rmin, 0.0), max(rmax, 0.0)  # zero must be representable
    scale = max((rmax - rmin) / 255.0, 1e-12)
    zp = int(round(-128 - rmin / scale))
    return scale, max(-128, min(127, zp))


def fq_act(x: torch.Tensor, scale: float, zp: int) -> torch.Tensor:
    return torch.fake_quantize_per_tensor_affine(x, scale, zp, -128, 127)


def fq_weight(w: torch.Tensor) -> torch.Tensor:
    """Per-channel symmetric int8 ([-127, 127], zero_point 0), axis 0."""
    flat = w.detach().reshape(w.shape[0], -1)
    scales = (flat.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
    zps = torch.zeros_like(scales, dtype=torch.int32)
    return torch.fake_quantize_per_channel_affine(w, scales, zps, 0, -127, 127)


@torch.no_grad()
def calibrate_act_ranges(
    model: DnsmosModel, wavs: torch.Tensor, percentile: float = 99.9
) -> dict[str, tuple[float, float]]:
    """Percentile activation ranges per site over a calibration batch."""
    lo_q, hi_q = (100.0 - percentile) / 100.0, percentile / 100.0
    samples: dict[str, list[torch.Tensor]] = {s: [] for s in ACT_SITES}

    def grab(site: str, t: torch.Tensor) -> None:
        flat = t.reshape(-1)
        if flat.numel() > 200_000:
            idx = torch.randint(0, flat.numel(), (200_000,))
            flat = flat[idx]
        samples[site].append(flat)

    body = model.body
    for i in range(wavs.shape[0]):
        x = model.frontend(wavs[i : i + 1])
        grab("feat", x)
        for k in range(1, 8):
            x = torch.relu(getattr(body, f"conv{k}")(x))
            grab(f"a{k}", x)
            if k in (4, 5, 6):
                x = body.pool(x)
        x = torch.amax(x, dim=(2, 3))
        x = torch.relu(body.fc1(x))
        grab("b1", x)
        x = torch.relu(body.fc2(x))
        grab("b2", x)
        raw = body.fc3(x)
        grab("raw", raw)

    ranges = {}
    for site, chunks in samples.items():
        allv = torch.cat(chunks)
        ranges[site] = (
            float(torch.quantile(allv, lo_q)),
            float(torch.quantile(allv, hi_q)),
        )
    return ranges


class QatDnsmosModel(nn.Module):
    """DnsmosModel body with frozen-range fake quantization for fine-tuning."""

    def __init__(self, model: DnsmosModel, act_ranges: dict[str, tuple[float, float]]) -> None:
        super().__init__()
        self.model = copy.deepcopy(model)
        for p in self.model.frontend.parameters():
            p.requires_grad_(False)  # frontend is excluded from quantization
        for p in self.model.body.parameters():
            p.requires_grad_(True)
        self.qparams = {site: _act_qparams(*act_ranges[site]) for site in ACT_SITES}

    def trainable_parameters(self):
        return [p for p in self.model.body.parameters() if p.requires_grad]

    def forward(self, wav: torch.Tensor):
        body, poly = self.model.body, self.model.poly
        q = self.qparams
        x = self.model.frontend(wav)
        x = fq_act(x, *q["feat"])
        for k in range(1, 8):
            conv = getattr(body, f"conv{k}")
            x = torch.relu(nn.functional.conv2d(x, fq_weight(conv.weight), conv.bias, padding=1))
            x = fq_act(x, *q[f"a{k}"])
            if k in (4, 5, 6):
                x = body.pool(x)
        x = torch.amax(x, dim=(2, 3))
        x = torch.relu(nn.functional.linear(x, fq_weight(body.fc1.weight), body.fc1.bias))
        x = fq_act(x, *q["b1"])
        x = torch.relu(nn.functional.linear(x, fq_weight(body.fc2.weight), body.fc2.bias))
        x = fq_act(x, *q["b2"])
        raw = nn.functional.linear(x, fq_weight(body.fc3.weight), body.fc3.bias)
        raw = fq_act(raw, *q["raw"])
        return raw, poly(raw)
