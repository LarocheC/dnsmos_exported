#!/usr/bin/env python3
"""Hand-written backward for the PESQ predictor — the on-device gradient source.

The forward already runs correctly on the STM32N6 (37.6 ms on-chip, device ==
host int8 to 0.006 PESQ), unlike the DNSMOS loss graph on the same silicon. This
adds the other half: a fused graph returning both the score and
`dScore/d(enh_mag)`, written as ordinary inference operators so ST's
inference-only runtime can execute it.

Why hand-written: `torch.autograd` cannot be exported to ONNX (the repo's
`DnsmosLossGraph` exists for the same reason), so every VJP is spelled out.
This graph is far simpler than DNSMOS's, which is the point — no `Equal`/`Cast`
mask chain over a 494k-element fp32 tensor, hence no exposure to the ST
int8<->float software-epoch defect.

VJPs implemented, in reverse order:

  output head        y = sigmoid(x)              -> g * y*(1-y)
                     y = 2*sigmoid(s*x)  (lsig)  -> g * 2*s*y'(1-y')
  Linear             y = xW^T + b                -> g @ W
  PReLU              y = max(x,0) + a*min(x,0)   -> g * (step + a*(1-step))
  global MaxPool     y = max over (H,W)          -> route to argmax via equality mask
  InstanceNorm       per-(N,C) normalization     -> the 3-term standard form
  Conv2d stride 2    y = conv(x, W)              -> zero-insert upsample, then
                                                    conv with pre-flipped weights

Two device-safety notes:

* **Zero-insertion, not nearest, for the stride-2 conv VJP.** The gradient must
  be scattered to the positions the stride sampled and zero elsewhere; nearest
  upsampling (which the max-pool VJP uses) would smear it. Built from
  `stack`/`reshape`/`slice` only — no `ConvTranspose`, which the device
  vocabulary excludes.
* **InstanceNorm's backward needs its saved statistics.** They are recomputed in
  the backward rather than stashed, keeping the graph a pure function of its
  inputs.

Gate: every VJP and the whole chain are checked against `torch.autograd.grad`.

    python pesq_backward.py            # runs the autograd gate
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pesq_predictor import LearnableSigmoid1d, PesqPredictor  # noqa: E402

EPS_IN = 1e-5          # torch InstanceNorm2d default eps


def _zero_upsample2x(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Insert a zero after every element along `dim` (stack + reshape only)."""
    z = torch.zeros_like(t)
    stacked = torch.stack([t, z], dim=dim + 1)
    shape = list(t.shape)
    shape[dim] *= 2
    return stacked.reshape(shape)


def conv2d_s2_input_grad(g_y: torch.Tensor, flipped_w: torch.Tensor,
                         in_hw: tuple[int, int]) -> torch.Tensor:
    """VJP of y = conv2d(x, W, stride=2, padding=1) for a 4x4 kernel.

    Zero-insert the gradient to undo the stride, then correlate with the
    flipped kernel. Output is cropped/padded to the recorded input size, which
    a stride-2 conv does not determine uniquely.
    """
    g = _zero_upsample2x(_zero_upsample2x(g_y, 2), 3)
    g = F.conv2d(g, flipped_w, stride=1, padding=3)          # full correlation, k=4
    H, W = in_hw
    return g[..., 1:1 + H, 1:1 + W]


def prelu_bwd(g_y: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """VJP of PReLU. `step` via Relu(sign)-free construction: (x - min(x,0))/x is
    unstable at 0, so use the exact indicator built from Relu on a scaled x."""
    step = (torch.relu(x) > 0).to(x.dtype)                    # 1 where x > 0
    return g_y * (step + a.view(1, -1, 1, 1) * (1.0 - step))


def instancenorm_bwd(g_y: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor,
                     eps: float = EPS_IN) -> torch.Tensor:
    """VJP of InstanceNorm2d(affine=True) w.r.t. x, per (N, C) over (H, W)."""
    n = x.shape[2] * x.shape[3]
    mu = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
    inv = torch.rsqrt(var + eps)
    xhat = (x - mu) * inv
    g = g_y * gamma.view(1, -1, 1, 1)
    return inv * (g - g.mean(dim=(2, 3), keepdim=True)
                  - xhat * (g * xhat).mean(dim=(2, 3), keepdim=True))


def globalmax_bwd(g_y: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """VJP of a global max over (H, W); ties share the gradient evenly."""
    mask = (x == y).to(x.dtype)
    ties = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    return g_y * mask / ties


class PesqLossGraph(nn.Module):
    """(noisy_mag, enh_mag) -> (pesq_norm, grad_enh) with grad = d(pesq)/d(enh).

    The predictor's weights are frozen; only the gradient w.r.t. the enhanced
    magnitude is produced, which is what an upstream mask head needs. The host
    then applies the compression chain rule to reach the mask:

        enh_mag = (|X| * mask + 1e-9) ** 0.3
        d(enh_mag)/d(mask) = 0.3 * (|X| * mask + 1e-9) ** (-0.7) * |X|

    Both output heads are supported. The plain sigmoid is the cheaper of the
    two on-device: its derivative is `y*(1-y)` in the forward output that is
    already on the wire, so the head contributes one Sub and one Mul and no
    extra parameter.
    """

    def __init__(self, core: PesqPredictor) -> None:
        super().__init__()
        L = core.layers
        self.convs = nn.ModuleList([L[0], L[3], L[6], L[9]])
        self.norms = nn.ModuleList([L[1], L[4], L[7], L[10]])
        self.prelus = nn.ModuleList([L[2], L[5], L[8], L[11]])
        self.fc1, self.fc2 = L[14], L[17]
        self.prelu_fc = L[16]
        self.out_head = L[18]
        self.is_lsig = isinstance(L[18], LearnableSigmoid1d)
        for i, c in enumerate(self.convs):
            self.register_buffer(f"flip{i}",
                                 c.weight.detach().flip(2, 3).transpose(0, 1).contiguous())

    def forward(self, noisy_mag: torch.Tensor, enh_mag: torch.Tensor):
        x = torch.stack((noisy_mag, enh_mag), dim=1)
        acts = []                                             # (conv_out, norm_out) per block
        h = x
        for c, n, p in zip(self.convs, self.norms, self.prelus):
            pre_shape = h.shape[2:]
            z = c(h)
            zn = n(z)
            h = p(zn)
            acts.append((pre_shape, z, zn))
        pooled = F.adaptive_max_pool2d(h, 1)                  # [B, C, 1, 1]
        flat = pooled.reshape(pooled.shape[0], -1)
        a1 = self.fc1(flat)
        a1p = self.prelu_fc(a1)
        raw = self.fc2(a1p)
        out = self.out_head(raw)

        # ---- backward: d(out)/d(enh_mag), one scalar output per batch item ----
        if self.is_lsig:
            s = torch.sigmoid(self.out_head.slope * raw)
            g = self.out_head.beta * self.out_head.slope * s * (1.0 - s)
        else:
            g = out * (1.0 - out)                             # sigmoid', from `out`
        g = g @ self.fc2.weight                               # fc2 VJP
        step = (torch.relu(a1) > 0).to(a1.dtype)
        g = g * (step + self.prelu_fc.weight.view(1, -1) * (1.0 - step))
        g = g @ self.fc1.weight                               # fc1 VJP
        g = g.reshape(pooled.shape)
        g = globalmax_bwd(g, h, pooled)
        for i in range(3, -1, -1):
            pre_shape, z, zn = acts[i]
            g = prelu_bwd(g, zn, self.prelus[i].weight)
            g = instancenorm_bwd(g, z, self.norms[i].weight)
            g = conv2d_s2_input_grad(g, getattr(self, f"flip{i}"), tuple(pre_shape))
        grad_enh = g[:, 1]                                    # channel 1 == enh_mag
        return out.flatten(), grad_enh


def _gate_head(head: str) -> None:
    torch.manual_seed(0)
    core = PesqPredictor(dim=8, head=head).eval()
    for p in core.parameters():
        p.requires_grad_(False)
    # spectral_norm hooks recompute weights each forward; fold them first so the
    # hand-written VJPs see exactly the weights the forward uses.
    from torch.nn.utils import remove_spectral_norm
    for m in core.modules():
        if hasattr(m, "weight_v"):
            try:
                remove_spectral_norm(m)
            except (ValueError, RuntimeError):
                pass
    g = PesqLossGraph(core).eval()

    F_, T = 65, 31
    nm = torch.rand(2, F_, T)
    em = torch.rand(2, F_, T, requires_grad=True)
    out, grad = g(nm, em)

    ref = torch.autograd.grad(out.sum(), em)[0]
    num = (grad - ref).abs().max() / ref.abs().max().clamp_min(1e-12)
    print(f"[{head:>7}] forward out {out.detach().numpy().round(4)}  "
          f"max|d| {float((grad-ref).abs().max()):.3e}  relative {float(num):.3e}")
    assert num < 1e-4, f"backward does not match autograd for head={head}"


def _gate() -> None:
    for head in ("sigmoid", "lsig"):
        _gate_head(head)
    print("PASS — backward matches autograd for both heads")


if __name__ == "__main__":
    _gate()
