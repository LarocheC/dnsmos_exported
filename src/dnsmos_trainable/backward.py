"""Hand-written backward pass for DNSMOS as ordinary forward ops.

`DnsmosLossGraph` recomputes the forward while stashing what the VJPs need,
then chains vector-Jacobian products top-down. It never calls autograd — the
module *is* the backward — so `torch.onnx.export` produces a plain inference
graph whose last output happens to be dL/d(waveform), runnable on any ONNX
runtime (including inference-only NPU toolchains such as ST Neural-ART).

Two op-vocabulary modes:

- ``mode="device"`` (default): STM32N6-safe. No ScatterElements, ConvTranspose,
  Resize, Expand, Where, Greater, ReduceSum, or tensor-divisor Div. Conv input
  gradients use plain Conv with pre-flipped constant weights; max-pool routing
  uses equality masks (upsampled via concat-interleave) with tie counts built
  from AvgPool/ReduceMean + Clip + Reciprocal; ReLU/clamp masks use Equal; the
  log backward uses Reciprocal. Global-max ties are split evenly in BOTH
  modes; max-pool ties are split evenly in device mode but routed to the
  first argmax in ort mode (exact autograd semantics). Tie handling matters:
  with an int8-quantized forward, ties in max windows are the norm (coarse
  value grid), and overcounting them destroys the gradient direction.
- ``mode="ort"``: exact autograd semantics (index-scatter max-pool routing,
  tie-splitting global max) for desktop ONNX Runtime, where ScatterElements
  and ConvTranspose are cheap.

I/O layouts: ``flat`` is `[B, 144160]`; ``rows`` is `[1, 901, 160]` for the
STM32N6 front-end constraint that every tensor dim stay below 65536.
"""

import torch
from torch import nn
import torch.nn.functional as F

from dnsmos_trainable.constants import EPS, HOP, INPUT_LEN, LN10, N_FRAMES, N_ROWS
from dnsmos_trainable.model import DnsmosModel


def relu_mask(act: torch.Tensor, mode: str) -> torch.Tensor:
    """Exact ReLU gradient mask from the post-activation tensor.

    act == 0 exactly where pre-activation <= 0, so both modes are identical in
    value; "device" phrases it with Equal (NPU-mapped) instead of Greater.
    """
    if mode == "device":
        return 1.0 - (act == 0).to(act.dtype)
    return (act > 0).to(act.dtype)


def linear_bwd(g_y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """VJP of y = x @ W.T for nn.Linear weight [out, in]: g_x = g_y @ W."""
    return g_y @ weight


def poly_bwd(g_mos: torch.Tensor, raw: torch.Tensor, c2: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
    return g_mos * (2.0 * c2 * raw + c1)


def global_maxpool_bwd(g_y: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    """VJP of y = amax(x, dims=(2,3)). Equality-mask routing, ties split evenly.

    Both modes split the gradient across ties (matching torch.amax backward).
    Tie handling is not optional: with an int8-quantized forward, ties inside
    max reductions are ubiquitous (coarse value grid), and routing the full
    gradient to every tie decorrelates the input gradient completely. "device"
    phrases the tie count with ReduceMean/Reciprocal/Clip (NPU-mapped) instead
    of ReduceSum/Div (software fallback on ST Neural-ART).
    """
    mask = (x == y[:, :, None, None]).to(x.dtype)
    g = g_y[:, :, None, None]
    if mode == "ort":
        ties = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        return mask / ties * g
    count = mask.mean(dim=(2, 3), keepdim=True) * float(x.shape[2] * x.shape[3])
    return mask * g * torch.reciprocal(torch.clamp(count, min=1.0))


def _interleave2x(t: torch.Tensor, dim: int) -> torch.Tensor:
    """2x nearest upsample along `dim` via stack+reshape (Concat/Reshape only)."""
    reps = torch.stack([t, t], dim=dim + 1)
    shape = list(t.shape)
    shape[dim] *= 2
    return reps.reshape(shape)


def maxpool2d_bwd_device(g_y: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """VJP of y = max_pool2d(x, 2, 2) from basic ops (STM32N6-safe).

    Upsample y and g_y by 2x nearest (concat-interleave), zero-pad to x's odd
    spatial dims, route with an equality mask, and split the gradient evenly
    among within-window ties (per-window tie count via AvgPool*4, upsampled,
    Clip(min=1) so zero-padded edges stay finite, Reciprocal instead of Div).
    Tie splitting is essential once the forward is int8-quantized: the coarse
    value grid makes multi-cell ties the norm, and overcounting them wrecks
    the gradient direction. Padded rows/cols receive zero gradient by
    construction (g_up is zero there).
    """
    y_up = _interleave2x(_interleave2x(y, 2), 3)
    g_up = _interleave2x(_interleave2x(g_y, 2), 3)
    pad_h = x.shape[2] - y_up.shape[2]
    pad_w = x.shape[3] - y_up.shape[3]
    y_up = F.pad(y_up, (0, pad_w, 0, pad_h))
    g_up = F.pad(g_up, (0, pad_w, 0, pad_h))
    mask = (x == y_up).to(x.dtype)
    win_ties = F.avg_pool2d(mask, 2, 2) * 4.0
    ties_up = _interleave2x(_interleave2x(win_ties, 2), 3)
    ties_up = F.pad(ties_up, (0, pad_w, 0, pad_h))
    inv_ties = torch.reciprocal(torch.clamp(ties_up, min=1.0))
    return mask * g_up * inv_ties


def maxpool2d_bwd_ort(g_y: torch.Tensor, indices: torch.Tensor, input_shape: list[int]) -> torch.Tensor:
    """VJP of max_pool2d via the recorded argmax indices (exact autograd ties)."""
    b, c, h, w = input_shape
    canvas = torch.zeros(b, c, h * w, dtype=g_y.dtype, device=g_y.device)
    canvas = canvas.scatter_add(2, indices.flatten(2), g_y.flatten(2))
    return canvas.reshape(b, c, h, w)


def conv2d_input_grad_ort(g_y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """VJP of y = conv2d(x, W, padding=1) w.r.t. x, via ConvTranspose."""
    return F.conv_transpose2d(g_y, weight, stride=1, padding=1)


def conv2d_input_grad_device(g_y: torch.Tensor, flipped_weight: torch.Tensor) -> torch.Tensor:
    """Same VJP as plain Conv with pre-flipped weights [C_in, C_out, 3, 3]."""
    return F.conv2d(g_y, flipped_weight, stride=1, padding=1)


class DnsmosLossGraph(nn.Module):
    """(wav, w) -> (raw [B,3], mos [B,3], grad_wav) with L = sum_b w . mos_b.

    DNSMOS weights are frozen; only the gradient w.r.t. the input waveform is
    produced — the intended use is as a perceptual loss for an upstream model.
    Rebuild this module if the underlying DnsmosModel weights change (the
    flipped conv kernels are precomputed at construction).
    """

    def __init__(self, model: DnsmosModel, mode: str = "device", io_layout: str = "flat") -> None:
        super().__init__()
        if mode not in ("device", "ort"):
            raise ValueError(f"mode must be 'device' or 'ort', got {mode!r}")
        if io_layout not in ("flat", "rows"):
            raise ValueError(f"io_layout must be 'flat' or 'rows', got {io_layout!r}")
        self.mode = mode
        self.io_layout = io_layout
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        body = model.body
        for k in range(1, 8):
            w = getattr(body, f"conv{k}").weight.detach()
            self.register_buffer(f"flip{k}", w.flip(2, 3).transpose(0, 1).contiguous())

    def forward(self, wav: torch.Tensor, w: torch.Tensor):
        fe, body, poly = self.model.frontend, self.model.body, self.model.poly
        mode = self.mode

        # ---- forward recompute, stashing what the VJPs need ----
        if self.io_layout == "rows":
            frames = torch.cat([wav[:, : N_ROWS - 1, :], wav[:, 1:, :]], dim=2)
        else:
            b = wav.shape[0]
            a = wav[:, : INPUT_LEN - HOP].reshape(b, N_FRAMES, HOP)
            c = wav[:, HOP:].reshape(b, N_FRAMES, HOP)
            frames = torch.cat([a, c], dim=2)
        re = frames @ fe.stft.w_re.T
        im = frames @ fe.stft.w_im.T
        p = re * re + im * im
        pc = torch.clamp(p, min=EPS)
        feat = torch.log(pc) * (1.0 / LN10)
        h0 = feat.unsqueeze(1)

        acts = []  # post-ReLU conv activations a1..a7
        pools = []  # (pool_input, pool_output[, indices]) per pool
        x = h0
        pool_after = {4, 5, 6}  # pools follow conv4, conv5, conv6
        for k in range(1, 8):
            x = torch.relu(getattr(body, f"conv{k}")(x))
            acts.append(x)
            if k in pool_after:
                if mode == "ort":
                    pooled, idx = F.max_pool2d(x, 2, 2, return_indices=True)
                    pools.append((x, pooled, idx))
                else:
                    pooled = F.max_pool2d(x, 2, 2)
                    pools.append((x, pooled))
                x = pooled
        gmax = torch.amax(x, dim=(2, 3))
        b1 = torch.relu(body.fc1(gmax))
        b2 = torch.relu(body.fc2(b1))
        raw = body.fc3(b2)
        mos = poly(raw)

        # ---- backward: chain VJPs top-down ----
        g_mos = w.reshape(1, 3).to(wav.dtype) * torch.ones_like(raw)
        g_raw = poly_bwd(g_mos, raw, poly.c2, poly.c1)
        g_b2 = linear_bwd(g_raw, body.fc3.weight) * relu_mask(b2, mode)
        g_b1 = linear_bwd(g_b2, body.fc2.weight) * relu_mask(b1, mode)
        g_gmax = linear_bwd(g_b1, body.fc1.weight)
        g = global_maxpool_bwd(g_gmax, acts[6], gmax, mode)

        for k in range(7, 0, -1):
            g = g * relu_mask(acts[k - 1], mode)
            if mode == "device":
                g = conv2d_input_grad_device(g, getattr(self, f"flip{k}"))
            else:
                g = conv2d_input_grad_ort(g, getattr(body, f"conv{k}").weight)
            if k - 1 in (4, 5, 6):  # conv{k-1}'s output was pooled in the fwd
                entry = pools[{4: 0, 5: 1, 6: 2}[k - 1]]
                if mode == "ort":
                    pin, _pout, idx = entry
                    g = maxpool2d_bwd_ort(g, idx, list(pin.shape))
                else:
                    pin, pout = entry
                    g = maxpool2d_bwd_device(g, pin, pout)

        g_feat = g.squeeze(1)
        if mode == "device":
            clamp_mask = (pc == p).to(p.dtype)
            g_p = g_feat * (1.0 / LN10) * torch.reciprocal(pc) * clamp_mask
        else:
            clamp_mask = (p >= EPS).to(p.dtype)
            g_p = g_feat * (1.0 / LN10) / pc * clamp_mask
        g_re = 2.0 * re * g_p
        g_im = 2.0 * im * g_p
        g_frames = g_re @ fe.stft.w_re + g_im @ fe.stft.w_im

        if self.io_layout == "rows":
            g_a = g_frames[:, :, :HOP]
            g_b = g_frames[:, :, HOP:]
            grad = F.pad(g_a, (0, 0, 0, 1)) + F.pad(g_b, (0, 0, 1, 0))
        else:
            bsz = g_frames.shape[0]
            g_a = g_frames[:, :, :HOP].reshape(bsz, INPUT_LEN - HOP)
            g_b = g_frames[:, :, HOP:].reshape(bsz, INPUT_LEN - HOP)
            grad = F.pad(g_a, (0, HOP)) + F.pad(g_b, (HOP, 0))

        return raw, mos, grad
