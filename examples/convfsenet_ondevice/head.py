"""Trainable mask head: forward and hand-written backward, both ONNX-exportable.

Both graphs take the head's weights as *runtime inputs* rather than baked-in
initializers. That is what makes on-device training possible without
regenerating and reflashing the network: the optimizer keeps W and b in RAM and
feeds them in each call.

The cost of that choice, on the Neural-ART: a MatMul is HW-mapped only when its
second input is constant, so these two MatMuls run as software epochs on the
Cortex-M55. At 256x192x564 = 27.7 MMAC each and one update per 9.01 s window,
that is the right trade — the frozen trunk (1.44 MMAC *per frame*, 564 frames)
is what needs the NPU.

Shapes (T = mask columns per window, F = frequency bins):
    h      [1, 192, T]     trunk output
    W      [F, 192]        head weight
    b      [F]             head bias
    mask   [1, F, T]       sigmoid gain in (0, 1)
"""

from __future__ import annotations

import torch
from torch import nn


class HeadForward(nn.Module):
    """(h, W, b) -> mask. Equivalent to Conv1d(192->F, k=1) + Sigmoid."""

    def forward(self, h: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(W, h) + b.reshape(1, -1, 1)   # [1, F, T]
        return torch.sigmoid(z)


class HeadBackward(nn.Module):
    """(dmask, mask, h) -> (dW, db): the head's weight gradients.

    dz = dmask * mask * (1 - mask)        sigmoid backward, from the saved mask
    dW = dz @ h^T                          [F, T] x [T, 192]
    db = sum_t dz                          expressed as ReduceMean * T, because
                                           ReduceSum has no NPU path on the
                                           Neural-ART while ReduceMean does.

    No trunk activations are needed — only h, which the forward already
    produced, and the mask, which the forward already produced.
    """

    def forward(self, dmask: torch.Tensor, mask: torch.Tensor, h: torch.Tensor):
        dz = dmask * mask * (1.0 - mask)                       # [1, F, T]
        dW = torch.matmul(dz, h.transpose(1, 2)).squeeze(0)    # [F, 192]
        n_t = float(dz.shape[-1])
        db = (dz.mean(dim=-1) * n_t).squeeze(0)                # [F]
        return dW, db


def head_reference_loss(h: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
                        dmask: torch.Tensor) -> torch.Tensor:
    """Scalar whose dW/db equal HeadBackward's outputs — used by the autograd test.

    For any downstream loss L, dL/dW depends on the head only through
    dL/dmask = dmask, so <mask, dmask> reproduces exactly the same weight
    gradients without needing the real loss in the loop.
    """
    mask = HeadForward()(h, W, b)
    return (mask * dmask).sum()
