import numpy as np
import pytest
import torch

from dnsmos_trainable.backward import DnsmosLossGraph
from dnsmos_trainable.constants import HOP, INPUT_LEN, N_ROWS
from dnsmos_trainable.verify import make_synthetic_batch

W_VECTORS = [
    torch.tensor([0.0, 0.0, -1.0]),  # descend on -OVRL: the headline use case
    torch.tensor([1.0, 0.0, 0.0]),
    torch.tensor([0.3, -0.2, 0.7]),
]


def autograd_reference(model, wav: torch.Tensor, w: torch.Tensor):
    wav = wav.clone().requires_grad_(True)
    raw, mos = model(wav)
    loss = (mos * w).sum()
    (grad,) = torch.autograd.grad(loss, wav)
    return raw.detach(), mos.detach(), grad


@pytest.mark.parametrize("mode", ["device", "ort"])
def test_full_chain_vs_autograd(transplanted, mode):
    lg = DnsmosLossGraph(transplanted, mode=mode)
    for seed in range(5):
        wav = torch.from_numpy(make_synthetic_batch(1, seed=seed))
        for w in W_VECTORS:
            raw_ref, mos_ref, grad_ref = autograd_reference(transplanted, wav, w)
            with torch.no_grad():
                raw, mos, grad = lg(wav, w)
            assert torch.allclose(raw, raw_ref, atol=1e-6)
            assert torch.allclose(mos, mos_ref, atol=1e-6)
            rel = (grad - grad_ref).norm() / grad_ref.norm().clamp(min=1e-12)
            max_abs = (grad - grad_ref).abs().max()
            assert rel < 1e-4, f"seed={seed} w={w.tolist()} rel={rel:.2e}"
            assert max_abs < 1e-5, f"seed={seed} w={w.tolist()} max_abs={max_abs:.2e}"


def test_rows_layout_matches_flat(transplanted):
    lg_flat = DnsmosLossGraph(transplanted, mode="device", io_layout="flat")
    lg_rows = DnsmosLossGraph(transplanted, mode="device", io_layout="rows")
    wav = torch.from_numpy(make_synthetic_batch(1, seed=1))
    w = W_VECTORS[0]
    with torch.no_grad():
        raw_f, mos_f, grad_f = lg_flat(wav, w)
        raw_r, mos_r, grad_r = lg_rows(wav.reshape(1, N_ROWS, HOP), w)
    assert torch.equal(raw_f, raw_r)
    assert torch.equal(grad_f.reshape(1, N_ROWS, HOP), grad_r)


def test_tie_policy(transplanted):
    """Constant input maximizes ties; both modes must stay finite, and 'ort'
    must still match autograd exactly (it splits/routes ties the same way)."""
    wav = torch.full((1, INPUT_LEN), 0.25)
    w = W_VECTORS[0]
    _, _, grad_ref = autograd_reference(transplanted, wav, w)

    lg_ort = DnsmosLossGraph(transplanted, mode="ort")
    with torch.no_grad():
        _, _, grad_ort = lg_ort(wav, w)
    assert torch.isfinite(grad_ort).all()
    # amax splits ties evenly and maxpool routes to the first index in both
    # autograd and our 'ort' implementation.
    assert torch.allclose(grad_ort, grad_ref, atol=1e-6)

    lg_dev = DnsmosLossGraph(transplanted, mode="device")
    with torch.no_grad():
        _, _, grad_dev = lg_dev(wav, w)
    # Documented policy: device mode splits ties evenly (global max and
    # max-pool); on this tie-saturated input it need not equal autograd's
    # first-index max-pool routing, but must stay finite.
    assert torch.isfinite(grad_dev).all()


def test_maxpool_bwd_device_tie_splitting():
    """Direct gate on the even tie split (essential for int8 forwards)."""
    from dnsmos_trainable.backward import maxpool2d_bwd_device

    # One 2x2 window with a 3-way tie, one with a unique max; odd extra col.
    x = torch.tensor([[[[2.0, 2.0, 9.0],
                        [2.0, 1.0, 9.0]]]])
    y = torch.nn.functional.max_pool2d(x, 2, 2)  # -> [[[[2.0]]]]
    g_y = torch.tensor([[[[3.0]]]])
    g_x = maxpool2d_bwd_device(g_y, x, y)
    expect = torch.tensor([[[[1.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0]]]])
    assert torch.allclose(g_x, expect), g_x
    # Gradient mass conserved; padded (uncovered) column got nothing.
    assert g_x.sum() == g_y.sum()

    # Tie-free window must match autograd exactly.
    torch.manual_seed(3)
    x2 = torch.randn(1, 2, 6, 6, requires_grad=True)
    y2 = torch.nn.functional.max_pool2d(x2, 2, 2)
    g = torch.randn_like(y2)
    (ref,) = torch.autograd.grad(y2, x2, g)
    ours = maxpool2d_bwd_device(g, x2.detach(), y2.detach())
    assert torch.allclose(ours, ref, atol=1e-7)


def test_global_maxpool_bwd_tie_splitting():
    from dnsmos_trainable.backward import global_maxpool_bwd

    x = torch.tensor([[[[5.0, 5.0], [1.0, 5.0]],   # channel 0: 3-way tie
                       [[7.0, 0.0], [0.0, 0.0]]]])  # channel 1: unique max
    y = torch.amax(x, dim=(2, 3))
    g_y = torch.tensor([[6.0, 2.0]])
    for mode in ("device", "ort"):
        g_x = global_maxpool_bwd(g_y, x, y, mode)
        expect = torch.tensor([[[[2.0, 2.0], [0.0, 2.0]],
                                [[2.0, 0.0], [0.0, 0.0]]]])
        assert torch.allclose(g_x, expect), (mode, g_x)
        assert torch.allclose(g_x.sum(dim=(2, 3)), g_y)


def test_grad_magnitude_sane(transplanted):
    lg = DnsmosLossGraph(transplanted, mode="device")
    wav = torch.from_numpy(make_synthetic_batch(1, seed=3))
    with torch.no_grad():
        _, _, grad = lg(wav, torch.tensor([0.0, 0.0, -1.0]))
    assert grad.abs().max() < 1e3
    assert grad.norm() > 0
