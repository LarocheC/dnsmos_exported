"""Correctness gates for the ConvFSENet on-device-training demo.

The demo's claim is that an ONNX-only, autograd-free pipeline computes the
*same* mask-head gradients PyTorch would. These tests check every link of that
chain against torch, and the vendored architecture against upstream.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "convfsenet_ondevice"
sys.path.insert(0, str(DEMO))

import dsp  # noqa: E402
from convfsenet_arch import ConvFSENetTrunk, MaskHead, build_split  # noqa: E402
from head import HeadBackward, HeadForward, head_reference_loss  # noqa: E402

ECO8 = Path("/workspace/eco8-neaixt")
N_FEATURES = 256
SEG = 144160


# --------------------------------------------------------------------------
# DSP: the host-side half of the split (M55 / CMSIS-DSP on device)
# --------------------------------------------------------------------------

def test_stft_matches_torch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SEG) * 0.1
    ref = torch.stft(torch.from_numpy(x), 512, hop_length=256, win_length=512,
                     window=torch.hann_window(512, dtype=torch.float64), center=True,
                     normalized=False, return_complex=True).numpy()
    got = dsp.stft(x)
    assert got.shape == (257, dsp.num_frames(SEG))
    assert np.abs(got - ref).max() < 1e-9


def test_istft_matches_torch():
    rng = np.random.default_rng(1)
    Y = (rng.standard_normal((257, 200)) + 1j * rng.standard_normal((257, 200))) * 0.1
    ref = torch.istft(torch.from_numpy(Y), 512, hop_length=256, win_length=512,
                      window=torch.hann_window(512, dtype=torch.float64), center=True,
                      normalized=False, length=50000).numpy()
    got = dsp.istft(Y, length=50000)
    assert np.abs(got - ref).max() < 1e-9


def test_istft_adjoint_matches_autograd():
    """The adjoint of ISTFT is NOT the STFT; this is the gate that proves it."""
    rng = np.random.default_rng(2)
    T, length = 120, 30000
    Y = (rng.standard_normal((257, T)) + 1j * rng.standard_normal((257, T))) * 0.1
    gy = rng.standard_normal(length)

    Yr = torch.tensor(Y.real, requires_grad=True)
    Yi = torch.tensor(Y.imag, requires_grad=True)
    y = torch.istft(torch.complex(Yr, Yi), 512, hop_length=256, win_length=512,
                    window=torch.hann_window(512, dtype=torch.float64), center=True,
                    normalized=False, length=length)
    (y * torch.from_numpy(gy)).sum().backward()

    G = dsp.istft_adjoint(gy, T=T)
    assert np.abs(G.real - Yr.grad.numpy()).max() < 1e-9
    assert np.abs(G.imag - Yi.grad.numpy()).max() < 1e-9


def test_mask_grad_matches_autograd():
    rng = np.random.default_rng(3)
    X = (rng.standard_normal((257, 40)) + 1j * rng.standard_normal((257, 40)))
    m = rng.random((257, 40))
    gY = (rng.standard_normal((257, 40)) + 1j * rng.standard_normal((257, 40)))

    mt = torch.tensor(m, requires_grad=True)
    Y = torch.from_numpy(X) * mt
    (Y.real * torch.from_numpy(gY.real) + Y.imag * torch.from_numpy(gY.imag)).sum().backward()
    assert np.abs(dsp.mask_grad(X, gY) - mt.grad.numpy()).max() < 1e-10


def test_sisnr_grad_matches_autograd():
    rng = np.random.default_rng(4)
    est = rng.standard_normal(4000)
    ref = rng.standard_normal(4000)
    e = torch.tensor(est, requires_grad=True)
    r = torch.from_numpy(ref)
    u, s = e - e.mean(), r - r.mean()
    alpha = torch.dot(u, s) / (torch.dot(s, s) + 1e-10)
    t, n = alpha * s, u - alpha * s
    val = 10 * torch.log10((torch.dot(t, t) + 1e-10) / (torch.dot(n, n) + 1e-10))
    val.backward()
    value, grad = dsp.sisnr_and_grad(est, ref)
    assert abs(value - val.item()) < 1e-8
    assert np.abs(grad - e.grad.numpy()).max() < 1e-7


# --------------------------------------------------------------------------
# Architecture + head
# --------------------------------------------------------------------------

def test_split_param_counts():
    trunk, head = build_split(N_FEATURES, seed=0)
    total = sum(p.numel() for p in trunk.parameters()) + head.n_params
    # Upstream's windowed drop_nyquist model reports exactly this.
    assert total == 1_443_136
    assert head.n_params == 49_408
    assert trunk.context_l == 42


def test_head_backward_matches_autograd():
    torch.manual_seed(0)
    F, C, T = N_FEATURES, 192, 48
    h = torch.randn(1, C, T, dtype=torch.float64)
    W = torch.randn(F, C, dtype=torch.float64) * 0.05
    b = torch.randn(F, dtype=torch.float64) * 0.1
    dmask = torch.randn(1, F, T, dtype=torch.float64)

    Wv, bv = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
    dW_ref, db_ref = torch.autograd.grad(head_reference_loss(h, Wv, bv, dmask), [Wv, bv])
    dW, db = HeadBackward()(dmask, HeadForward()(h, W, b), h)
    assert (dW - dW_ref).norm() / dW_ref.norm() < 1e-12
    assert (db - db_ref).norm() / db_ref.norm() < 1e-12


def test_head_equals_conv1d():
    torch.manual_seed(1)
    head = MaskHead(N_FEATURES).double()
    h = torch.randn(1, 192, 32, dtype=torch.float64)
    W = head.conv.weight.detach().squeeze(-1)
    got = HeadForward()(h, W, head.conv.bias.detach())
    # matmul and Conv1d(k=1) are the same map but not contractually bit-identical.
    assert torch.allclose(got, head(h), atol=1e-12, rtol=0)


@pytest.mark.parametrize("B", [1, 2, 3])
def test_head_backward_reduces_over_batch(B):
    """Guard the squeeze-vs-sum trap: weight grads must sum over the batch."""
    torch.manual_seed(11)
    F, C, T = 8, 5, 6
    h = torch.randn(B, C, T, dtype=torch.float64)
    W = torch.randn(F, C, dtype=torch.float64)
    b = torch.randn(F, dtype=torch.float64)
    dmask = torch.randn(B, F, T, dtype=torch.float64)

    Wv, bv = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
    dW_ref, db_ref = torch.autograd.grad(head_reference_loss(h, Wv, bv, dmask), [Wv, bv])
    dW, db = HeadBackward()(dmask, HeadForward()(h, W, b), h)
    assert dW.shape == (F, C) and db.shape == (F,)
    assert torch.allclose(dW, dW_ref, atol=1e-12)
    assert torch.allclose(db, db_ref, atol=1e-12)


def test_stft_rejects_too_short_input():
    """The 'matches torch.stft' contract must not silently hold for buffers
    torch would refuse (numpy reflect-pads them happily)."""
    with pytest.raises(ValueError):
        dsp.stft(np.zeros(100))


@pytest.mark.skipif(not ECO8.exists(), reason="eco8-neaixt clone not present")
def test_vendored_arch_matches_upstream():
    """Bit-exactness against the upstream windowed model with shared weights."""
    import json

    sys.path.insert(0, str(ECO8))
    from common.env import AttrDict
    from convfsenet.model import build_causal_model
    from convfsenet.streaming import ConvFSENetWindowedONNX

    torch.manual_seed(0)
    h = AttrDict(json.loads((ECO8 / "configs" / "convfsenet.json").read_text()))
    base = build_causal_model(h).eval()
    T = 24
    win = ConvFSENetWindowedONNX(base, T=T, drop_nyquist=True).eval()

    trunk, head = ConvFSENetTrunk(N_FEATURES).eval(), MaskHead(N_FEATURES).eval()
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

    x = torch.rand(1, N_FEATURES, win.L + T) * 3.0
    with torch.no_grad():
        assert torch.equal(win.forward_mask_window(x), head(trunk(x)))
    assert win.L == trunk.context_l


@pytest.mark.skipif(not ECO8.exists(), reason="eco8-neaixt clone not present")
def test_checkpoint_loader_roundtrip(tmp_path):
    """`--checkpoint` (the path a real deployment uses) must reproduce upstream.

    Builds a synthetic eco8-style checkpoint so the loader is exercised without
    needing the gated published weights.
    """
    import json

    sys.path.insert(0, str(ECO8))
    from common.env import AttrDict
    from convfsenet.model import build_causal_model
    from convfsenet.streaming import ConvFSENetWindowedONNX

    from convfsenet_arch import load_from_eco8_checkpoint

    torch.manual_seed(7)
    cfg = json.loads((ECO8 / "configs" / "convfsenet.json").read_text())
    base = build_causal_model(AttrDict(cfg))
    ckpt = tmp_path / "g_best"
    torch.save({"generator": base.state_dict()}, ckpt)
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    trunk, head = load_from_eco8_checkpoint(ckpt, N_FEATURES, eco8_repo=ECO8)
    base.eval()
    T = 16
    win = ConvFSENetWindowedONNX(base, T=T, drop_nyquist=True).eval()
    x = torch.rand(1, N_FEATURES, win.L + T) * 3.0
    with torch.no_grad():
        assert torch.equal(win.forward_mask_window(x), head(trunk(x)))


# --------------------------------------------------------------------------
# The composite claim: ONNX-free-of-autograd == autograd
# --------------------------------------------------------------------------

def test_end_to_end_head_gradient_matches_autograd():
    """dW/db from (ISTFT-adjoint -> mask VJP -> HeadBackward) == autograd
    through (head -> mask -> complex mul -> ISTFT)."""
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    T = 64
    length = 256 * (T - 1)
    F, C = N_FEATURES, 192

    h_np = rng.standard_normal((1, C, T)) * 0.5
    W_np = rng.standard_normal((F, C)) * 0.05
    b_np = rng.standard_normal(F) * 0.1
    X_np = (rng.standard_normal((257, T)) + 1j * rng.standard_normal((257, T))) * 0.2
    g_np = rng.standard_normal(length)          # dL/d(enhanced), e.g. from DNSMOS

    # --- autograd reference: the whole chain in torch ---
    Wt = torch.tensor(W_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    ht = torch.tensor(h_np)
    mask = torch.sigmoid(torch.matmul(Wt, ht) + bt.reshape(1, -1, 1))[0]   # [F, T]
    mask_full = torch.cat([mask, torch.zeros(257 - F, T, dtype=mask.dtype)], dim=0)
    Y = torch.from_numpy(X_np) * mask_full
    y = torch.istft(Y, 512, hop_length=256, win_length=512,
                    window=torch.hann_window(512, dtype=torch.float64), center=True,
                    normalized=False, length=length)
    (y * torch.from_numpy(g_np)).sum().backward()

    # --- the demo's pipeline: no autograd anywhere ---
    gY = dsp.istft_adjoint(g_np, T=T)
    dmask_full = dsp.mask_grad(X_np, gY)
    dmask = torch.from_numpy(dmask_full[:F][None])
    mask_np = HeadForward()(torch.from_numpy(h_np), torch.from_numpy(W_np),
                            torch.from_numpy(b_np))
    dW, db = HeadBackward()(dmask, mask_np, torch.from_numpy(h_np))

    assert (dW - Wt.grad).norm() / Wt.grad.norm() < 1e-9, (dW - Wt.grad).abs().max()
    assert (db - bt.grad).norm() / bt.grad.norm() < 1e-9


# --------------------------------------------------------------------------
# Exported artifacts
# --------------------------------------------------------------------------

ARTIFACTS = DEMO / "artifacts"
_HAVE_ARTIFACTS = (ARTIFACTS / "convfsenet_trunk_int8.onnx").exists()
skip_artifacts = pytest.mark.skipif(
    not _HAVE_ARTIFACTS, reason="run examples/convfsenet_ondevice/export_demo_artifacts.py first")


@skip_artifacts
def test_onnx_head_graphs_match_torch():
    import onnxruntime as ort

    torch.manual_seed(2)
    fwd = ort.InferenceSession(str(ARTIFACTS / "convfsenet_head_fwd.onnx"),
                               providers=["CPUExecutionProvider"])
    bwd = ort.InferenceSession(str(ARTIFACTS / "convfsenet_head_bwd.onnx"),
                               providers=["CPUExecutionProvider"])
    T = fwd.get_inputs()[0].shape[2]
    F, C = N_FEATURES, 192
    h = torch.randn(1, C, T)
    W = torch.randn(F, C) * 0.05
    b = torch.randn(F) * 0.1
    dmask = torch.randn(1, F, T)

    mask_ref = HeadForward()(h, W, b)
    mask = fwd.run(None, {"h": h.numpy(), "W": W.numpy(), "b": b.numpy()})[0]
    assert np.abs(mask - mask_ref.numpy()).max() < 1e-5

    dW_ref, db_ref = HeadBackward()(dmask, mask_ref, h)
    dW, db = bwd.run(None, {"dmask": dmask.numpy(), "mask": mask_ref.numpy(), "h": h.numpy()})
    assert np.abs(dW - dW_ref.numpy()).max() < 1e-3
    assert np.abs(db - db_ref.numpy()).max() < 1e-3


@skip_artifacts
def test_onnx_trunk_matches_torch():
    import onnxruntime as ort

    trunk, _ = build_split(N_FEATURES, seed=0)
    state = torch.load(ARTIFACTS / "convfsenet_split.pt", map_location="cpu", weights_only=True)
    trunk.load_state_dict(state["trunk"])
    sess = ort.InferenceSession(str(ARTIFACTS / "convfsenet_trunk_fp32.onnx"),
                                providers=["CPUExecutionProvider"])
    W = sess.get_inputs()[0].shape[2]
    x = torch.rand(1, N_FEATURES, W) * 3.0
    with torch.no_grad():
        ref = trunk(x)
    got = sess.run(None, {"noisy_mag_window": x.numpy()})[0]
    assert np.abs(got - ref.numpy()).max() < 1e-4


@skip_artifacts
def test_int8_trunk_tracks_fp32():
    """int8 trunk must stay close enough to fp32 that the head still trains."""
    import onnxruntime as ort

    f32 = ort.InferenceSession(str(ARTIFACTS / "convfsenet_trunk_fp32.onnx"),
                               providers=["CPUExecutionProvider"])
    i8 = ort.InferenceSession(str(ARTIFACTS / "convfsenet_trunk_int8.onnx"),
                              providers=["CPUExecutionProvider"])
    W = f32.get_inputs()[0].shape[2]
    rng = np.random.default_rng(7)
    cos = []
    for k in range(4):
        x = (rng.random((1, N_FEATURES, W)) * 3.0).astype(np.float32)
        a = f32.run(None, {"noisy_mag_window": x})[0].ravel()
        b = i8.run(None, {"noisy_mag_window": x})[0].ravel()
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    assert min(cos) > 0.95, cos


@skip_artifacts
def test_compression_prologue_stays_float():
    """The exporter's headline fidelity feature, asserted rather than printed.

    If the eps-Add/Pow prologue were quantized, raw |STFT| would land on a
    coarse int8 grid and lose exactly the low-energy detail the compression
    exists to preserve. Structural check: the graph input must reach Add->Pow
    before any QuantizeLinear.
    """
    import onnx

    m = onnx.load(str(ARTIFACTS / "convfsenet_trunk_int8.onnx"))
    producers = {o: n for n in m.graph.node for o in n.output}
    consumers = {}
    for n in m.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    cur = m.graph.input[0].name
    chain = []
    for _ in range(4):
        nxt = consumers.get(cur, [])
        assert len(nxt) == 1, f"unexpected fan-out at {cur}"
        node = nxt[0]
        chain.append(node.op_type)
        if node.op_type == "QuantizeLinear":
            break
        cur = node.output[0]
    assert chain[:2] == ["Add", "Pow"], chain
    assert "QuantizeLinear" in chain, chain
    # ...and quantization must actually be present downstream.
    assert sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear") > 5


@skip_artifacts
def test_ondevice_loop_runs_and_improves():
    """Execute the shipping loop itself: run_trunk windowing, Nyquist handling,
    loss reshape and hinge sign are otherwise uncovered by the unit tests."""
    import importlib

    ort = pytest.importorskip("onnxruntime")
    loss_graph = ROOT / "artifacts" / "dnsmos_loss_fp32.onnx"
    if not loss_graph.exists():
        pytest.skip("run scripts/export_all_fp32.py first")

    mod = importlib.import_module("ondevice_train")
    eng = mod.OnDeviceEnhancer(ARTIFACTS, loss_graph, use_int8_trunk=True)
    noisy, _ = mod.make_noisy(0)
    X, mag = eng.analyse(noisy)
    h = eng.run_trunk(mag)
    assert h.shape == (1, 192, X.shape[1])

    W = np.zeros((eng.n_features, 192), dtype=np.float64)
    b = np.full(eng.n_features, 4.0, dtype=np.float64)
    w_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    opt = mod.NpAdam([W.shape, b.shape], lr=3e-3)

    mask = eng.mask_of(h, W, b)
    enhanced, _ = eng.synthesise(X, mask, len(noisy))
    # Pass-through init must reproduce the input up to the ~0.982 gain.
    assert np.corrcoef(enhanced, noisy)[0, 1] > 0.99
    mos_start, _ = eng.dnsmos(enhanced, w_vec)

    for _ in range(8):
        mask = eng.mask_of(h, W, b)
        enhanced, _ = eng.synthesise(X, mask, len(noisy))
        _, g = eng.dnsmos(enhanced, w_vec)
        gY = dsp.istft_adjoint(g.astype(np.float64), T=X.shape[1])
        dmask = dsp.mask_grad(X, gY)[: eng.n_features][None]
        dW, db = eng.head_grads(dmask, mask, h)
        uW, ub = opt.step([dW, db])
        W -= uW
        b -= ub

    mask = eng.mask_of(h, W, b)
    enhanced, _ = eng.synthesise(X, mask, len(noisy))
    mos_end, _ = eng.dnsmos(enhanced, w_vec)
    assert mos_end[2] > mos_start[2] + 0.1, (mos_start[2], mos_end[2])


@skip_artifacts
@pytest.mark.parametrize("name,kw", [
    ("convfsenet_trunk_fp32.onnx", dict(allow_dynamic_batch=True,
                                        batched_io={"noisy_mag_window", "h"})),
    ("convfsenet_trunk_int8.onnx", dict(allow_dynamic_batch=True,
                                        batched_io={"noisy_mag_window", "h"})),
    ("convfsenet_head_fwd.onnx", dict(batched_io={"h", "mask"})),
    ("convfsenet_head_bwd.onnx", dict(batched_io={"dmask", "mask", "h"})),
])
def test_artifacts_pass_device_lint(name, kw):
    from dnsmos_trainable.verify import check_device_constraints

    problems = check_device_constraints(ARTIFACTS / name, max_opset=17, **kw)
    assert not problems, problems
