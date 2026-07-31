import numpy as np
import onnx
import pytest
import torch

from dnsmos_trainable.backward import DnsmosLossGraph
from dnsmos_trainable.constants import HOP, INPUT_LEN, N_ROWS
from dnsmos_trainable.export import export_forward, export_loss_graph
from dnsmos_trainable.verify import finite_diff_check, make_synthetic_batch, ort_session

W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


@pytest.fixture(scope="module")
def artifacts(transplanted, tmp_path_factory):
    art = tmp_path_factory.mktemp("artifacts")
    paths = {
        "fwd": export_forward(transplanted, art / "fwd.onnx", device=False),
        "fwd_dev": export_forward(transplanted, art / "fwd_dev.onnx", device=True),
        "loss": export_loss_graph(
            DnsmosLossGraph(transplanted, mode="ort", io_layout="flat"), art / "loss.onnx"
        ),
        "loss_dev": export_loss_graph(
            DnsmosLossGraph(transplanted, mode="device", io_layout="rows"), art / "loss_dev.onnx"
        ),
    }
    return paths


def test_forward_artifact_matches_torch(artifacts, transplanted, synthetic_batch):
    sess = ort_session(artifacts["fwd"])
    for b in (1, 3):
        batch = synthetic_batch[:b]
        raw, mos = sess.run(None, {"wav": batch})
        with torch.no_grad():
            raw_t, mos_t = transplanted(torch.from_numpy(batch))
        assert np.abs(raw - raw_t.numpy()).max() < 1e-4
        assert np.abs(mos - mos_t.numpy()).max() < 1e-4


def test_device_forward_artifact(artifacts, transplanted, synthetic_batch):
    sess = ort_session(artifacts["fwd_dev"])
    wav = synthetic_batch[:1].reshape(1, N_ROWS, HOP)
    raw, mos = sess.run(None, {"wav": wav})
    with torch.no_grad():
        raw_t, _ = transplanted(torch.from_numpy(synthetic_batch[:1]))
    assert np.abs(raw - raw_t.numpy()).max() < 1e-4


@pytest.mark.parametrize("key,rows", [("loss", False), ("loss_dev", True)])
def test_loss_artifact_matches_torch(artifacts, transplanted, synthetic_batch, key, rows):
    mode = "device" if rows else "ort"
    lg = DnsmosLossGraph(transplanted, mode=mode, io_layout="rows" if rows else "flat")
    sess = ort_session(artifacts[key])
    wav = synthetic_batch[:1]
    feed_wav = wav.reshape(1, N_ROWS, HOP) if rows else wav
    raw, mos, grad = sess.run(None, {"wav": feed_wav, "w": W})
    with torch.no_grad():
        raw_t, mos_t, grad_t = lg(torch.from_numpy(feed_wav), torch.from_numpy(W))
    assert np.abs(raw - raw_t.numpy()).max() < 1e-4
    rel = np.linalg.norm(grad - grad_t.numpy()) / max(np.linalg.norm(grad_t.numpy()), 1e-12)
    assert rel < 1e-3, f"{key}: ORT vs torch grad rel {rel:.2e}"


def test_loss_dynamic_batch(artifacts, synthetic_batch):
    sess = ort_session(artifacts["loss"])
    for b in (1, 3):
        raw, mos, grad = sess.run(None, {"wav": synthetic_batch[:b], "w": W})
        assert raw.shape == (b, 3) and grad.shape == (b, INPUT_LEN)


def test_finite_differences_on_ort(artifacts, synthetic_batch):
    sess = ort_session(artifacts["loss"])
    rel_errs = finite_diff_check(sess, synthetic_batch[:1], W, n_coords=20)
    # fp32 central differences are noisy; median must be solid, allow outliers.
    assert np.median(rel_errs) < 5e-2, f"median FD rel err {np.median(rel_errs):.2e}"
    assert (rel_errs < 0.2).mean() >= 0.8


# Ops absent from the ST Neural-ART mapping table (or known import-blockers)
# must never appear in device artifacts. Log stays: it is documented as a
# float SW epoch (the frontend is float by design).
FORBIDDEN_DEVICE_OPS = {
    "ScatterElements", "ScatterND", "Scatter", "ConvTranspose", "Where",
    "Expand", "Resize", "Greater", "GreaterOrEqual", "Less", "ReduceSum",
    "Div", "Shape", "ConstantOfShape", "Range", "NonZero", "Loop", "If",
}


@pytest.mark.parametrize("key", ["fwd_dev", "loss_dev"])
def test_device_artifact_constraints(artifacts, key):
    model = onnx.load(str(artifacts[key]))
    opset = {o.domain: o.version for o in model.opset_import}[""]
    assert opset == 13
    ops = {n.op_type for n in model.graph.node}
    assert not (ops & FORBIDDEN_DEVICE_OPS), ops & FORBIDDEN_DEVICE_OPS
    # Every I/O dim static and < 65536 (ST front-end constraint).
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        assert all(0 < d < 65536 for d in dims), (vi.name, dims)
    # Batch is 1 on every rank>1 tensor (w is a rank-1 [3] weight vector).
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if len(dims) > 1:
            assert dims[0].dim_value == 1, vi.name
