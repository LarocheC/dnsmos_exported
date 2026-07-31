"""Int8 artifact tests.

The shipping int8 path is `qdq_writer.write_qdq_from_sim` (the sim's own
min/max grids, float tail) — these tests gate THAT path with the same gates
as scripts/export_int8.py. The ORT quantize_static path is exercised only
structurally/behaviorally: its percentile recalibration is documented as
unsuitable for this max-pooling model and never ships the forward artifact.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from dnsmos_trainable.export import export_forward, make_calibration_batches, quantize_qdq
from dnsmos_trainable.qat import QatDnsmosModel, calibrate_act_ranges
from dnsmos_trainable.qdq_writer import write_qdq_from_sim
from dnsmos_trainable.verify import (
    check_device_constraints,
    int8_delta_report,
    make_synthetic_batch,
    ort_session,
)

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def act_ranges(transplanted):
    # Mirror scripts/export_int8.py exactly: seed, calibration data, order.
    torch.manual_seed(0)
    calib = torch.from_numpy(make_synthetic_batch(16, seed=100))
    return calibrate_act_ranges(transplanted, calib)


@pytest.fixture(scope="module")
def int8_source_model(official_path):
    """QAT weights when available (the shipping configuration), else transplant."""
    from dnsmos_trainable import load_transplanted

    for name in ("dnsmos_qat_int8_ft.pt", "dnsmos_qat_int8.pt"):
        path = ROOT / "models" / name
        if path.exists():
            return load_transplanted(path), True
    from dnsmos_trainable.model import DnsmosModel
    from dnsmos_trainable.transplant import transplant

    model = DnsmosModel()
    model.load_state_dict(transplant(official_path), strict=False)
    model.eval()
    return model, False


@pytest.fixture(scope="module")
def diy_artifacts(int8_source_model, transplanted, act_ranges, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("int8")
    model, is_qat = int8_source_model
    ref = export_forward(transplanted, tmp / "ref_fp32.onnx", device=False)
    src = export_forward(model, tmp / "src_fp32.onnx", device=False)
    src_dev = export_forward(model, tmp / "src_dev.onnx", device=True)
    out = write_qdq_from_sim(src, tmp / "int8.onnx", act_ranges, float_tail=True)
    out_dev = write_qdq_from_sim(src_dev, tmp / "int8_dev.onnx", act_ranges, float_tail=True)
    return {"ref": ref, "int8": out, "int8_dev": out_dev, "is_qat": is_qat}


def test_shipping_gates(diy_artifacts):
    """Same gates as scripts/export_int8.py, on the artifact family that ships."""
    evalb = make_synthetic_batch(50, seed=999)
    rep = int8_delta_report(diy_artifacts["ref"], diy_artifacts["int8"], evalb)
    if diy_artifacts["is_qat"]:
        assert all(m < 0.05 for m in rep["mean"]), rep
        p95_gates = (0.18, 0.10, 0.10)
        assert all(p < g for p, g in zip(rep["p95"], p95_gates)), rep
    else:
        # Transplant-only fallback: SIG is documented to need QAT.
        assert rep["mean"][1] < 0.05 and rep["mean"][2] < 0.05, rep


def test_device_int8_constraints_and_parity(diy_artifacts):
    problems = check_device_constraints(diy_artifacts["int8_dev"])
    assert not problems, problems
    # Desktop and device int8 artifacts must broadly agree (same weights and
    # grids, but different exporters/opsets: values sitting on int8 rounding
    # boundaries occasionally flip, and max-pooling propagates the flips).
    evalb = make_synthetic_batch(10, seed=555)
    rep = int8_delta_report(diy_artifacts["int8"], diy_artifacts["int8_dev"], evalb)
    assert max(rep["p95"]) < 0.05, rep
    assert max(rep["max"]) < 0.12, rep


def test_writer_matches_sim(int8_source_model, act_ranges, diy_artifacts):
    """The DIY artifact must track the fake-quant sim it claims to deploy."""
    model, _ = int8_source_model
    sim = QatDnsmosModel(model, act_ranges, float_tail=True)
    evalb = make_synthetic_batch(8, seed=777)
    with torch.no_grad():
        sim_mos = sim(torch.from_numpy(evalb))[1].numpy()
    sess = ort_session(diy_artifacts["int8"])
    mos = sess.run(None, {"wav": evalb})[1]
    # Small divergence allowed: int32 bias rounding + pool re-quantization.
    assert np.abs(mos - sim_mos).max() < 0.05, np.abs(mos - sim_mos).max()


def test_quantize_static_structural(transplanted, tmp_path):
    """The ORT quantize_static path stays functional: QDQ inserted, runs, shrinks."""
    import onnx

    fp32 = export_forward(transplanted, tmp_path / "q_fp32.onnx", device=False)
    calib = make_calibration_batches(make_synthetic_batch(8, seed=100))
    int8 = quantize_qdq(fp32, tmp_path / "q_int8.onnx", calib)  # MinMax default
    m = onnx.load(str(int8))
    ops = [n.op_type for n in m.graph.node]
    assert ops.count("QuantizeLinear") >= 10
    assert ops.count("DequantizeLinear") >= 17
    assert ops.count("Conv") == 7
    assert int8.stat().st_size < 0.75 * fp32.stat().st_size
    # Runs and produces plausible scores.
    sess = ort_session(int8)
    mos = sess.run(None, {"wav": make_synthetic_batch(2, seed=1)})[1]
    assert np.isfinite(mos).all() and (0.5 < mos).all() and (mos < 5.5).all()
