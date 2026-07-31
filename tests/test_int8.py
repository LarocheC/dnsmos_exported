from pathlib import Path

import numpy as np
import pytest

from dnsmos_trainable.export import export_forward, make_calibration_batches, quantize_qdq
from dnsmos_trainable.verify import (
    check_device_constraints,
    int8_delta_report,
    make_synthetic_batch,
)

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def int8_source_model(official_path):
    """QAT weights when available (the shipping configuration), else transplant."""
    from dnsmos_trainable import load_transplanted

    qat = ROOT / "models" / "dnsmos_qat_int8.pt"
    if qat.exists():
        return load_transplanted(qat), True
    from dnsmos_trainable.model import DnsmosModel
    from dnsmos_trainable.transplant import transplant

    model = DnsmosModel()
    model.load_state_dict(transplant(official_path), strict=False)
    model.eval()
    return model, False


def test_int8_forward_gates(int8_source_model, transplanted, tmp_path):
    from onnxruntime.quantization import CalibrationMethod

    model, is_qat = int8_source_model
    fp32_ref = export_forward(transplanted, tmp_path / "ref_fp32.onnx", device=False)
    fp32_src = export_forward(model, tmp_path / "src_fp32.onnx", device=False)
    calib = make_calibration_batches(make_synthetic_batch(16, seed=100))
    int8 = quantize_qdq(
        fp32_src, tmp_path / "int8.onnx", calib,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"percentile": 99.99},
    )
    rep = int8_delta_report(fp32_ref, int8, make_synthetic_batch(50, seed=999))
    if is_qat:
        assert all(m < 0.05 for m in rep["mean"]), rep
        assert all(p < 0.10 for p in rep["p95"]), rep
    else:
        # PTQ-only fallback: BAK/OVRL still gate; SIG is documented to need QAT.
        assert rep["mean"][1] < 0.05 and rep["mean"][2] < 0.05, rep

    # int8 payload should be materially smaller than fp32.
    assert int8.stat().st_size < 0.75 * fp32_src.stat().st_size


def test_int8_device_artifact_constraints(int8_source_model, tmp_path):
    from onnxruntime.quantization import CalibrationMethod

    model, _ = int8_source_model
    fp32_dev = export_forward(model, tmp_path / "src_dev.onnx", device=True)
    calib = make_calibration_batches(make_synthetic_batch(8, seed=100), rows=True)
    int8 = quantize_qdq(
        fp32_dev, tmp_path / "int8_dev.onnx", calib,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"percentile": 99.99},
    )
    problems = check_device_constraints(int8)
    assert not problems, problems


def test_int8_conv_actually_quantized(int8_source_model, tmp_path):
    """Guard against silent no-op quantization: QDQ pairs must surround convs."""
    import onnx

    from onnxruntime.quantization import CalibrationMethod

    model, _ = int8_source_model
    fp32_src = export_forward(model, tmp_path / "q_fp32.onnx", device=False)
    calib = make_calibration_batches(make_synthetic_batch(8, seed=100))
    int8 = quantize_qdq(
        fp32_src, tmp_path / "q_int8.onnx", calib,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"percentile": 99.99},
    )
    m = onnx.load(str(int8))
    ops = [n.op_type for n in m.graph.node]
    assert ops.count("QuantizeLinear") >= 10
    assert ops.count("DequantizeLinear") >= 17  # activations + per-channel weights
    assert ops.count("Conv") == 7
