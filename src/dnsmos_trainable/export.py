"""ONNX export and QDQ quantization for all artifact variants.

Desktop artifacts: dynamo exporter, opset 20, dynamic batch, flat [B, 144160]
I/O. STM32N6 artifacts: legacy exporter, opset 13 (ST Edge AI recommends 13;
its cap is 20; the dynamo exporter cannot go below 18), static batch 1,
[1, 901, 160] I/O so every tensor dim stays below the device's 65536 limit.
"""

from pathlib import Path

import numpy as np
import onnx
import torch

from dnsmos_trainable.backward import DnsmosLossGraph
from dnsmos_trainable.constants import HOP, INPUT_LEN, N_ROWS
from dnsmos_trainable.model import DnsmosModel

DESKTOP_OPSET = 20
DEVICE_OPSET = 13


def _check_and_smoke(path: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    import onnxruntime as ort

    onnx.checker.check_model(str(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def _postprocess(path: Path, simplify: bool) -> None:
    """Embed weights into a single file; optionally constant-fold shape chains.

    The tracer emits Shape->Gather->Concat->Reshape machinery for static
    shapes; on STM32N6 those become software epochs (Shape/ConstantOfShape are
    not in the Neural-ART mapping table at all), so device artifacts are
    simplified until the op set is purely structural + compute.
    """
    model = onnx.load(str(path))  # pulls in any external .data
    if simplify:
        import onnxsim

        model, ok = onnxsim.simplify(model)
        if not ok:
            raise RuntimeError(f"onnxsim could not validate simplified {path}")
    data_file = path.with_suffix(path.suffix + ".data")
    onnx.save(model, str(path))
    if data_file.exists():
        data_file.unlink()


def export_forward(model: DnsmosModel, out_path: str | Path, device: bool = False) -> Path:
    """Forward-only artifact: wav -> (raw_scores, mos_scores)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()

    class _Fwd(torch.nn.Module):
        def __init__(self, m: DnsmosModel, rows: bool) -> None:
            super().__init__()
            self.m = m
            self.rows = rows

        def forward(self, wav: torch.Tensor):
            if self.rows:
                wav = wav.reshape(1, INPUT_LEN)
            return self.m(wav)

    if device:
        wrapper = _Fwd(model, rows=True)
        example = torch.zeros(1, N_ROWS, HOP)
        torch.onnx.export(
            wrapper,
            (example,),
            str(out_path),
            dynamo=False,
            opset_version=DEVICE_OPSET,
            input_names=["wav"],
            output_names=["raw_scores", "mos_scores"],
        )
    else:
        wrapper = _Fwd(model, rows=False)
        example = torch.zeros(2, INPUT_LEN)
        torch.onnx.export(
            wrapper,
            (example,),
            str(out_path),
            dynamo=True,
            opset_version=DESKTOP_OPSET,
            input_names=["wav"],
            output_names=["raw_scores", "mos_scores"],
            dynamic_shapes={"wav": {0: torch.export.Dim("batch", min=1, max=4096)}},
        )
    _postprocess(out_path, simplify=device)
    feeds = {"wav": np.zeros(example.shape, dtype=np.float32)}
    _check_and_smoke(out_path, feeds)
    return out_path


def export_loss_graph(loss: DnsmosLossGraph, out_path: str | Path) -> Path:
    """Fused fwd+bwd artifact: (wav, w) -> (raw_scores, mos_scores, grad_wav)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loss = loss.eval()
    w = torch.tensor([0.0, 0.0, -1.0])

    if loss.io_layout == "rows":
        example = torch.zeros(1, N_ROWS, HOP)
        torch.onnx.export(
            loss,
            (example, w),
            str(out_path),
            dynamo=False,
            opset_version=DEVICE_OPSET,
            input_names=["wav", "w"],
            output_names=["raw_scores", "mos_scores", "grad_wav"],
        )
    else:
        example = torch.zeros(2, INPUT_LEN)
        torch.onnx.export(
            loss,
            (example, w),
            str(out_path),
            dynamo=True,
            opset_version=DESKTOP_OPSET,
            input_names=["wav", "w"],
            output_names=["raw_scores", "mos_scores", "grad_wav"],
            dynamic_shapes={
                "wav": {0: torch.export.Dim("batch", min=1, max=4096)},
                "w": None,
            },
        )
    _postprocess(out_path, simplify=loss.io_layout == "rows")
    feeds = {
        "wav": np.zeros(example.shape, dtype=np.float32),
        "w": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    _check_and_smoke(out_path, feeds)
    return out_path


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

# Ops eligible for QDQ insertion. The frontend/backward tails are protected by
# node exclusion lists computed from the actual artifact, never hardcoded.
QUANT_OP_TYPES = ["Conv", "Gemm", "MatMul"]


def list_matmul_frontend_nodes(model_path: str | Path) -> list[str]:
    """Names of MatMul nodes belonging to the stft frontend (must stay float).

    Identified structurally: their constant input is one of the two
    [161, 320]-shaped stft matrices (in either orientation).
    """
    model = onnx.load(str(model_path))
    from onnx import numpy_helper

    stft_shapes = {(161, 320), (320, 161)}
    inits = {t.name: tuple(t.dims) for t in model.graph.initializer}
    # Constant nodes can also carry the weight (dynamo export style).
    const_shapes = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const_shapes[node.output[0]] = tuple(attr.t.dims)
    names = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        for inp in node.input:
            shape = inits.get(inp) or const_shapes.get(inp)
            if shape in stft_shapes:
                names.append(node.name)
                break
    return names


def quantize_qdq(
    fp32_path: str | Path,
    out_path: str | Path,
    calibration_batches: "list[dict[str, np.ndarray]]",
    extra_exclude: list[str] | None = None,
    per_channel: bool = True,
    calibrate_method=None,
    extra_options: dict | None = None,
) -> Path:
    """Static QDQ int8 quantization (ss/sa scheme: symmetric per-channel
    weights, asymmetric per-tensor activations) matching the ST Neural-ART
    requirements as well as stock ORT expectations."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if calibrate_method is None:
        calibrate_method = CalibrationMethod.MinMax

    fp32_path, out_path = Path(fp32_path), Path(out_path)
    pre_path = out_path.with_suffix(".preproc.onnx")
    # ORT's symbolic shape inference asserts on dynamo graphs with a symbolic
    # batch dim; plain ONNX shape inference (still applied) is sufficient here.
    quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=True)

    class _Reader(CalibrationDataReader):
        def __init__(self, batches):
            self._it = iter(batches)

        def get_next(self):
            return next(self._it, None)

    exclude = list_matmul_frontend_nodes(pre_path)
    if extra_exclude:
        exclude += list(extra_exclude)

    quantize_static(
        str(pre_path),
        str(out_path),
        _Reader(calibration_batches),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        op_types_to_quantize=QUANT_OP_TYPES,
        nodes_to_exclude=exclude,
        calibrate_method=calibrate_method,
        extra_options=extra_options or {},
    )
    pre_path.unlink(missing_ok=True)
    onnx.checker.check_model(str(out_path))
    return out_path


def make_calibration_batches(
    wavs: np.ndarray, w: np.ndarray | None = None, rows: bool = False
) -> list[dict[str, np.ndarray]]:
    """Per-segment feed dicts for calibration ([N] segments -> N feeds)."""
    feeds = []
    for i in range(wavs.shape[0]):
        wav = wavs[i : i + 1]
        if rows:
            wav = wav.reshape(1, N_ROWS, HOP)
        feed = {"wav": wav.astype(np.float32)}
        if w is not None:
            feed["w"] = w.astype(np.float32)
        feeds.append(feed)
    return feeds
