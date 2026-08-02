"""ONNX export and QDQ quantization for all artifact variants.

Desktop artifacts: dynamo exporter, opset 20, dynamic batch, flat [B, 144160]
I/O. STM32N6 artifacts: legacy exporter, opset 13 (ST Edge AI recommends 13;
its cap is 20; the dynamo exporter cannot go below 18), static batch 1,
[1, 901, 160] I/O so every tensor dim stays below the device's 65536 limit.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import torch

from dnsmos_trainable.backward import DnsmosLossGraph
from dnsmos_trainable.constants import HOP, INPUT_LEN, N_ROWS  # official defaults
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

    cfg = getattr(model, "cfg", None)
    f_len = cfg.input_len if cfg else INPUT_LEN
    f_rows = cfg.n_rows if cfg else N_ROWS
    f_hop = cfg.hop if cfg else HOP

    class _Fwd(torch.nn.Module):
        def __init__(self, m: DnsmosModel, rows: bool) -> None:
            super().__init__()
            self.m = m
            self.rows = rows

        def forward(self, wav: torch.Tensor):
            if self.rows:
                # Build frames directly from the [1, 901, 160] rows layout —
                # never materialize a flat [1, 144160] tensor: the ST front
                # end rejects ANY tensor dim >= 65536, interior ones included.
                frames = torch.cat([wav[:, :-1, :], wav[:, 1:, :]], dim=2)
                re, im = self.m.frontend.stft(frames)
                feat = self.m.frontend.logpower(re, im).unsqueeze(1)
                raw = self.m.body(feat)
                return raw, self.m.poly(raw)
            return self.m(wav)

    if device:
        wrapper = _Fwd(model, rows=True)
        example = torch.zeros(1, f_rows, f_hop)
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
        example = torch.zeros(2, f_len)
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
    if device:
        _lint_device(out_path)
    feeds = {"wav": np.zeros(example.shape, dtype=np.float32)}
    _check_and_smoke(out_path, feeds)
    return out_path


def _lint_device(path: Path) -> None:
    from dnsmos_trainable.verify import check_device_constraints

    problems = check_device_constraints(path)
    if problems:
        raise RuntimeError(f"device constraint violations in {path.name}: {problems}")


def export_loss_graph(loss: DnsmosLossGraph, out_path: str | Path) -> Path:
    """Fused fwd+bwd artifact: (wav, w) -> (raw_scores, mos_scores, grad_wav)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loss = loss.eval()
    w = torch.tensor([0.0, 0.0, -1.0])

    cfg = getattr(loss, "cfg", None)
    n_rows = cfg.n_rows if cfg else N_ROWS
    hop = cfg.hop if cfg else HOP
    flat_len = cfg.input_len if cfg else INPUT_LEN
    if loss.io_layout == "rows":
        example = torch.zeros(1, n_rows, hop)
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
        example = torch.zeros(2, flat_len)
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
    if loss.io_layout == "rows":
        _lint_device(out_path)
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

# The backward's elementwise tail is what actually sets the activation peak:
# `relu_mask` materializes Equal -> Cast -> Sub -> Mul at conv1's full size, and
# none of those are in ORT's QDQ registry, so they hold the largest tensor in
# the graph in fp32 -- a 4x penalty on the number that decides whether the
# artifact fits on-chip.
#
# ORT will quantize them generically if they are registered against
# QDQOperatorBase, but only Mul and Sub may be. Measured on the 1 s crop, by
# gradient norm against the un-quantized-elementwise baseline (11.07):
#
#     +Sub   11.07   harmless
#     +Mul    0.00   DESTROYS the gradient when applied to every Mul
#     +Add    0.54   severe degradation
#
# Mul is both the one that matters for memory and the dangerous one. The
# resolution is to quantize elementwise nodes only where they set the peak
# (see `large_elementwise_nodes`): the small Muls deep in the fc backward
# carry the surviving gradient signal and cost nothing to leave in float.
ELEMENTWISE_QUANT_OPS = ("Mul", "Sub")


@contextmanager
def _elementwise_qdq_registered(op_types):
    """Temporarily teach ORT's QDQ quantizer about elementwise ops.

    `QDQRegistry` is module-global, so this restores it rather than leaving
    the process's quantizer permanently altered.
    """
    from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
    from onnxruntime.quantization.registry import QDQRegistry

    saved = {op: QDQRegistry.get(op) for op in op_types}
    try:
        for op in op_types:
            QDQRegistry[op] = QDQOperatorBase
        yield
    finally:
        for op, prev in saved.items():
            if prev is None:
                QDQRegistry.pop(op, None)
            else:
                QDQRegistry[op] = prev


def large_elementwise_nodes(model_path: str | Path, ops=ELEMENTWISE_QUANT_OPS
                            ) -> tuple[list[str], list[str]]:
    """Split elementwise nodes into (peak-setting, negligible).

    A node is peak-setting when its fp32 output would be larger than the whole
    graph's activation peak once quantized -- i.e. `elems > peak_elems / 4`.
    Those are exactly the tensors that stop the artifact fitting a pool; every
    other elementwise node is left in float, which is both safer for the
    gradient and free.
    """
    m = onnx.shape_inference.infer_shapes(onnx.load(str(model_path)))
    sizes = {}
    for vi in list(m.graph.value_info) + list(m.graph.output):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        sizes[vi.name] = int(np.prod(dims)) if dims else 0
    peak = max(sizes.values(), default=0)
    cutoff = peak / 4.0
    big, small = [], []
    for node in m.graph.node:
        if node.op_type not in ops:
            continue
        (big if sizes.get(node.output[0], 0) > cutoff else small).append(node.name)
    return big, small


def list_matmul_frontend_nodes(model_path: str | Path) -> list[str]:
    """Names of MatMul nodes belonging to the stft frontend (must stay float).

    Identified structurally: an input traces back — possibly through
    Transpose/Reshape/Identity (the exporter feeds `frames @ w.T` via a
    Transpose of the initializer) — to one of the two [161, 320]-shaped stft
    matrices (in either orientation).
    """
    model = onnx.load(str(model_path))

    stft_shapes = {(161, 320), (320, 161)}
    inits = {t.name: tuple(t.dims) for t in model.graph.initializer}
    const_shapes = {}
    by_output = {}
    for node in model.graph.node:
        for out in node.output:
            by_output[out] = node
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const_shapes[node.output[0]] = tuple(attr.t.dims)

    def source_shape(name: str, hops: int = 4):
        for _ in range(hops):
            shape = inits.get(name) or const_shapes.get(name)
            if shape is not None:
                return shape
            prod = by_output.get(name)
            if prod is None or prod.op_type not in ("Transpose", "Reshape", "Identity"):
                return None
            name = prod.input[0]
        return None

    names = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        for inp in node.input:
            if source_shape(inp) in stft_shapes:
                names.append(node.name)
                break
    return names


def list_shared_weight_matmuls(model_path: str | Path) -> list[str]:
    """MatMul nodes whose constant input is shared with a Gemm/Conv.

    The dynamo exporter deduplicates identical initializers, so a loss graph's
    backward MatMul `g @ W` reuses the forward Gemm's weight tensor. ORT's
    per-channel quantizer assigns those consumers different channel axes and
    its int32-bias scale adjustment then crashes (scale count vs bias length
    mismatch). These MatMuls are tiny; keep them float.
    """
    model = onnx.load(str(model_path))
    consumers: dict[str, set[str]] = {}
    for node in model.graph.node:
        for inp in node.input[1:]:
            consumers.setdefault(inp, set()).add(node.op_type)
    names = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        for inp in node.input:
            if consumers.get(inp, set()) - {"MatMul"}:
                names.append(node.name)
                break
    return names


_MASK_CHAIN_OPS = {"MaxPool", "ReduceMax", "Unsqueeze", "Reshape", "Concat", "Pad",
                   "QuantizeLinear", "DequantizeLinear", "Squeeze", "Transpose"}


def repair_mask_consistency(model_path: str | Path) -> int:
    """Make equality masks quantization-consistent after QDQ insertion.

    The loss graph routes max-pool/global-max gradients with `Equal(x, max_up)`
    masks. quantize_static rewires *some* consumers of a quantized tensor to
    the DequantizeLinear output and leaves others on the float instance, so a
    mask can end up comparing a float tensor against a dequantized max — which
    is never equal, silently zeroing the whole gradient. For each such Equal,
    this pass rebuilds the max-side chain (MaxPool/ReduceMax + reshaping)
    rooted at the exact tensor instance the mask reads, skipping Q/DQ pairs;
    max-of-the-same-tensor is then exact by construction. Returns the number
    of repaired masks.
    """
    model = onnx.load(str(model_path))
    graph = model.graph
    nodes = list(graph.node)
    by_output = {out: n for n in nodes for out in n.output}
    inits = {t.name for t in graph.initializer}

    def chain_to_source(tensor: str) -> tuple[list, str] | None:
        """Walk producer chain up through mask-chain ops; return (chain, source)."""
        chain = []
        cur = tensor
        while True:
            prod = by_output.get(cur)
            if prod is None:
                return None
            chain.append(prod)
            if prod.op_type in ("MaxPool", "ReduceMax"):
                return chain, prod.input[0]
            if prod.op_type not in _MASK_CHAIN_OPS:
                return None
            cur = prod.input[0]

    def strip_qdq(name: str) -> str:
        for suffix in ("_DequantizeLinear_Output", "_QuantizeLinear_Output"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    repaired = 0
    new_nodes: list = []
    for node in nodes:
        insert_before: list = []
        if node.op_type == "Equal":
            a, b = node.input[0], node.input[1]
            if b not in inits and by_output.get(b) is not None:
                walk = chain_to_source(b)
                if walk is not None:
                    chain, source = walk
                    has_qdq = any(
                        n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in chain
                    )
                    if source != a or has_qdq:
                        # Rebuild the chain rooted at `a`, skipping Q/DQ nodes.
                        keep = [n for n in reversed(chain) if n.op_type not in
                                ("QuantizeLinear", "DequantizeLinear")]
                        if keep and strip_qdq(source) == strip_qdq(a):
                            cur, prev_orig = a, source
                            for i, orig in enumerate(keep):
                                clone = onnx.NodeProto()
                                clone.CopyFrom(orig)
                                clone.name = f"{node.name}_maskchain_{i}"
                                # Rewire every input matching the original
                                # predecessor (Concat "stack" repeats it).
                                for j, inp in enumerate(clone.input):
                                    if inp == prev_orig or strip_qdq(inp) == strip_qdq(prev_orig):
                                        clone.input[j] = cur
                                out_name = f"{clone.name}_out"
                                prev_orig = orig.output[0]
                                del clone.output[:]
                                clone.output.append(out_name)
                                insert_before.append(clone)
                                cur = out_name
                            node.input[1] = cur
                            repaired += 1
        new_nodes.extend(insert_before)
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))
    return repaired


def quantize_qdq(
    fp32_path: str | Path,
    out_path: str | Path,
    calibration_batches: "list[dict[str, np.ndarray]]",
    extra_exclude: list[str] | None = None,
    per_channel: bool = True,
    calibrate_method=None,
    extra_options: dict | None = None,
    preprocessed: bool = False,
    quantize_elementwise: bool = False,
) -> Path:
    """Static QDQ int8 quantization (ss/sa scheme: symmetric per-channel
    weights, asymmetric per-tensor activations) matching the ST Neural-ART
    requirements as well as stock ORT expectations.

    `quantize_elementwise` additionally quantizes the peak-setting Mul/Sub
    nodes of the hand-written backward -- the difference between an artifact
    whose largest activation is fp32 and one where it is int8, and so between
    needing PSRAM and fitting `n6-noextmem`. See `ELEMENTWISE_QUANT_OPS` for
    why it is deliberately not applied to every elementwise node.
    """
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
    if preprocessed:
        # Caller already ran quant_pre_process — REQUIRED when extra_exclude
        # names were read from the graph: preprocessing renames nodes, so
        # names from the unprocessed graph would silently exclude nothing.
        pre_path = fp32_path
    else:
        pre_path = out_path.with_suffix(".preproc.onnx")
        # ORT's symbolic shape inference asserts on dynamo graphs with a
        # symbolic batch dim; plain ONNX shape inference still applies.
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=True)

    class _Reader(CalibrationDataReader):
        def __init__(self, batches):
            self._it = iter(batches)

        def get_next(self):
            return next(self._it, None)

    exclude = list_matmul_frontend_nodes(pre_path) + list_shared_weight_matmuls(pre_path)
    if extra_exclude:
        exclude += list(extra_exclude)

    op_types = list(QUANT_OP_TYPES)
    registered = ()
    if quantize_elementwise:
        registered = ELEMENTWISE_QUANT_OPS
        op_types += list(registered)
        big, small = large_elementwise_nodes(pre_path, registered)
        if not big:
            raise RuntimeError("quantize_elementwise: no peak-setting elementwise nodes found")
        exclude += small

    with _elementwise_qdq_registered(registered):
        quantize_static(
            str(pre_path),
            str(out_path),
            _Reader(calibration_batches),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            op_types_to_quantize=op_types,
            nodes_to_exclude=exclude,
            calibrate_method=calibrate_method,
            extra_options=extra_options or {},
        )
    if not preprocessed:
        pre_path.unlink(missing_ok=True)
    repaired = repair_mask_consistency(out_path)
    if repaired:
        print(f"repaired {repaired} quantization-inconsistent equality masks")
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
