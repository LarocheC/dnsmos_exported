"""Shared verification utilities: parity, gradients, quantization deltas."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from dnsmos_trainable.constants import INPUT_LEN, SR


def make_synthetic_batch(n: int, seed: int = 0, length: int = INPUT_LEN) -> np.ndarray:
    """Deterministic speech-like test signals: chirps, tones, and shaped noise.

    Exercises the model across realistic levels without shipping audio fixtures.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length) / SR
    batch = np.zeros((n, length), dtype=np.float32)
    for i in range(n):
        sig = np.zeros(length)
        # A few harmonics with vibrato, roughly speech F0 range.
        f0 = rng.uniform(90, 260)
        vib = 1.0 + 0.02 * np.sin(2 * np.pi * rng.uniform(3, 7) * t)
        for h in range(1, 6):
            sig += rng.uniform(0.1, 1.0) / h * np.sin(2 * np.pi * f0 * h * vib * t)
        # Amplitude modulation to mimic syllabic rhythm.
        sig *= 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * rng.uniform(2, 5) * t), 0, None)
        # Additive noise, occasionally pink-ish.
        noise = rng.standard_normal(length)
        if i % 2:
            noise = np.cumsum(noise) / np.sqrt(np.arange(1, length + 1))
        snr_scale = rng.uniform(0.02, 0.3)
        sig = sig / (np.abs(sig).max() + 1e-9) * rng.uniform(0.1, 0.5)
        sig += noise / (np.abs(noise).max() + 1e-9) * snr_scale
        batch[i] = sig.astype(np.float32)
    return batch


def ort_session(path: str | Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def compare_raw_scores(model: torch.nn.Module, sess: ort.InferenceSession, batch: np.ndarray) -> float:
    """Max abs diff between torch raw scores and an official-graph ORT session."""
    with torch.no_grad():
        raw, _ = model(torch.from_numpy(batch))
    input_name = sess.get_inputs()[0].name
    ref = sess.run(None, {input_name: batch})[0]
    return float(np.abs(raw.numpy() - ref).max())


# Intermediate tensors of the official graph, matched to our torch stages.
_OFFICIAL_STAGES = {
    "frames": ("Concat", 0),
    "log_feat": ("Div", 0),
    "conv1_relu": ("Relu", 0),
    "conv4_relu": ("Relu", 3),
    "pool3": ("MaxPool", 2),
    "global_max": ("ReduceMax", 0),
    "fc2_relu": ("Relu", 8),
}


def _official_intermediates(official_path: str | Path, batch: np.ndarray) -> dict[str, np.ndarray]:
    model = onnx.load(str(official_path))
    picked: dict[str, str] = {}
    counters: dict[str, int] = {}
    for node in model.graph.node:
        idx = counters.get(node.op_type, 0)
        counters[node.op_type] = idx + 1
        for stage, (op, want) in _OFFICIAL_STAGES.items():
            if node.op_type == op and idx == want:
                picked[stage] = node.output[0]
    for name in picked.values():
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        )
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input_1": batch})
    names = [o.name for o in sess.get_outputs()]
    by_name = dict(zip(names, outs))
    return {stage: by_name[tensor] for stage, tensor in picked.items()}


def _torch_intermediates(model: torch.nn.Module, batch: np.ndarray) -> dict[str, np.ndarray]:
    fe, body = model.frontend, model.body
    with torch.no_grad():
        x = torch.from_numpy(batch)
        frames = fe.framing(x)
        re, im = fe.stft(frames)
        feat = fe.logpower(re, im)
        h = feat.unsqueeze(1)
        c1 = torch.relu(body.conv1(h))
        c2 = torch.relu(body.conv2(c1))
        c3 = torch.relu(body.conv3(c2))
        c4 = torch.relu(body.conv4(c3))
        p1 = body.pool(c4)
        c5 = torch.relu(body.conv5(p1))
        p2 = body.pool(c5)
        c6 = torch.relu(body.conv6(p2))
        p3 = body.pool(c6)
        c7 = torch.relu(body.conv7(p3))
        gmax = torch.amax(c7, dim=(2, 3))
        f1 = torch.relu(body.fc1(gmax))
        f2 = torch.relu(body.fc2(f1))
    return {
        "frames": frames.numpy(),
        "log_feat": feat.numpy(),
        "conv1_relu": c1.numpy(),
        "conv4_relu": c4.numpy(),
        "pool3": p3.numpy(),
        "global_max": gmax.numpy(),
        "fc2_relu": f2.numpy(),
    }


def compare_intermediates(
    official_path: str | Path, model: torch.nn.Module, batch: np.ndarray
) -> dict[str, float]:
    """Stage-by-stage max abs diff torch vs official graph. NCHW/NHWC-aware."""
    ref = _official_intermediates(official_path, batch)
    ours = _torch_intermediates(model, batch)
    diffs: dict[str, float] = {}
    for stage, r in ref.items():
        o = ours[stage]
        if o.ndim == 4 and r.ndim == 4 and o.shape != r.shape:
            r = np.transpose(r, (0, 3, 1, 2))  # official CNN stages are NHWC-ish
        diffs[stage] = float(np.abs(o - r).max())
    return diffs


def finite_diff_check(
    sess: ort.InferenceSession,
    wav: np.ndarray,
    w: np.ndarray,
    n_coords: int = 20,
    eps: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """Central-difference check of grad_wav from a loss-graph ORT session.

    Returns relative errors per probed coordinate. fp32 finite differences are
    noisy; treat this as a sanity gate (rel ~5e-2), not the primary grad test.
    """
    input_names = [i.name for i in sess.get_inputs()]
    feed = {input_names[0]: wav, input_names[1]: w}
    outs = sess.run(None, feed)
    grad = outs[-1]
    mos_idx = 1  # outputs: raw, mos, grad
    rng = np.random.default_rng(seed)
    flat = wav.reshape(wav.shape[0], -1)
    gflat = grad.reshape(grad.shape[0], -1)
    rel_errs = []
    for _ in range(n_coords):
        b = rng.integers(0, flat.shape[0])
        j = rng.integers(0, flat.shape[1])
        for sign, store in ((1, "plus"), (-1, "minus")):
            pert = flat.copy()
            pert[b, j] += sign * eps
            outs_p = sess.run(None, {input_names[0]: pert.reshape(wav.shape), input_names[1]: w})
            if sign == 1:
                loss_plus = float((outs_p[mos_idx] * w).sum())
            else:
                loss_minus = float((outs_p[mos_idx] * w).sum())
        fd = (loss_plus - loss_minus) / (2 * eps)
        an = float(gflat[b, j])
        rel_errs.append(abs(fd - an) / max(abs(fd), abs(an), 1e-6))
    return np.array(rel_errs)


def _fwd_mos(sess, batch: np.ndarray, chunk: int) -> np.ndarray:
    """Run a forward artifact over flat [N, 144160] segments, adapting the
    feed to the artifact's I/O layout (flat batched or device rows B=1)."""
    inp = sess.get_inputs()[0]
    rows = len(inp.shape) == 3
    out = []
    if rows:
        for i in range(batch.shape[0]):
            feed = batch[i].reshape(1, inp.shape[1], inp.shape[2])
            out.append(sess.run(None, {inp.name: feed})[1])
    else:
        for i in range(0, batch.shape[0], chunk):
            out.append(sess.run(None, {inp.name: batch[i : i + chunk]})[1])
    return np.concatenate(out)


def int8_delta_report(
    fp32_path: str | Path, int8_path: str | Path, batch: np.ndarray, chunk: int = 8
) -> dict:
    """MOS deltas between fp32 and int8 forward artifacts (chunked: conv1
    activations are ~74 MB/segment, so large batches OOM in one session run).
    Accepts flat [B,144160] or device rows [1,901,160] artifacts on either side."""
    s32, s8 = ort_session(fp32_path), ort_session(int8_path)
    d = np.abs(_fwd_mos(s32, batch, chunk) - _fwd_mos(s8, batch, chunk))
    return {
        "mean": d.mean(axis=0).tolist(),
        "p95": np.percentile(d, 95, axis=0).tolist(),
        "max": d.max(axis=0).tolist(),
    }


# Op vocabulary allowed in STM32N6 device artifacts. Everything here is either
# in the ST Neural-ART mapping table (HW or documented SW fallback) or purely
# structural. Log is a documented float SW epoch (frontend only, by design).
STM32N6_ALLOWED_OPS = {
    "Add", "AveragePool", "Cast", "Clip", "Concat", "Constant", "Conv",
    "Equal", "Gemm", "GlobalAveragePool", "Log", "MatMul", "MaxPool", "Mul",
    "Pad", "Reciprocal", "ReduceMax", "ReduceMean", "Relu", "Reshape",
    "Slice", "Squeeze", "Sub", "Transpose", "Unsqueeze",
    "QuantizeLinear", "DequantizeLinear",
}


def check_op_vocabulary(model_path: str | Path, allowed: set[str] = STM32N6_ALLOWED_OPS) -> set[str]:
    """Return the set of ops in the graph that are NOT in the allowed set."""
    model = onnx.load(str(model_path))
    ops = {node.op_type for node in model.graph.node}
    return ops - allowed


def check_device_constraints(model_path: str | Path, max_opset: int = 13) -> list[str]:
    """Lint a device artifact against the ST Edge AI front-end constraints.

    Returns a list of violation strings (empty = clean): opset <= max_opset
    (ST's cap is 20; this project pins device exports to ST's recommended 13),
    static shapes, EVERY tensor dim < 65536 (interior tensors included — the
    ST front end rejects them the same as I/O), batch 1 on I/O, op vocabulary.
    """
    model = onnx.load(str(model_path))
    problems = []
    opset = {o.domain: o.version for o in model.opset_import}.get("", 0)
    if opset > max_opset:
        problems.append(f"opset {opset} > {max_opset}")
    extra = check_op_vocabulary(model_path)
    if extra:
        problems.append(f"ops outside STM32N6 vocabulary: {sorted(extra)}")
    io_names = {vi.name for vi in list(model.graph.input) + list(model.graph.output)}
    inferred = onnx.shape_inference.infer_shapes(model)
    all_vi = (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    )
    for vi in all_vi:
        dims = vi.type.tensor_type.shape.dim
        for d in dims:
            if d.dim_param or d.dim_value <= 0:
                if vi.name in io_names:
                    problems.append(f"{vi.name}: non-static dim")
            elif d.dim_value >= 65536:
                problems.append(f"{vi.name}: dim {d.dim_value} >= 65536")
        if vi.name in io_names and len(dims) > 1 and dims[0].dim_value != 1:
            problems.append(f"{vi.name}: batch != 1")
    return problems


def int8_grad_report(fp32_path: str | Path, int8_path: str | Path, batch: np.ndarray, w: np.ndarray) -> dict:
    """Gradient quality of an int8 loss graph vs its fp32 reference."""
    s32, s8 = ort_session(fp32_path), ort_session(int8_path)

    def run(sess):
        inp = sess.get_inputs()[0]
        w_name = sess.get_inputs()[1].name
        grads = []
        for i in range(batch.shape[0]):
            feed = batch[i : i + 1]
            if len(inp.shape) == 3:  # device rows layout
                feed = feed.reshape(1, inp.shape[1], inp.shape[2])
            out = sess.run(None, {inp.name: feed, w_name: w})
            grads.append(out[-1].ravel())
        return grads

    g32, g8 = run(s32), run(s8)
    cos, ratio = [], []
    for a, b in zip(g32, g8):
        cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        ratio.append(float(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-12)))
    cos, ratio = np.array(cos), np.array(ratio)
    return {
        "cosine_min": float(cos.min()),
        "cosine_mean": float(cos.mean()),
        "norm_ratio_min": float(ratio.min()),
        "norm_ratio_max": float(ratio.max()),
    }
