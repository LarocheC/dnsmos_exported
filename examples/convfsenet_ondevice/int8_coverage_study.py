#!/usr/bin/env python3
"""How much of the DNSMOS loss graph can actually run in int8?

Motivation: the STM32N6 defect we hit lives at the int8<->float boundary of
*software* epochs (see ST_BUG_REPORT.md). A graph with no such boundary cannot
trip it. Separately, every tensor moved from fp32 to int8 is 4x less memory and
4x less bus traffic on a part where the fp32 elementwise tail sets the peak.

So: which ops in this graph are float only because nobody quantized them, and
which are float because they *cannot* be int8?

The shipped recipe quantizes Conv/Gemm/MatMul plus the peak-setting Mul/Sub.
That leaves ~22 MB of fp32 traffic in the 0.25 s graph, of which about three
quarters is pure data movement (Concat / Reshape / Unsqueeze) and equality
masks — ops that do no arithmetic at all, and for which int8 is not an
approximation:

  * `Concat`/`Reshape`/`Unsqueeze`/`Slice`/`Squeeze` only move bytes.
  * `Equal` on two int8 tensors sharing a scale is EXACT — arguably better
    than the fp32 comparison it replaces, which is why the repo has a
    `repair_mask_consistency` pass to stop quantization from breaking it.
  * `MaxPool`/`AveragePool`/`Relu`/`Clip` are standard QDQ ops.

Genuinely irreducible: `Log` and `Reciprocal` (transcendental / division),
which would need a lookup table or a fixed-point reformulation. In this graph
they account for ~3% of the float traffic on small tensors.

This script measures the actual trade: for each candidate op set it quantizes,
lints, and reports gradient fidelity against the fp32 reference plus the
remaining fp32 traffic. Gradient fidelity is the gate that matters — the repo
already found that quantizing *every* `Mul` zeroes the gradient and `Add`
severely degrades it, so "quantize everything" is not free.

    python int8_coverage_study.py --window-s 0.25
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

# Data-movement + standard QDQ ops that carry the bulk of the float traffic.
# Deliberately excludes Mul and Add: measured to destroy / severely degrade the
# gradient when applied indiscriminately (see export.ELEMENTWISE_QUANT_OPS).
MOVEMENT_OPS = ("Reshape", "Unsqueeze", "Squeeze", "Slice", "Transpose")
POOL_OPS = ("MaxPool", "AveragePool", "Relu", "Clip")

RECIPES = {
    "shipped":            (),
    "+movement":          MOVEMENT_OPS,
    "+movement+pool":     MOVEMENT_OPS + POOL_OPS,
}


def float_traffic(path: Path) -> tuple[float, dict[str, float], int]:
    """(total fp32 MB, per-op MB, node count) for tensors that are NOT QDQ-fused.

    A node whose every consumer is a QuantizeLinear gets folded into an int8
    kernel by the compiler and never materializes as float on device; anything
    else does.
    """
    m = onnx.load(str(path))
    inf = shape_inference.infer_shapes(m, strict_mode=False)
    vi = {v.name: v for v in
          list(inf.graph.value_info) + list(inf.graph.output) + list(inf.graph.input)}

    def sz(t: str) -> int:
        v = vi.get(t)
        if v is None:
            return 0
        d = [x.dim_value if x.HasField("dim_value") else 1
             for x in v.type.tensor_type.shape.dim]
        return int(np.prod(d)) if d else 0

    consumers: dict[str, list] = defaultdict(list)
    for n in m.graph.node:
        for i in n.input:
            consumers[i].append(n)

    per_op: dict[str, float] = defaultdict(float)
    nodes = 0
    for n in m.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        outs = consumers.get(n.output[0], [])
        if outs and all(c.op_type == "QuantizeLinear" for c in outs):
            continue                                   # fused to int8
        per_op[n.op_type] += sz(n.output[0]) * 4 / 2 ** 20
        nodes += 1
    return sum(per_op.values()), dict(per_op), nodes


@contextmanager
def _registered(op_types):
    """Temporarily register extra ops against ORT's generic QDQ handler."""
    from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
    from onnxruntime.quantization.registry import QDQRegistry

    saved = {op: QDQRegistry.get(op) for op in op_types}
    try:
        for op in op_types:
            QDQRegistry.setdefault(op, QDQOperatorBase)
        yield
    finally:
        for op, prev in saved.items():
            if prev is None:
                QDQRegistry.pop(op, None)
            else:
                QDQRegistry[op] = prev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-s", type=float, default=0.25)
    ap.add_argument("--out-dir", type=Path, default=HERE / "artifacts_crop")
    ap.add_argument("--cache", type=Path, default=HERE / "artifacts" / "vbd_cache.npz")
    ap.add_argument("--calib-clips", type=int, default=8)
    ap.add_argument("--eval-clips", type=int, default=6)
    ap.add_argument("--exclude-depth", type=int, default=2)
    args = ap.parse_args()

    from crop_study import cropped_config, grad_cosines
    from dnsmos_trainable.export import quantize_qdq
    from dnsmos_trainable.verify import DNSMOS_DEVICE_OPS, check_device_constraints
    from onnxruntime.quantization.shape_inference import quant_pre_process

    cfg = cropped_config(args.window_s)
    tag = f"crop{args.window_s:g}s".replace(".", "p")
    fp32 = args.out_dir / f"{tag}_fp32.onnx"
    if not fp32.exists():
        raise SystemExit(f"missing {fp32} — run crop_study.py --window-s {args.window_s} first")

    clips = np.load(args.cache)["clips"]
    W_VEC = np.array([0, 0, -1], np.float32)
    calib = [{"wav": c[: cfg.input_len].astype(np.float32)[None], "w": W_VEC}
             for c in clips[: args.calib_clips]]
    evalc = clips[args.calib_clips: args.calib_clips + args.eval_clips]

    pre = args.out_dir / f"{tag}_cov_pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    convs = [n.name for n in onnx.load(str(pre)).graph.node if n.op_type == "Conv"]
    bwd = list(reversed(convs[7:]))

    print(f"{args.window_s:g} s window, {cfg.n_frames} frames — int8 coverage study")
    print(f"{'recipe':18s} {'fp32 MB':>8s} {'nodes':>6s} {'grad cos':>9s} {'cos min':>8s} "
          f"{'|g8|/|g32|':>10s}  lint")
    for name, extra in RECIPES.items():
        out = args.out_dir / f"{tag}_cov_{name.replace('+','p')}.onnx"
        with _registered(extra):
            from dnsmos_trainable import export as _exp
            saved = _exp.QUANT_OP_TYPES
            try:
                _exp.QUANT_OP_TYPES = list(saved) + list(extra)
                quantize_qdq(pre, out, calib, extra_exclude=bwd[: args.exclude_depth],
                             preprocessed=True, quantize_elementwise=True)
            finally:
                _exp.QUANT_OP_TYPES = saved
        mb, per_op, nodes = float_traffic(out)
        rep = grad_cosines(fp32, out, evalc, threads=1)
        problems = check_device_constraints(out, max_opset=13, allowed=DNSMOS_DEVICE_OPS)
        print(f"{name:18s} {mb:8.2f} {nodes:6d} {rep['cosine_mean']:9.3f} "
              f"{rep['cosine_min']:8.3f} {rep['ratio_mean']:10.2f}  "
              f"{'OK' if not problems else str(problems)[:40]}")
        top = sorted(per_op.items(), key=lambda kv: -kv[1])[:5]
        print("     remaining float: " +
              ", ".join(f"{o} {v:.2f}MB" for o, v in top if v > 0.01))
    pre.unlink(missing_ok=True)

    print("\nIrreducible without a LUT or fixed-point rewrite: Log, Reciprocal "
          "(transcendental / division).")


if __name__ == "__main__":
    main()
