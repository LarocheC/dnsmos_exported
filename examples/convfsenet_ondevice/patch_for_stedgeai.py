#!/usr/bin/env python3
"""Make a DNSMOS loss-graph artifact compilable by ST Edge AI Core 4.0.1.

The repo's export-time lint (`verify.check_device_constraints`) approximates
the ST front end; real `stedgeai generate` runs found five graph-level gaps
plus one quant-JSON gap (see ONBOARD_RESULTS.md "Graph patches required").
This script applies the graph-level patches in one pass and verifies bit-exact
parity against the unpatched artifact:

  1. Slice sentinels (-1 / INT64_MAX ends) -> explicit in-range bounds.
  2. Rank-1 `w [3]` input + in-graph Reshape -> `w [1,3]` input, Reshape dropped.
  3. Pad ops -> Concat with zeros initializers; zeros in quantized regions are
     pre-quantized (int8 at the sibling tensor's zero-point, shared-scale DQ).
  4. Clip min/max routed through DequantizeLinear -> folded constants.

(The dynamo-exporter patches — allowzero/Shape/Expand — are not needed for
rows-layout opset-13 exports, which is what this script expects. The quant-JSON
BOOL strip happens at stedgeai time: shim `atonn` per ONBOARD_RESULTS.md.)

Typical use, starting from a crop_study run:

    python patch_for_stedgeai.py \
        --rows-fp32 artifacts_crop/crop0p25s_fp32_stm32n6.onnx \
        --flat-int8 artifacts_crop/crop0p25s_int8_d2.onnx \
        --out artifacts_crop/crop0p25s_int8_d2_rows_dev.onnx

The flat int8 artifact is used only as the parity reference for the quantized
rows export this script produces (same calibration clips, same exclusion
depth, elementwise on).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

TRANSPARENT = {"Unsqueeze", "Reshape", "Concat", "Slice", "Squeeze", "Transpose"}


def normalize_slices(m: onnx.ModelProto) -> int:
    g = m.graph
    inf = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    shapes = {v.name: [d.dim_value if d.HasField("dim_value") else None
                       for d in v.type.tensor_type.shape.dim]
              for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output)}
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    new_inits, patched = [], 0
    for n in g.node:
        if n.op_type != "Slice":
            continue
        shp = shapes.get(n.input[0])
        starts, ends = consts.get(n.input[1]), consts.get(n.input[2])
        axes = (consts.get(n.input[3]) if len(n.input) > 3 and n.input[3]
                else np.arange(len(starts)) if starts is not None else None)
        if shp is None or any(s is None for s in shp) or starts is None or ends is None or axes is None:
            continue
        ns, ne = [], []
        for s, e, ax in zip(starts.tolist(), ends.tolist(), np.atleast_1d(axes).tolist()):
            d = shp[ax]
            ns.append(s + d if s < 0 else min(s, d))
            ne.append(e + d if e < 0 else min(e, d))
        if ns == starts.tolist() and ne == ends.tolist():
            continue
        sn, en = f"{n.name}_starts_norm", f"{n.name}_ends_norm"
        new_inits += [numpy_helper.from_array(np.array(ns, np.int64), sn),
                      numpy_helper.from_array(np.array(ne, np.int64), en)]
        n.input[1], n.input[2] = sn, en
        patched += 1
    g.initializer.extend(new_inits)
    return patched


def promote_w(m: onnx.ModelProto) -> int:
    g = m.graph
    resh = [n for n in g.node if n.op_type == "Reshape" and n.input[0] == "w"]
    if not resh:
        return 0
    assert len(resh) == 1
    out = resh[0].output[0]
    keep = [n for n in g.node if n is not resh[0]]
    del g.node[:]
    g.node.extend(keep)
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp == out:
                n.input[i] = "w"
    for inp in g.input:
        if inp.name == "w":
            del inp.type.tensor_type.shape.dim[:]
            for v in (1, 3):
                d = inp.type.tensor_type.shape.dim.add()
                d.dim_value = v
    return 1


def pads_to_concats(m: onnx.ModelProto) -> int:
    g = m.graph
    inf = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    shapes = {v.name: [d.dim_value if d.HasField("dim_value") else None
                       for d in v.type.tensor_type.shape.dim]
              for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output)}
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    new_nodes, new_inits, replaced = [], [], 0
    for n in g.node:
        if n.op_type != "Pad":
            new_nodes.append(n)
            continue
        mode = next((a.s.decode() for a in n.attribute if a.name == "mode"), "constant")
        pads, shp = consts.get(n.input[1]), shapes.get(n.input[0])
        cval = consts.get(n.input[2]) if len(n.input) > 2 and n.input[2] else np.float32(0.0)
        assert mode == "constant" and pads is not None and shp is not None and float(cval) == 0.0, n.name
        rank = len(shp)
        begins, ends = pads[:rank].tolist(), pads[rank:].tolist()
        cur, cur_shape = n.input[0], list(shp)
        for ax in range(rank):
            for pos, amt in (("begin", begins[ax]), ("end", ends[ax])):
                if amt == 0:
                    continue
                zshape = list(cur_shape)
                zshape[ax] = amt
                zname = f"{n.name}_zeros_ax{ax}_{pos}"
                new_inits.append(numpy_helper.from_array(np.zeros(zshape, np.float32), zname))
                out = f"{n.name}_cc_ax{ax}_{pos}"
                ins = [zname, cur] if pos == "begin" else [cur, zname]
                new_nodes.append(helper.make_node("Concat", ins, [out], name=out, axis=ax))
                cur = out
                cur_shape[ax] += amt
        new_nodes[-1].output[0] = n.output[0]
        replaced += 1
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)
    return replaced


def quantize_pad_zeros(m: onnx.ModelProto) -> int:
    """Concat in a quantized region must not mix fp32 zeros with int8 data."""
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    byout = {o: n for n in g.node for o in n.output}

    def find_dq(name, depth=0):
        n = byout.get(name)
        if n is None or depth > 12:
            return None
        if n.op_type == "DequantizeLinear":
            return n
        if n.op_type in TRANSPARENT:
            return find_dq(n.input[0], depth + 1)
        return None

    new_nodes, converted = [], 0
    for n in g.node:
        if n.op_type == "Concat" and "_cc_ax" in n.name:
            zi = [j for j, x in enumerate(n.input) if "zeros" in x]
            di = [j for j, x in enumerate(n.input) if "zeros" not in x]
            if not zi or not di:
                continue
            dq = find_dq(n.input[di[0]])
            if dq is None:
                continue
            zname = n.input[zi[0]]
            scale_name, zp_name = dq.input[1], dq.input[2]
            zp = numpy_helper.to_array(inits[zp_name])
            shape = list(numpy_helper.to_array(inits[zname]).shape)
            q0 = numpy_helper.from_array(np.full(shape, int(zp), np.int8), zname + "_q")
            g.initializer.append(q0)
            dq_out = zname + "_dq_out"
            new_nodes.append((n, helper.make_node(
                "DequantizeLinear", [zname + "_q", scale_name, zp_name], [dq_out],
                name=zname + "_dq")))
            n.input[zi[0]] = dq_out
            converted += 1
    nodes = list(g.node)
    for consumer, dqn in new_nodes:
        idx = next(i for i, x in enumerate(nodes) if x is consumer)
        nodes.insert(idx, dqn)
    del g.node[:]
    g.node.extend(nodes)
    return converted


def fix_clip_dq(m: onnx.ModelProto) -> int:
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    byout = {o: n for n in g.node for o in n.output}
    new_inits, fixed = {}, 0
    for n in g.node:
        if n.op_type != "Clip":
            continue
        for j in (1, 2):
            if j >= len(n.input) or not n.input[j]:
                continue
            prod = byout.get(n.input[j])
            if prod is not None and prod.op_type == "DequantizeLinear":
                x = inits[prod.input[0]].astype(np.float32)
                sc = inits[prod.input[1]].astype(np.float32)
                zp = inits[prod.input[2]].astype(np.float32) if len(prod.input) > 2 else 0.0
                val = np.float32((x - zp) * sc)
                key = f"clip_const_{float(val)}"
                if key not in new_inits:
                    new_inits[key] = numpy_helper.from_array(np.array(val, np.float32), key)
                n.input[j] = key
                fixed += 1
    g.initializer.extend(new_inits.values())
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows-fp32", type=Path, required=True,
                    help="crop_study's rows-layout fp32 device export")
    ap.add_argument("--flat-int8", type=Path, required=True,
                    help="crop_study's shipped flat int8 (parity reference)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-depth", type=int, default=2)
    ap.add_argument("--cache", type=Path, default=HERE / "artifacts" / "vbd_cache.npz")
    ap.add_argument("--calib-clips", type=int, default=8)
    args = ap.parse_args()

    from dnsmos_trainable.export import quantize_qdq
    from dnsmos_trainable.verify import DNSMOS_DEVICE_OPS, check_device_constraints
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # infer crop geometry from the rows graph input
    rows_in = onnx.load(str(args.rows_fp32)).graph.input[0]
    _, n_rows, hop = [d.dim_value for d in rows_in.type.tensor_type.shape.dim]
    input_len = n_rows * hop
    print(f"rows layout [1,{n_rows},{hop}] ({input_len/16000:.2f} s)")

    pre = args.out.with_suffix(".pre.onnx")
    quant_pre_process(str(args.rows_fp32), str(pre), skip_symbolic_shape=True)
    convs = [n.name for n in onnx.load(str(pre)).graph.node if n.op_type == "Conv"]
    assert len(convs) == 14, len(convs)
    bwd = list(reversed(convs[7:]))
    W_VEC = np.array([0, 0, -1], np.float32)
    clips = np.load(args.cache)["clips"]
    calib = [{"wav": c[:input_len].astype(np.float32).reshape(1, n_rows, hop), "w": W_VEC}
             for c in clips[: args.calib_clips]]
    quantize_qdq(pre, args.out, calib, extra_exclude=bwd[: args.exclude_depth],
                 preprocessed=True, quantize_elementwise=True)
    pre.unlink(missing_ok=True)

    m = onnx.load(str(args.out))
    print(f"patches: slices={normalize_slices(m)} w={promote_w(m)} "
          f"pads={pads_to_concats(m)} pad_zeros_q={quantize_pad_zeros(m)} "
          f"clips={fix_clip_dq(m)}")
    onnx.save(m, str(args.out))

    problems = check_device_constraints(args.out, max_opset=13, allowed=DNSMOS_DEVICE_OPS)
    print("device lint:", "OK" if not problems else problems)

    # parity vs the shipped flat artifact (same grids by construction)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    a = ort.InferenceSession(str(args.flat_int8), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(args.out), so, providers=["CPUExecutionProvider"])
    worst = [0.0, 0.0, 0.0]
    cmin = 1.0
    for c in clips[8:12]:
        x = c[:input_len].astype(np.float32)
        ra = a.run(None, {"wav": x[None], "w": W_VEC})
        rb = b.run(None, {"wav": x.reshape(1, n_rows, hop), "w": W_VEC[None]})
        for i in (0, 1):
            worst[i] = max(worst[i], float(np.abs(ra[i] - rb[i]).max()))
        g1, g2 = ra[2].ravel(), rb[2].ravel()
        cmin = min(cmin, float(g1 @ g2 / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-12)))
    print(f"parity vs flat int8 (4 clips): max|d raw| {worst[0]:.2e} "
          f"max|d mos| {worst[1]:.2e}  grad cos min {cmin:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
