"""Deterministic QDQ insertion using the QAT sim's exact quantization grids.

`quantize_static` recalibrates activation ranges from data, so a QAT-trained
model gets deployed on a *different* grid than the one it was trained against
— consistently losing accuracy relative to the sim. This writer instead
stamps the sim's own scales/zero-points into the fp32 graph:

- activations: per-tensor asymmetric int8 Q/DQ after each ReLU (and matching
  Q/DQ after MaxPool/ReduceMax so downstream int8 kernels fuse — pools are
  quantization-transparent, so they reuse their input's parameters), before
  conv1 (the log-power feature) and after fc3 (raw scores);
- weights: per-channel symmetric int8 ([-127, 127], axis 0) initializers with
  DequantizeLinear;
- biases: int32 initializers with per-channel scale = act_scale * w_scale
  (the pattern ST Edge AI expects).

The result follows the ss/sa scheme required by ST Neural-ART and matches the
`QatDnsmosModel` fake-quant semantics exactly, so the deployed artifact
inherits the sim's measured deltas.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from dnsmos_trainable.qat import ACT_SITES, _act_qparams


def _weight_qparams(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = w.reshape(w.shape[0], -1)
    scales = np.maximum(np.abs(flat).max(axis=1) / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.round(w / scales.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127)
    return q.astype(np.int8), scales


class _QdqBuilder:
    def __init__(self, graph: onnx.GraphProto) -> None:
        self.graph = graph
        self.counter = 0
        self.inserts: dict[int, list] = {}  # node index -> nodes to insert after
        self.by_output = {out: i for i, n in enumerate(graph.node) for out in n.output}

    def fresh(self, base: str) -> str:
        self.counter += 1
        return f"{base}_qdq{self.counter}"

    def add_init(self, name: str, arr: np.ndarray) -> str:
        t = numpy_helper.from_array(arr, name)
        self.graph.initializer.append(t)
        return name

    def qdq_activation(self, tensor: str, scale: float, zp: int) -> None:
        """Insert tensor -> Q -> DQ and rewire all consumers to the DQ output.

        If `tensor` is a graph output, the producer is renamed and the DQ
        output takes the original name, so the graph output carries the
        quantized value (matching the sim, which returns the fake-quantized
        tensor) instead of silently bypassing the Q/DQ pair.
        """
        graph_out = {o.name for o in self.graph.output}
        src = tensor
        if tensor in graph_out:
            src = f"{tensor}_prequant"
            producer = self.graph.node[self.by_output[tensor]]
            for j, out in enumerate(producer.output):
                if out == tensor:
                    producer.output[j] = src
            self.by_output[src] = self.by_output.pop(tensor)
        s = self.add_init(self.fresh(f"{tensor}_scale"), np.float32(scale))
        z = self.add_init(self.fresh(f"{tensor}_zp"), np.int8(zp))
        qname = self.fresh(f"{tensor}_q")
        dqname = tensor if tensor in graph_out else self.fresh(f"{tensor}_dq")
        qnode = helper.make_node("QuantizeLinear", [src, s, z], [qname], name=qname)
        dqnode = helper.make_node(
            "DequantizeLinear", [qname, s, z], [dqname], name=self.fresh(f"{tensor}_dq")
        )
        for node in self.graph.node:
            if node.op_type in ("QuantizeLinear",):
                continue
            for j, inp in enumerate(node.input):
                if inp == src and node is not qnode:
                    node.input[j] = dqname
        idx = self.by_output[src]
        self.inserts.setdefault(idx, []).extend([qnode, dqnode])
        if dqname != tensor:
            self.by_output[dqname] = idx

    def dq_weight(self, node: onnx.NodeProto, w_name: str, w: np.ndarray) -> np.ndarray:
        q, scales = _weight_qparams(w)
        self.graph.initializer.remove(
            next(t for t in self.graph.initializer if t.name == w_name)
        )
        qn = self.add_init(w_name + "_int8", q)
        sn = self.add_init(w_name + "_wscale", scales)
        zn = self.add_init(w_name + "_wzp", np.zeros(len(scales), dtype=np.int8))
        dqname = self.fresh(w_name + "_dq")
        dq = helper.make_node(
            "DequantizeLinear", [qn, sn, zn], [dqname], name=dqname, axis=0
        )
        idx = min(self.by_output.get(i, 10**9) for i in node.input if i in self.by_output) \
            if any(i in self.by_output for i in node.input) else 0
        self.inserts.setdefault(-1, []).append(dq)  # -1: prepend at graph start
        for j, inp in enumerate(node.input):
            if inp == w_name:
                node.input[j] = dqname
        return scales

    def dq_bias(self, node: onnx.NodeProto, b_name: str, b: np.ndarray,
                act_scale: float, w_scales: np.ndarray) -> None:
        scales = (act_scale * w_scales).astype(np.float32)
        q = np.round(b / scales).astype(np.int32)
        self.graph.initializer.remove(
            next(t for t in self.graph.initializer if t.name == b_name)
        )
        qn = self.add_init(b_name + "_int32", q)
        sn = self.add_init(b_name + "_bscale", scales)
        zn = self.add_init(b_name + "_bzp", np.zeros(len(scales), dtype=np.int32))
        dqname = self.fresh(b_name + "_dq")
        dq = helper.make_node("DequantizeLinear", [qn, sn, zn], [dqname], name=dqname, axis=0)
        self.inserts.setdefault(-1, []).append(dq)
        for j, inp in enumerate(node.input):
            if inp == b_name:
                node.input[j] = dqname

    def finalize(self) -> None:
        new_nodes = list(self.inserts.get(-1, []))
        for i, node in enumerate(self.graph.node):
            new_nodes.append(node)
            new_nodes.extend(self.inserts.get(i, []))
        del self.graph.node[:]
        self.graph.node.extend(new_nodes)


def write_qdq_from_sim(
    fp32_path: str | Path,
    out_path: str | Path,
    act_ranges: dict[str, tuple[float, float]],
    float_tail: bool = False,
) -> Path:
    """Insert QDQ into an exported forward graph using the sim's grids.

    Works on both the desktop (dynamo) and device (legacy exporter,
    onnxsim-simplified) forward artifacts: nodes are matched structurally
    (conv order, relu consumers), never by name.
    """
    fp32_path, out_path = Path(fp32_path), Path(out_path)
    model = onnx.load(str(fp32_path))
    graph = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    convs = [n for n in graph.node if n.op_type == "Conv"
             and inits[n.input[1]].ndim == 4 and inits[n.input[1]].shape[2:] == (3, 3)]
    gemms = [n for n in graph.node if n.op_type == "Gemm"]
    if len(convs) != 7 or len(gemms) != 3:
        raise RuntimeError(f"unexpected structure: {len(convs)} convs, {len(gemms)} gemms")

    def relu_after(node):
        (relu,) = [c for c in consumers[node.output[0]] if c.op_type == "Relu"]
        return relu

    qp = {site: _act_qparams(*act_ranges[site]) for site in ACT_SITES}
    b = _QdqBuilder(graph)

    # Activation sites. Pools/ReduceMax reuse their input's params.
    b.qdq_activation(convs[0].input[0], *qp["feat"])
    for k, conv in enumerate(convs, start=1):
        if float_tail and k == 7:
            continue  # a7 and everything after stays float
        b.qdq_activation(relu_after(conv).output[0], *qp[f"a{k}"])
    for pool in [n for n in graph.node if n.op_type == "MaxPool"]:
        src_relu_site = None
        # pool input is (rewired) relu output; find which a-site it came from
        for k, conv in enumerate(convs, start=1):
            if pool.input[0].startswith(relu_after(conv).output[0]):
                src_relu_site = f"a{k}"
        if src_relu_site is None:
            raise RuntimeError(f"cannot map pool input {pool.input[0]} to a site")
        b.qdq_activation(pool.output[0], *qp[src_relu_site])
    if not float_tail:
        (rmax,) = [n for n in graph.node if n.op_type == "ReduceMax"]
        b.qdq_activation(rmax.output[0], *qp["a7"])
        b.qdq_activation(relu_after(gemms[0]).output[0], *qp["b1"])
        b.qdq_activation(relu_after(gemms[1]).output[0], *qp["b2"])
        b.qdq_activation(gemms[2].output[0], *qp["raw"])

    # Weights and biases. Input-activation scale per layer for bias scaling.
    in_site = {0: "feat", 1: "a1", 2: "a2", 3: "a3", 4: "a4", 5: "a5", 6: "a6"}
    for k, conv in enumerate(convs):
        w_name = conv.input[1]
        w_scales = b.dq_weight(conv, w_name, inits[w_name])
        if len(conv.input) > 2:
            b.dq_bias(conv, conv.input[2], inits[conv.input[2]], qp[in_site[k]][0], w_scales)
    if not float_tail:
        gemm_in = {0: "a7", 1: "b1", 2: "b2"}
        for k, gemm in enumerate(gemms):
            w_name = gemm.input[1]
            w_scales = b.dq_weight(gemm, w_name, inits[w_name])
            if len(gemm.input) > 2:
                b.dq_bias(gemm, gemm.input[2], inits[gemm.input[2]], qp[gemm_in[k]][0], w_scales)

    b.finalize()
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    return out_path
