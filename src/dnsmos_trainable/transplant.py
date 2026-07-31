"""Exact weight extraction from the official sig_bak_ovr.onnx into a state dict.

Nodes are mapped by *ordered role* (kernel shapes and graph order), not by
name — keras2onnx node names are not stable across exports. The fc3 bias is
fingerprinted against known values so a silently changed upstream file fails
loudly instead of producing a subtly wrong model.
"""

import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from dnsmos_trainable.constants import (
    FC3_BIAS_FINGERPRINT,
    N_BINS,
    OFFICIAL_SHA256,
    WIN,
)


def load_official(path: str | Path, check_hash: bool = True) -> onnx.ModelProto:
    path = Path(path)
    if check_hash:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != OFFICIAL_SHA256:
            raise RuntimeError(
                f"sha256 mismatch for {path}: got {digest}, expected {OFFICIAL_SHA256}"
            )
    return onnx.load(str(path))


def extract_params(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Walk the graph in order; classify Conv/MatMul(+Add) nodes by role."""
    graph = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    stft = []  # (name, weight, bias) for the two [161, 320, 1] convs
    convs = []  # (weight, bias) for the seven 3x3 convs
    denses = []  # (weight, bias) for the three MatMul+Add pairs
    eps = None

    nodes = list(graph.node)
    for i, node in enumerate(nodes):
        if node.op_type == "Conv":
            w = inits[node.input[1]]
            b = inits[node.input[2]] if len(node.input) > 2 else None
            if w.shape == (N_BINS, WIN, 1):
                stft.append((node.name, w.squeeze(-1), b))
            elif w.ndim == 4 and w.shape[2:] == (3, 3):
                convs.append((w, b))
            else:
                raise RuntimeError(f"unexpected Conv kernel shape {w.shape} at {node.name}")
        elif node.op_type == "MatMul":
            w = inits[node.input[1]]  # stored [in, out]
            nxt = nodes[i + 1]
            if nxt.op_type != "Add" or nxt.input[0] != node.output[0]:
                raise RuntimeError(f"MatMul {node.name} not followed by bias Add")
            b = inits[nxt.input[1]]
            denses.append((w, b))
        elif node.op_type == "Max":
            scalars = [inits[x] for x in node.input if x in inits and inits[x].ndim == 0]
            if scalars:
                eps = float(scalars[0])

    if len(stft) != 2 or len(convs) != 7 or len(denses) != 3:
        raise RuntimeError(
            f"unexpected graph structure: {len(stft)} stft convs, "
            f"{len(convs)} 3x3 convs, {len(denses)} dense layers"
        )
    if eps is None or not np.isclose(eps, 1e-12):
        raise RuntimeError(f"unexpected power-floor eps: {eps}")

    def stft_by_role(role: str) -> tuple[np.ndarray, np.ndarray | None]:
        named = [s for s in stft if role in s[0].lower()]
        if len(named) == 1:
            return named[0][1], named[0][2]
        # Fallback: graph order is real first, imag second.
        idx = 0 if role == "real" else 1
        return stft[idx][1], stft[idx][2]

    w_re, b_re = stft_by_role("real")
    w_im, b_im = stft_by_role("imag")
    if b_re is not None or b_im is not None:
        raise RuntimeError("official stft convs are expected to be bias-free")

    params: dict[str, np.ndarray] = {"stft.w_re": w_re, "stft.w_im": w_im}
    for k, (w, b) in enumerate(convs, start=1):
        params[f"conv{k}.weight"] = w
        params[f"conv{k}.bias"] = b
    for k, (w, b) in enumerate(denses, start=1):
        params[f"fc{k}.weight"] = w.T  # ONNX MatMul stores [in, out]; nn.Linear wants [out, in]
        params[f"fc{k}.bias"] = b

    fingerprint = np.array(FC3_BIAS_FINGERPRINT, dtype=np.float32)
    if not np.allclose(params["fc3.bias"], fingerprint, atol=1e-4):
        raise RuntimeError(
            f"fc3 bias fingerprint mismatch: got {params['fc3.bias']}, "
            f"expected ~{fingerprint}. Wrong or changed upstream model."
        )
    return params


def build_state_dict(params: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    sd["frontend.stft.w_re"] = torch.from_numpy(params["stft.w_re"].copy())
    sd["frontend.stft.w_im"] = torch.from_numpy(params["stft.w_im"].copy())
    for k in range(1, 8):
        sd[f"body.conv{k}.weight"] = torch.from_numpy(params[f"conv{k}.weight"].copy())
        sd[f"body.conv{k}.bias"] = torch.from_numpy(params[f"conv{k}.bias"].copy())
    for k in range(1, 4):
        sd[f"body.fc{k}.weight"] = torch.from_numpy(np.ascontiguousarray(params[f"fc{k}.weight"]))
        sd[f"body.fc{k}.bias"] = torch.from_numpy(params[f"fc{k}.bias"].copy())
    return sd


def transplant(official_path: str | Path) -> dict[str, torch.Tensor]:
    return build_state_dict(extract_params(load_official(official_path)))
