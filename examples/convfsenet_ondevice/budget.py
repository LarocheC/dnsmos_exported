#!/usr/bin/env python3
"""STM32N6 memory / compute budget for the on-device training loop.

Numbers are read out of the actual exported artifacts (weight bytes, tensor
shapes, MAC counts from conv/matmul attributes), not estimated, so the table
moves when the graphs move. Pool sizes come from ST Edge AI's shipped profiles
as used by eco8-neaixt:

    n6-noextmem       ~2.8 MB usable (npuRAM3-6 ~1.8 MB + cpuRAM2 1 MB)
    n6-allmems-O3     + 32 MB hexa-SPI PSRAM / 64 MB octoFlash (N6570-DK only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dsp  # noqa: E402

MB = 1024 * 1024
NOEXTMEM_POOL = 2.8 * MB


def weight_bytes(model_path: Path) -> int:
    m = onnx.load(str(model_path))
    return sum(numpy_helper.to_array(t).nbytes for t in m.graph.initializer)


def macs(model_path: Path) -> int:
    """MAC count for Conv / Gemm / MatMul nodes, from inferred shapes.

    Resolves weight shapes through DequantizeLinear so QDQ int8 graphs are
    measured, not silently counted as zero.
    """
    m = onnx.shape_inference.infer_shapes(onnx.load(str(model_path)))
    shapes = {}
    for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    inits = {t.name: tuple(t.dims) for t in m.graph.initializer}
    producers = {out: n for n in m.graph.node for out in n.output}

    def tensor_shape(name, hops=4):
        """Shape of `name`, walking back through Q/DQ and reshaping ops."""
        for _ in range(hops):
            if name in inits:
                return list(inits[name])
            if name in shapes:
                return shapes[name]
            prod = producers.get(name)
            if prod is None or prod.op_type not in (
                    "DequantizeLinear", "QuantizeLinear", "Transpose", "Identity", "Cast"):
                return None
            name = prod.input[0]
        return None

    total = 0
    for node in m.graph.node:
        if node.op_type == "Conv":
            w = tensor_shape(node.input[1])
            out = shapes.get(node.output[0])
            if w and out:
                cin_per_group, k = w[1], int(np.prod(w[2:])) if len(w) > 2 else 1
                total += int(np.prod(out)) * cin_per_group * k
        elif node.op_type in ("MatMul", "Gemm"):
            a = tensor_shape(node.input[0])
            out = shapes.get(node.output[0])
            if a and out:
                total += int(np.prod(out)) * a[-1]
    return total


def peak_activation_bytes(model_path: Path, dtype_bytes: int) -> int:
    """Crude high-water mark: the largest single intermediate tensor."""
    m = onnx.shape_inference.infer_shapes(onnx.load(str(model_path)))
    peak = 0
    for vi in list(m.graph.value_info) + list(m.graph.output):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        peak = max(peak, int(np.prod(dims)) * dtype_bytes)
    return peak


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    p.add_argument("--dnsmos-dir", type=Path, default=HERE.parents[1] / "artifacts")
    p.add_argument("--n-features", type=int, default=256)
    args = p.parse_args()

    A, D = args.artifacts, args.dnsmos_dir
    seg = 144160
    T = dsp.num_frames(seg)
    F, C = args.n_features, 192

    trunk_i8 = A / "convfsenet_trunk_int8.onnx"
    head_f = A / "convfsenet_head_fwd.onnx"
    head_b = A / "convfsenet_head_bwd.onnx"
    loss_i8 = D / "dnsmos_loss_int8_qdq_stm32n6.onnx"
    loss_f32 = D / "dnsmos_loss_fp32_stm32n6.onnx"

    print(f"Adaptation window: {seg} samples = {seg/16000:.2f} s = {T} STFT frames\n")

    print("== Persistent state (must stay resident across the whole loop) ==")
    head_params = F * C + F
    dnsmos_path = loss_i8 if loss_i8.exists() else loss_f32
    rows = []
    if trunk_i8.exists():
        rows.append(("ConvFSENet trunk weights (int8, frozen)", weight_bytes(trunk_i8), False))
    if dnsmos_path.exists():
        tag = "int8" if dnsmos_path is loss_i8 else "fp32"
        rows.append((f"DNSMOS loss graph weights ({tag})", weight_bytes(dnsmos_path), True))
    rows.append((f"Head W,b (fp32, trainable, {head_params:,} params)", head_params * 4, True))
    rows.append(("Adam moments m,v (fp32)", head_params * 4 * 2, True))
    for label, b, _ in rows:
        print(f"  {label:<48s} {b/1024:9.1f} KiB")
    persistent = sum(b for _, b, _ in rows)
    training_only = sum(b for _, b, t in rows if t)
    print(f"  {'TOTAL persistent':<48s} {persistent/MB:9.2f} MB")
    print(f"  {'  ...of which is NEW for training':<48s} {training_only/MB:9.2f} MB")

    print("\n== Per-window working buffers (host / M55) ==")
    work = [
        ("noisy waveform", seg * 4),
        ("STFT X (257 x T, complex fp32)", 257 * T * 8),
        ("|X| magnitude (F x T)", F * T * 4),
        ("trunk output h (192 x T)", C * T * 4),
        ("mask (F x T)", F * T * 4),
        ("enhanced waveform", seg * 4),
        ("dL/d(enhanced)", seg * 4),
        ("dL/dmask (F x T)", F * T * 4),
        ("dW, db", (F * C + F) * 4),
    ]
    for label, b in work:
        print(f"  {label:<48s} {b/1024:9.1f} KiB")
    working = sum(b for _, b in work)
    print(f"  {'TOTAL working':<48s} {working/MB:9.2f} MB")

    print("\n== Transient activation peaks (one session at a time) ==")
    peaks = []
    if trunk_i8.exists():
        peaks.append((f"trunk int8 (emit_T window)", peak_activation_bytes(trunk_i8, 1)))
    if head_f.exists():
        peaks.append(("head forward (fp32)", peak_activation_bytes(head_f, 4)))
    if head_b.exists():
        peaks.append(("head backward (fp32)", peak_activation_bytes(head_b, 4)))
    if loss_i8.exists():
        peaks.append(("DNSMOS loss graph int8", peak_activation_bytes(loss_i8, 1)))
    for label, b in peaks:
        print(f"  {label:<48s} {b/1024:9.1f} KiB")
    act_peak = max((b for _, b in peaks), default=0)

    print("\n== Compute per adaptation window ==")
    n_calls = 0
    if trunk_i8.exists():
        per_call = macs(trunk_i8)
        emit = int(onnx.load(str(trunk_i8)).graph.output[0].type.tensor_type.shape.dim[2].dim_value)
        n_calls = int(np.ceil(T / emit))
        print(f"  trunk  : {per_call/1e6:8.2f} MMAC/call x {n_calls} calls = "
              f"{per_call*n_calls/1e6:8.1f} MMAC   (NPU, int8)")
    if head_f.exists():
        print(f"  head fw: {macs(head_f)/1e6:8.2f} MMAC                        "
              f"      (M55 SW: runtime weights)")
    if head_b.exists():
        print(f"  head bw: {macs(head_b)/1e6:8.2f} MMAC                        "
              f"      (M55 SW: both operands dynamic)")
    if loss_i8.exists():
        print(f"  DNSMOS : {macs(loss_i8)/1e6:8.2f} MMAC  (fwd+bwd fused)      (NPU, int8)")
    print(f"  STFT/ISTFT/adjoint: 3 x {T} x 512-pt FFT              (M55 CMSIS-DSP)")

    # Anchor to eco8-neaixt's measured on-target throughput rather than a
    # datasheet peak: ConvFSENet int8 ran 1.47 MMAC/frame in 4.40 ms
    # (n6-noextmem) and LiSenNet windowed 177.7 MMAC in 73.63 ms — i.e. a
    # measured 0.33-2.41 GMAC/s depending on how epoch-bound the graph is.
    lo, hi = 0.33e9, 2.41e9
    total_macs = 0
    if trunk_i8.exists():
        total_macs += macs(trunk_i8) * n_calls
    for path in (head_f, head_b):
        if path.exists():
            total_macs += macs(path)
    if dnsmos_path.exists():
        total_macs += macs(dnsmos_path)
    print(f"\n  total {total_macs/1e6:,.0f} MMAC per adaptation window")
    print(f"  extrapolated wall-clock at eco8's measured 0.33-2.41 GMAC/s: "
          f"{total_macs/hi:.1f}-{total_macs/lo:.1f} s")
    print(f"  (one update per {seg/16000:.2f} s of audio; adaptation is a background")
    print("   task, not part of the real-time enhancement path. Only stedgeai")
    print("   validate --mode target / npu_profiler gives real numbers.)")

    dnsmos_peak = next((b for lbl, b in peaks if lbl.startswith("DNSMOS")), 0)
    enh_peak = max((b for lbl, b in peaks if not lbl.startswith("DNSMOS")), default=0)
    enh_total = persistent - (weight_bytes(dnsmos_path) if dnsmos_path.exists() else 0) \
        + working + enh_peak
    total = persistent + working + act_peak

    print("\n== Fit ==")
    print(f"  enhancer side only (trunk + head + training buffers) = {enh_total/MB:5.2f} MB")
    print(f"  + DNSMOS loss graph activation peak                  = {dnsmos_peak/MB:5.2f} MB")
    print(f"  TOTAL                                                = {total/MB:5.2f} MB")
    print(f"  n6-noextmem usable pools                             ~{NOEXTMEM_POOL/MB:5.2f} MB")
    print()
    if enh_total < NOEXTMEM_POOL:
        print("  -> The ConvFSENet half (inference + on-device training of the mask")
        print("     head) FITS in internal SRAM. The DNSMOS teacher is what forces")
        print("     external memory: its fused fwd+bwd graph peaks at "
              f"{dnsmos_peak/MB:.1f} MB of")
        print("     activations because it scores a whole 9.01 s segment at once.")
    print("  -> As built, the loop needs the N6570-DK's 32 MB hexa-SPI PSRAM")
    print("     (profile n6-allmems-O3). To reach an internal-SRAM-only build,")
    print("     distill the compact DNSMOS student (configs/student_small.py in")
    print("     this repo) and re-export its loss graph.")


if __name__ == "__main__":
    main()
