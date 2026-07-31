#!/usr/bin/env python3
"""Export all fp32 artifacts: forward + loss graphs, desktop + STM32N6 variants."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dnsmos_trainable import load_transplanted
from dnsmos_trainable.backward import DnsmosLossGraph
from dnsmos_trainable.export import export_forward, export_loss_graph

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    model = load_transplanted(root / "models" / "dnsmos_transplanted.pt")
    art = root / "artifacts"

    print(export_forward(model, art / "dnsmos_fwd_fp32.onnx", device=False))
    print(export_forward(model, art / "dnsmos_fwd_fp32_stm32n6.onnx", device=True))
    print(export_loss_graph(DnsmosLossGraph(model, mode="ort", io_layout="flat"), art / "dnsmos_loss_fp32.onnx"))
    print(export_loss_graph(DnsmosLossGraph(model, mode="device", io_layout="rows"), art / "dnsmos_loss_fp32_stm32n6.onnx"))
