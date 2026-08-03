#!/usr/bin/env bash
# Assemble the attachment bundle for the ST support ticket described in
# ST_BUG_REPORT.md. Produces st_bug_report_<date>.tar.gz containing exactly the
# files §7 of the report lists, plus a host-reference JSON regenerated here so
# the numbers in the ticket are reproducible rather than transcribed.
#
# Usage:  ./package_st_report.sh [output_dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE}"
PY="${PY:-$HERE/../../.venv/bin/python}"
STAMP="$(date +%Y%m%d)"
STAGE="$(mktemp -d)"
BUNDLE="$STAGE/st_bug_report_$STAMP"
mkdir -p "$BUNDLE"

echo "[pkg] report + reproducer sources"
cp "$HERE/ST_BUG_REPORT.md" "$BUNDLE/"
mkdir -p "$BUNDLE/firmware_loss"
cp "$HERE/firmware_loss/orchestrator_loss.c" \
   "$HERE/firmware_loss/build_and_run.sh" \
   "$HERE/firmware_loss/gen_loss_input.py" \
   "$HERE/firmware_loss/README.md" "$BUNDLE/firmware_loss/"

echo "[pkg] deployable ONNX artifacts"
mkdir -p "$BUNDLE/models"
cp "$HERE/artifacts_crop/crop0p25s_int8_d2_rows_dev.onnx" "$BUNDLE/models/"
cp "$HERE/artifacts_crop/crop1s_int8_d2_rows_dev2.onnx"   "$BUNDLE/models/"

echo "[pkg] stedgeai generate outputs (text + OE graph + quant json only; no blobs)"
copy_gen() {                       # $1 = source gen dir, $2 = name in the bundle
  local src="$1" dst="$BUNDLE/$2"
  [ -d "$src" ] || { echo "  (skip $2 — not built)"; return; }
  mkdir -p "$dst"
  for f in network_generate_report.txt network.c network.h network_c_info.json; do
    [ -f "$src/$f" ] && cp "$src/$f" "$dst/"
  done
  cp "$src"/*_OE_*.onnx "$dst/" 2>/dev/null || true
  cp "$src"/*_Q.json    "$dst/" 2>/dev/null || true
}
copy_gen "$HERE/n6_gen/dnsmos_loss_crop0p25s_n6-allmems-O3" "gen_025_O3"
copy_gen "$HERE/n6_gen/dnsmos_loss_crop0p25s_O1"            "gen_025_O1"
copy_gen "$HERE/n6_gen/dnsmos_loss_crop0p25s_nocache"       "gen_025_nocacheopt"
copy_gen "$HERE/n6_gen/dnsmos_loss_crop0p25s_preserve"      "gen_025_preserve_inputs"
copy_gen "$HERE/n6_gen/dnsmos_loss_crop1s_int8_d2"          "gen_1s_hang"

echo "[pkg] UART capture"
for f in /tmp/loss_uart.txt /tmp/loss_uart2.txt; do
  [ -f "$f" ] && cat "$f" >> "$BUNDLE/uart_capture.txt"
done
[ -f "$BUNDLE/uart_capture.txt" ] || echo "(no UART capture found; re-run firmware_loss/build_and_run.sh)" \
  > "$BUNDLE/uart_capture.txt"

echo "[pkg] host reference (onnxruntime) for the two clips in the report"
"$PY" - "$BUNDLE" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np, onnxruntime as ort

bundle = Path(sys.argv[1])
here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
root = Path("/home/claroche/dnsmos_exported/examples/convfsenet_ondevice")
clips = np.load(root / "artifacts" / "vbd_cache.npz")["clips"]
w_flat = np.array([0.0, 0.0, -1.0], np.float32)
w_rows = w_flat[None]
so = ort.SessionOptions(); so.intra_op_num_threads = 1

dep = ort.InferenceSession(str(root / "artifacts_crop" / "crop0p25s_int8_d2_rows_dev.onnx"),
                           so, providers=["CPUExecutionProvider"])
oe = None
for cand in (root / "n6_gen" / "dnsmos_loss_crop0p25s_preserve").glob("*_OE_*.onnx"):
    oe = ort.InferenceSession(str(cand), so, providers=["CPUExecutionProvider"]); break

out = {"note": "expected values for the clips baked into firmware_loss/loss_input.h",
       "onnxruntime": ort.__version__, "clips": {}}
for ci in (12, 30):
    x = clips[ci][:4000].astype(np.float32).reshape(1, 25, 160)
    raw, mos, grad = dep.run(None, {"wav": x, "w": w_rows})
    entry = {
        "deployable_onnx": {
            "mos": [round(float(v), 4) for v in mos.ravel()],
            "grad_l2": float(np.linalg.norm(grad)),
            "grad_first8": [float(v) for v in grad.ravel()[:8]],
            "grad_nonfinite": int((~np.isfinite(grad)).sum()),
        }
    }
    if oe is not None:
        feeds = {i.name: (x.transpose(0, 2, 1).copy() if tuple(i.shape) == (1, 160, 25) else w_rows)
                 for i in oe.get_inputs()}
        res = {o.name: v for o, v in zip(oe.get_outputs(), oe.run(None, feeds))}
        mos_oe = next(v for v in res.values() if v.dtype == np.float32 and v.size == 3)
        grad_oe = next(v for v in res.values() if v.size == 4000)
        entry["st_optimized_export_in_ort"] = {
            "mos": [round(float(v), 4) for v in mos_oe.ravel()],
            "grad_l2": float(np.linalg.norm(grad_oe)),
        }
    out["clips"][f"vbd_cache_index_{ci}"] = entry

out["observed_on_target_both_clips"] = {
    "mos": [1.9094, 1.4481, 1.1435],
    "mos_bits": ["0x3ff465d9", "0x3fb959d2", "0x3f925cd8"],
    "comment": "bit-identical for both clips; gradient contains 0x808080f2 words and NaNs",
}
(bundle / "host_reference.json").write_text(json.dumps(out, indent=2))
print("  wrote host_reference.json")
PYEOF

echo "[pkg] environment"
strip_ansi() { sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g'; }
{
  echo "ST Edge AI Core:   $(~/stedgeai/install/4.0/Utilities/linux/stedgeai --version 2>/dev/null | head -1 | strip_ansi)"
  echo "CubeProgrammer:    $(~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI --version 2>/dev/null | strip_ansi | grep -i version | head -1 | tr -d '\r')"
  echo "arm-none-eabi-gcc: $(~/toolchains/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc -dumpversion 2>/dev/null)"
  echo "onnxruntime:       $("$PY" -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null)"
  echo "board:             STM32N6570-DK, device 0x486 rev B, MCU 800 MHz / NPU 1 GHz"
  echo "host OS:           $(uname -s) $(uname -r) $(uname -m)"
} > "$BUNDLE/environment.txt"

TAR="$OUT/st_bug_report_$STAMP.tar.gz"
tar -czf "$TAR" -C "$STAGE" "st_bug_report_$STAMP"
rm -rf "$STAGE"
echo
echo "[pkg] wrote $TAR  ($(du -h "$TAR" | cut -f1))"
tar -tzf "$TAR" | sed 's/^/       /' | head -40
