#!/usr/bin/env python3
"""Run the PESQ predictor on the STM32N6 and compare against the host int8 ONNX.

The board must already be flashed with a `pesq_*` build:

    cd $STEDGEAI/scripts/N6_scripts
    python n6_loader.py --config <cfg>.json \\
        -nf .../n6_gen/pesq_sig_n6-noextmem/network.c -bc N6-DK

The comparison that matters is **device vs host int8**, not device vs fp32:
int8 is the artifact that was deployed, so any difference against it is the
device failing to reproduce its own network. Device-vs-true-PESQ is reported
too, but that gap is the model's error, not the port's.

Two things this checks that a single number cannot:

* **The output must vary with the input.** The DNSMOS loss graph on this same
  silicon returns a fixed constant regardless of what it is fed (see
  ST_BUG_REPORT.md), which looks fine until you feed it a second clip. Feeding
  candidates that span the PESQ range makes that failure mode visible.
* **I/O layout is read from the device, not assumed.** The compiler may
  transpose input layouts per graph, so shapes come from `get_info()`.

    python run_pesq_on_target.py --candidates 3 17 42 88 123 260
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEDGEAI = Path("/home/claroche/stedgeai/install/4.0")
sys.path.insert(0, str(STEDGEAI / "scripts" / "ai_runner"))

DESC = "serial:921600"          # ST-LINK VCP; the runner probes for the port


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path,
                    default=HERE / "artifacts" / "pesq_predictor_int8.onnx",
                    help="the host reference — must be the graph that was compiled")
    ap.add_argument("--data", type=Path, default=HERE / "artifacts" / "pesq_dataset.npz")
    ap.add_argument("--candidates", type=int, nargs="+",
                    default=[3, 17, 42, 88, 123, 260])
    ap.add_argument("--frames", type=int, default=63)
    ap.add_argument("--desc", default=DESC)
    args = ap.parse_args()

    from stm_ai_runner import AiRunner
    import onnxruntime as ort

    z = np.load(args.data)
    feats, labels = z["feats"], z["labels"] * 3.5 + 1.0        # stored normalized
    T = args.frames

    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(args.onnx), so, providers=["CPUExecutionProvider"])

    runner = AiRunner()
    if not runner.connect(args.desc):
        raise SystemExit(f"could not connect to the board via {args.desc!r} — "
                         "check the ST-LINK is attached and the build is flashed")
    info = runner.get_info()
    print(f"device: {info['name']}  inputs {[i['shape'] for i in info['inputs']]}  "
          f"outputs {[o['shape'] for o in info['outputs']]}\n")

    print(f"{'cand':>5} {'true':>6} {'host int8':>10} {'device':>8} {'|d|':>7} {'ms':>7}")
    diffs, times, devs = [], [], []
    for c in args.candidates:
        nm = feats[c, 0, :, :T][None].astype(np.float32)
        em = feats[c, 1, :, :T][None].astype(np.float32)
        host = float(sess.run(None, {"noisy_mag": nm, "enh_mag": em})[0].ravel()[0])

        outs, prof = runner.invoke([nm[..., None], em[..., None]])
        dev = float(np.asarray(outs[0]).ravel()[0])
        ms = float(prof["c_durations"][0]) if prof.get("c_durations") else float("nan")

        hp, dp = host * 3.5 + 1.0, dev * 3.5 + 1.0
        diffs.append(abs(hp - dp)); times.append(ms); devs.append(dp)
        print(f"{c:>5} {labels[c]:6.2f} {hp:10.2f} {dp:8.2f} {diffs[-1]:7.3f} {ms:7.1f}")

    runner.disconnect()
    spread = max(devs) - min(devs)
    print(f"\ndevice vs host int8: mean {np.mean(diffs):.4f} PESQ, max {np.max(diffs):.4f}")
    print(f"latency: {np.mean(times):.1f} ms/inference")
    print(f"device output spread {min(devs):.2f}..{max(devs):.2f} ({spread:.2f}) — "
          f"{'input-dependent' if spread > 0.5 else 'SUSPECT: near-constant'}")


if __name__ == "__main__":
    main()
