#!/usr/bin/env python3
"""How small must DNSMOS and ConvFSENet be to train on-chip in real time?

Answers two coupled questions for the STM32N6:

1. How short can the DNSMOS adaptation window get, and how narrow the model,
   before the fused forward+backward loss graph fits the device?
2. How small must ConvFSENet be to leave enough NPU time for that, while the
   enhancement path itself stays real-time?

The MAC model is calibrated against the exported artifacts: for the official
geometry it reproduces `budget.macs` on the real ONNX graph exactly (ratio
1.000), and the fused loss graph measures 2.00x the forward graph.

Latency model, for width-scaling the SAME topology, uses eco8-neaixt's measured
on-target split of ConvFSENet int8 at n6-noextmem: 4.40 ms/frame total of which
1.26 ms is NPU core, leaving ~3.14 ms of M55 epoch/state-plumbing overhead that
does NOT shrink with width ("ConvFSENet is at its N6 floor"). So narrowing the
enhancer buys much less than its MAC count suggests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from dnsmos_trainable.constants import SR, DnsmosConfig, OFFICIAL_CONFIG  # noqa: E402

MB = 1024 * 1024
NOEXTMEM = 2.8 * MB
PSRAM = 32 * MB

FRAME_MS = 16.0                 # hop 256 @ 16 kHz
ENH_EPOCH_FLOOR_MS = 3.14       # M55 state-plumbing floor, measured by eco8
ENH_NPU_MS_AT_REF = 1.26        # NPU core time at the reference MAC count
ENH_REF_MACS = 1_435_776        # ConvFSENet 192/384, 9 blocks, 256 bins
GMAC_LO, GMAC_HI = 0.33e9, 2.41e9   # eco8's measured on-target throughput span


# ---------------------------------------------------------------- DNSMOS side

def dnsmos_forward_macs(cfg: DnsmosConfig) -> int:
    T, F = cfg.n_frames, cfg.n_bins
    c = cfg.channels
    total = T * cfg.win * F * 2                     # frontend re/im projections
    prev, t, f = 1, T, F
    for k in range(7):
        if k in (4, 5, 6):                          # pools precede conv5/6/7
            t, f = t // 2, f // 2
        total += t * f * c[k] * prev * 9
        prev = c[k]
    f1, f2 = cfg.fc
    return total + c[6] * f1 + f1 * f2 + f2 * 3


def dnsmos_loss_macs(cfg: DnsmosConfig) -> int:
    """Fused fwd+bwd: forward recompute + the VJP chain. Measured 2.00x."""
    return 2 * dnsmos_forward_macs(cfg)


def dnsmos_params(cfg: DnsmosConfig) -> int:
    c = cfg.channels
    n = 2 * cfg.n_bins * cfg.win                    # stft projections
    prev = 1
    for k in range(7):
        n += c[k] * prev * 9 + c[k]
        prev = c[k]
    f1, f2 = cfg.fc
    return n + c[6] * f1 + f1 + f1 * f2 + f2 + f2 * 3 + 3


def dnsmos_peak_bytes(cfg: DnsmosConfig, elementwise_int8: bool = False) -> int:
    """conv1's output, the largest live tensor.

    The elementwise mask/multiply ops of the hand-written backward are NOT in
    `op_types_to_quantize` (only Conv/Gemm/MatMul are), so today they carry
    that tensor in fp32 and it is genuinely materialized — which is what the
    70.75 MB measurement on the shipped artifact reflects. Quantizing the
    elementwise ops too would cut it 4x; that is the `elementwise_int8` column.
    """
    return cfg.peak_activation_elems * (1 if elementwise_int8 else 4)


# ------------------------------------------------------------ ConvFSENet side

def convfsenet_macs_per_frame(bins=256, c_res=192, c_conv=384, n_blocks=9, k=3) -> int:
    return (bins * c_res
            + n_blocks * (c_res * c_conv + c_conv * k + c_conv * c_res)
            + c_res * bins)


def enhancer_ms_per_frame(macs: int) -> float:
    """Same topology, different widths: epoch overhead is fixed, NPU core scales."""
    return ENH_EPOCH_FLOOR_MS + ENH_NPU_MS_AT_REF * macs / ENH_REF_MACS


# ------------------------------------------------------------------- analysis

def adaptation_budget_gmac(window_s: float, enh_ms: float, duty: float = 1.0):
    """NPU-seconds left over per window after real-time enhancement."""
    frames = window_s * SR / 256          # ConvFSENet hop
    left_s = window_s * duty - frames * enh_ms / 1000.0
    if left_s <= 0:
        return 0.0, 0.0, left_s
    return left_s * GMAC_LO / 1e9, left_s * GMAC_HI / 1e9, left_s


def scaled(channels, factor):
    return tuple(max(4, int(round(c * factor))) for c in channels)


def frames_to_len(n_frames, win=320, hop=160):
    return win + (n_frames - 1) * hop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duty", type=float, default=1.0,
                    help="fraction of wall-clock the NPU may spend (1.0 = fully busy)")
    ap.add_argument("--enh-bins", type=int, default=256)
    args = ap.parse_args()

    print("=" * 78)
    print("1. ENHANCER: how small does ConvFSENet need to be?")
    print("=" * 78)
    print(f"frame period {FRAME_MS:.0f} ms; latency = {ENH_EPOCH_FLOOR_MS:.2f} ms epoch floor "
          f"+ NPU core scaled by MACs\n")
    print(f"{'config (res/conv x blocks)':<30}{'MMAC/frame':>12}{'ms/frame':>10}{'RTF':>8}"
          f"{'NPU busy':>10}")
    enh_options = [
        ("192/384 x9  (deployed)", 192, 384, 9),
        ("128/256 x9  (published)", 128, 256, 9),
        ("96/192  x9", 96, 192, 9),
        ("64/128  x9", 64, 128, 9),
        ("64/128  x6", 64, 128, 6),
        ("48/96   x6", 48, 96, 6),
        ("32/64   x6", 32, 64, 6),
    ]
    enh_rows = []
    for label, cres, cconv, nb in enh_options:
        m = convfsenet_macs_per_frame(args.enh_bins, cres, cconv, nb)
        ms = enhancer_ms_per_frame(m)
        rtf = ms / FRAME_MS
        enh_rows.append((label, m, ms, rtf))
        print(f"{label:<30}{m/1e6:>12.2f}{ms:>10.2f}{rtf:>8.3f}{rtf*100:>9.1f}%")
    print("\nAll are real-time. Narrowing helps little: the epoch/state-plumbing")
    print(f"floor is {ENH_EPOCH_FLOOR_MS:.2f} ms/frame ({ENH_EPOCH_FLOOR_MS/FRAME_MS*100:.0f}% of the frame) "
          "and does not scale with width.")

    print()
    print("=" * 78)
    print("2. ADAPTATION BUDGET left over per window, after real-time enhancement")
    print("=" * 78)
    print(f"{'enhancer':<30}{'window':>9}{'NPU left':>10}{'DNSMOS budget (GMAC)':>24}")
    budgets = {}
    for label, m, ms, _ in enh_rows[:1] + enh_rows[3:5]:
        for win_s in (9.01, 4.0, 2.0, 1.0):
            lo, hi, left = adaptation_budget_gmac(win_s, ms, args.duty)
            budgets[(label, win_s)] = (lo, hi)
            print(f"{label:<30}{win_s:>8.2f}s{left:>9.2f}s{lo:>12.1f} - {hi:<9.1f}")

    print()
    print("=" * 78)
    print("3. DNSMOS STUDENT: what fits?  (peak = conv1 output, the largest tensor)")
    print("=" * 78)
    print(f"{'window':>7}{'T':>5}{'bins':>6}{'c1':>5}{'params':>9}"
          f"{'peak fp32':>11}{'peak int8':>11}{'loss GMAC':>11}  fit")
    ref_enh_ms = enh_rows[0][2]
    rows = []
    for win_s in (9.01, 4.0, 2.0, 1.0):
        n_frames = int(round(win_s * SR / 160)) - 1
        input_len = frames_to_len(n_frames)
        for bins in (161, 96, 64, 32):
            for fac in (1.0, 0.5, 0.25, 0.125):
                ch = scaled(OFFICIAL_CONFIG.channels, fac)
                fc = scaled(OFFICIAL_CONFIG.fc, fac)
                if min(ch) < 4 or n_frames // 8 < 1 or bins // 8 < 1:
                    continue
                cfg = DnsmosConfig(input_len=input_len, n_bins=bins, channels=ch, fc=fc)
                peak32 = dnsmos_peak_bytes(cfg)
                peak8 = dnsmos_peak_bytes(cfg, elementwise_int8=True)
                gmac = dnsmos_loss_macs(cfg) / 1e9
                lo, hi = budgets[(enh_rows[0][0], win_s)]
                mem_ok = "on-chip" if peak8 <= NOEXTMEM else ("PSRAM" if peak8 <= PSRAM else "--")
                time_ok = gmac <= hi
                rows.append((cfg, win_s, bins, ch[0], peak32, peak8, gmac, mem_ok, time_ok, lo, hi))
    # Print a readable subset: everything that fits memory somewhere.
    shown = 0
    for cfg, win_s, bins, c1, p32, p8, gmac, mem_ok, time_ok, lo, hi in rows:
        if mem_ok == "--":
            continue
        flag = f"{mem_ok:<8}" + ("time OK" if time_ok else f"too slow (>{hi:.1f})")
        print(f"{win_s:>6.2f}s{cfg.n_frames:>5}{bins:>6}{c1:>5}{dnsmos_params(cfg):>9,}"
              f"{p32/MB:>10.2f}M{p8/MB:>10.2f}M{gmac:>11.2f}  {flag}")
        shown += 1
    if not shown:
        print("  (nothing fits)")

    print()
    print("=" * 78)
    print("4. RECOMMENDED: smallest configs meeting BOTH memory and time")
    print("=" * 78)
    feasible = [r for r in rows if r[7] == "on-chip" and r[8]]
    if not feasible:
        feasible = [r for r in rows if r[7] in ("on-chip", "PSRAM") and r[8]]
    feasible.sort(key=lambda r: (-r[1], -r[2], -r[3]))   # prefer longest window / most bins / widest
    for cfg, win_s, bins, c1, p32, p8, gmac, mem_ok, _, lo, hi in feasible[:6]:
        print(f"  {win_s:.2f}s window, {cfg.n_frames} frames, {bins} bins, channels {cfg.channels}")
        print(f"     params {dnsmos_params(cfg):,} | peak {p8/MB:.2f} MB int8 ({p32/MB:.2f} MB if "
              f"elementwise stays fp32) | {gmac:.2f} GMAC vs budget {lo:.1f}-{hi:.1f}")
    print()
    print("Caveats: MAC-derived timing, not measured on target; DNSMOS quality at a")
    print("shortened window is an empirical question (validate_student.py); and the")
    print("fp32 column assumes today's op_types_to_quantize=[Conv,Gemm,MatMul].")


if __name__ == "__main__":
    main()
