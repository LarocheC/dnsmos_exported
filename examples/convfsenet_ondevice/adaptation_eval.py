#!/usr/bin/env python3
"""How big is the effect, and how big is the noise?

Every design decision in FEASIBILITY.md rests on one number: the change in the
OFFICIAL model's OVRL after adapting ConvFSENet's mask head with some gradient
source. Until now that number came from ONE clip, and 5c showed the resulting
noise floor is comparable to the differences being ranked -- the 1 s int8 build
"beat" its own fp32 build, which cannot happen.

This runs the same adaptation over many clips and several head initializations
and reports the spread. Two things make it much sharper than repeating the
single-clip test:

  * PAIRED comparison. Every arm sees the identical (clip, seed) list, so the
    per-pair difference cancels the clip-to-clip variation that dominates the
    raw spread. A 0.5 MOS spread across clips can hide a 0.05 MOS difference
    between arms that is nonetheless perfectly consistent pair by pair.
  * BOOTSTRAP intervals over pairs, so nothing is assumed about the shape of
    the distribution (these deltas are visibly skewed -- a clip that is already
    clean has little room to improve).

    python adaptation_eval.py --clips 30 --seeds 2

Needs the crop artifacts (`crop_study.py`) and the VBD cache
(`validate_student.py`).

KNOWN LIMIT, and it bounds what this can settle. The official judge needs
9.01 s, so a shorter crop is tiled up to that length -- 4.5x for the 2 s arms,
9x for the 1 s arms. Comparisons WITHIN a window are clean (both arms hand the
judge an identically-constructed signal). Comparisons ACROSS windows are not:
the judge is looking at differently-built inputs, so a 1 s-vs-2 s difference
here confounds window length with tiling. Settling that one properly means
adapting on the crop and then scoring the learned mask applied to the FULL
9.01 s clip, which also tests something more useful -- whether adaptation on a
short window generalizes off it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from crop_study import (  # noqa: E402
    adapt_one,
    build_engine,
    clips_from_cache,
    cropped_config,
    official_session,
)


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile CI for the mean, resampling the observations."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=2,
                    help="head initializations per clip (0 = the fixed W=0,b=4 init only)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--skip-clips", type=int, default=16,
                    help="clips reserved for calibration/gradient eval in crop_study.py")
    ap.add_argument("--cache", type=Path, default=HERE / "artifacts" / "vbd_cache.npz")
    ap.add_argument("--official", type=Path,
                    default=HERE.parents[1] / "models" / "sig_bak_ovr.onnx")
    ap.add_argument("--demo-artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--arms", type=str, default=None,
                    help="comma-separated substrings; run only matching arms "
                         "(the first one kept becomes the paired reference)")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "adaptation_eval.npz")
    args = ap.parse_args()

    # (label, loss graph, window seconds). The student is the anchor: its
    # effect is known to be large and negative, so if the protocol cannot
    # separate it from the rest, the protocol is broken.
    A2, A1, AD = HERE / "artifacts_crop", HERE / "artifacts_crop1s", HERE / "artifacts"
    A1EW = HERE / "artifacts_crop1s_ew"
    candidates = [
        ("fp32 2 s crop",      A2 / "crop2s_fp32.onnx",    2.0),
        ("int8 2 s, excl 0",   A2 / "crop2s_int8_d0.onnx", 2.0),
        ("int8 2 s, excl 2",   A2 / "crop2s_int8_d2.onnx", 2.0),
        ("fp32 1 s crop",      A1 / "crop1s_fp32.onnx",    1.0),
        ("int8 1 s, excl 0",   A1 / "crop1s_int8_d0.onnx", 1.0),
        ("int8 1 s, excl 2",   A1 / "crop1s_int8_d2.onnx", 1.0),
        ("int8 1 s ew, excl 0", A1EW / "crop1s_int8_d0.onnx", 1.0),
        ("int8 1 s ew, excl 2", A1EW / "crop1s_int8_d2.onnx", 1.0),
        ("distilled student",  AD / "dnsmos_student_loss.onnx", None),
    ]
    if args.arms:
        keep = [a.strip() for a in args.arms.split(",")]
        unknown = [k for k in keep if not any(k in lab for lab, _, _ in candidates)]
        if unknown:
            raise SystemExit(f"no arm matches {unknown}; known: "
                             f"{[lab for lab, _, _ in candidates]}")
        candidates = [c for c in candidates if any(k in c[0] for k in keep)]
    arms = [(lab, p, w) for lab, p, w in candidates if p.exists()]
    missing = [lab for lab, p, _ in candidates if not p.exists()]
    if missing:
        print(f"skipping (artifact absent): {', '.join(missing)}")
    if not arms:
        raise SystemExit("no arms available -- run crop_study.py first")

    clips = clips_from_cache(args.cache)[args.skip_clips: args.skip_clips + args.clips]
    seeds = [None] if args.seeds == 0 else list(range(args.seeds))
    n_pairs = len(clips) * len(seeds)
    print(f"{len(arms)} arms x {len(clips)} clips x {len(seeds)} init(s) "
          f"= {len(arms) * n_pairs} adaptations of {args.steps} steps\n")

    official_scores = official_session(args.official)
    results: dict[str, np.ndarray] = {}
    sisnrs: dict[str, np.ndarray] = {}

    for label, path, window_s in arms:
        # The student graph carries its own window; read it off the graph
        # rather than assuming, so an arm can never be scored at the wrong
        # crop length.
        eng = build_engine(path, args.demo_artifacts,
                           _input_len(path, window_s), args.threads)
        input_len = int(np.prod(eng.loss_wav_shape))
        t0 = time.time()
        d, s = [], []
        for clip in clips:
            for seed in seeds:
                base, final, sisnr = adapt_one(eng, official_scores, clip,
                                               input_len, args.steps, seed=seed)
                d.append(final[2] - base[2])
                s.append(sisnr)
        results[label] = np.array(d)
        sisnrs[label] = np.array(s)
        print(f"  {label:<20} done in {time.time()-t0:5.0f}s  "
              f"mean {results[label].mean():+.3f}")

    # ---- report -----------------------------------------------------------
    print(f"\n{'='*78}\nΔ TRUE OVRL over {n_pairs} (clip, init) pairs\n{'='*78}")
    print(f"{'arm':<20}{'mean':>8}{'sd':>8}{'95% CI of mean':>22}{'win rate':>10}{'SI-SNR':>9}")
    for label in results:
        d = results[label]
        lo, hi = bootstrap_ci(d)
        print(f"{label:<20}{d.mean():>+8.3f}{d.std(ddof=1):>8.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]{(d > 0).mean()*100:>9.0f}%"
              f"{sisnrs[label].mean():>8.1f}dB")

    ref = arms[0][0]
    print(f"\n{'='*78}\nPAIRED differences vs '{ref}' (same clips, same inits)\n{'='*78}")
    print(f"{'arm':<20}{'mean diff':>11}{'sd':>8}{'95% CI':>22}  verdict")
    for label in results:
        if label == ref:
            continue
        diff = results[label] - results[ref]
        lo, hi = bootstrap_ci(diff)
        if lo > 0:
            verdict = "BETTER than ref"
        elif hi < 0:
            verdict = "WORSE than ref"
        else:
            verdict = "indistinguishable"
        print(f"{label:<20}{diff.mean():>+11.3f}{diff.std(ddof=1):>8.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]  {verdict}")

    np.savez(args.out, **{f"d_{k}": v for k, v in results.items()},
             **{f"sisnr_{k}": v for k, v in sisnrs.items()})
    print(f"\nraw per-pair deltas saved to {args.out.name}")
    print("A CI that straddles zero means the single-clip ranking was noise, "
          "not a finding.")


def _input_len(path: Path, window_s: float | None) -> int:
    """Window length the graph itself expects."""
    if window_s is not None:
        return cropped_config(window_s).input_len
    import onnx

    m = onnx.load(str(path))
    dims = m.graph.input[0].type.tensor_type.shape.dim
    return int(np.prod([d.dim_value for d in dims if d.dim_value > 0]))


if __name__ == "__main__":
    main()
