#!/usr/bin/env python3
"""Window-cropping the OFFICIAL DNSMOS weights, in fp32 and int8.

`validate_student.py` shows that a distilled proxy has gradients orthogonal to
the truth. This script builds the alternative that works: the official topology
is fully convolutional up to a *global* max pool, so the published weights
accept any window >= 8 frames. Nothing is retrained -- the window is simply
shorter.

Steps:

  1. transplant the official weights into a cropped DnsmosConfig
  2. export the fused fwd+bwd loss graph (flat + rows layouts), lint for STM32N6
  3. quantize the flat graph to int8 QDQ at several backward-conv exclusion
     depths, calibrated on real VoiceBank-DEMAND audio
  4. cos(grad_int8, grad_fp32) per clip -- how much the quantized graph's
     gradient field survives
  5. the honest test: adapt ConvFSENet's mask head for 30 steps using ONLY the
     cropped graph's gradients, then score the result with the OFFICIAL model

Step 5 answers "can you train with a quantized DNSMOS?" -- correlation and
cosine do not. (Measured at a 2 s window: yes, retaining ~70% of the fp32
crop's true-metric gain. See FEASIBILITY.md 5b.)

Needs the VoiceBank-DEMAND cache that validate_student.py writes.

    python crop_study.py --window-s 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import dsp  # noqa: E402

from dnsmos_trainable.backward import DnsmosLossGraph  # noqa: E402
from dnsmos_trainable.constants import SR, OFFICIAL_CONFIG, DnsmosConfig  # noqa: E402
from dnsmos_trainable.export import export_loss_graph, quantize_qdq  # noqa: E402
from dnsmos_trainable.verify import DNSMOS_DEVICE_OPS, check_device_constraints  # noqa: E402

OFFICIAL_LEN = 144160
W_VEC = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def cropped_config(window_s: float) -> DnsmosConfig:
    """Official channel widths and bin count, shorter window."""
    n_frames = int(round(window_s * SR / 160)) - 1
    return DnsmosConfig(
        input_len=320 + (n_frames - 1) * 160,
        n_bins=OFFICIAL_CONFIG.n_bins,
        channels=OFFICIAL_CONFIG.channels,
        fc=OFFICIAL_CONFIG.fc,
    )


def load_cropped(ckpt: Path, cfg: DnsmosConfig):
    """The published weights in a cropped geometry.

    Every *parameter* is shape-independent of the window -- the convs are
    translation-invariant and the head sees a globally max-pooled vector -- so
    nothing is dropped or reinitialized here. Only window-derived buffers may
    differ, and any real parameter mismatch is a hard error.
    """
    from dnsmos_trainable.model import DnsmosModel

    model = DnsmosModel(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    params = {n for n, _ in model.named_parameters()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    dropped = [k for k in missing if k in params and not k.startswith("poly.")]
    if dropped or unexpected:
        raise RuntimeError(f"crop changed parameters: missing={dropped} unexpected={unexpected}")
    return model.eval()


def clips_from_cache(cache: Path) -> np.ndarray:
    if not cache.exists():
        raise SystemExit(
            f"missing {cache}. Run validate_student.py once to build the "
            "VoiceBank-DEMAND cache, or pass --cache."
        )
    return np.load(cache)["clips"]


def _session(path: Path, threads: int | None):
    import onnxruntime as ort

    opts = None
    if threads is not None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def determinism_probe(fp32_path: Path, int8_path: Path, clips: np.ndarray,
                      thread_counts=(1, 2, 4)) -> None:
    """Is the gradient a function of the input alone?

    fp32: effectively yes. A few executions differ, but only in the last ULP
    -- gradient cosine stays 1.000 and scores move by ~5e-7 MOS.

    int8: INTERMITTENTLY no. Between 5% and 50% of executions diverge --
    the rate rises with machine load, since it is thread scheduling that
    decides reduction order -- and when they do, the backward is hit far
    harder than the forward: scores stay within ~7e-2 MOS while gradient
    cosine has been seen at 0.036 with a 27x norm blow-up. Because it is
    both intermittent and load-dependent, a single comparison proves nothing
    and a quiet machine flatters the result; this samples several fresh
    sessions per thread count and reports the WORST pair seen.

    The amplifier is tie routing. int8 collapses many activations onto the
    same code, so a max pool's argmax becomes a large tie SET, and the
    backward decides membership with an `Equal` against the pooled value. A
    ULP of reduction-order difference moves elements in and out of that set,
    redirecting gradient mass wholesale. Part of the int8 gradient is
    therefore arbitrary -- which is also why its cosine against fp32 plateaus
    near 0.5 however many convs are excluded from quantization.
    """
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        sessions = [_session(path, t) for t in thread_counts for _ in range(2)]
        n = int(np.prod([d for d in sessions[0].get_inputs()[0].shape if isinstance(d, int)]))
        worst_cos, worst_rat, worst_score, n_diff, n_pairs = 1.0, 1.0, 0.0, 0, 0
        for clip in clips:
            x = clip[:n].astype(np.float32)[None]
            runs = [s.run(None, {"wav": x, "w": W_VEC}) for s in sessions]
            ref_g = runs[0][2].ravel().astype(np.float64)
            ref_s = runs[0][1].ravel()
            for r in runs[1:]:
                g = r[2].ravel().astype(np.float64)
                n_pairs += 1
                if not np.array_equal(ref_g, g):
                    n_diff += 1
                c = ref_g @ g / (np.linalg.norm(ref_g) * np.linalg.norm(g) + 1e-30)
                worst_cos = min(worst_cos, float(c))
                # Report the worst norm DISAGREEMENT, in whichever direction:
                # a divergence that halves the gradient matters as much as one
                # that doubles it.
                rat = float(np.linalg.norm(g) / (np.linalg.norm(ref_g) + 1e-30))
                worst_rat = max(worst_rat, rat, 1.0 / max(rat, 1e-30))
                worst_score = max(worst_score, float(np.abs(ref_s - r[1].ravel()).max()))
        print(f"  {name}: {n_diff}/{n_pairs} executions differed | worst grad cosine "
              f"{worst_cos:.3f} | worst |g'|/|g| {worst_rat:.2f}x | worst score drift "
              f"{worst_score:.1e} MOS")


def grad_cosines(fp32_path: Path, int8_path: Path, clips: np.ndarray,
                 threads: int | None = 1) -> dict:
    """cos(grad_int8, grad_fp32) of the SAME cropped model, per clip."""
    s32 = _session(fp32_path, threads)
    s8 = _session(int8_path, threads)
    n = int(np.prod([d for d in s32.get_inputs()[0].shape if isinstance(d, int)]))
    cos, ratio = [], []
    for clip in clips:
        x = clip[:n].astype(np.float32)[None]
        g32 = s32.run(None, {"wav": x, "w": W_VEC})[2].ravel().astype(np.float64)
        g8 = s8.run(None, {"wav": x, "w": W_VEC})[2].ravel().astype(np.float64)
        d = np.linalg.norm(g32) * np.linalg.norm(g8) + 1e-30
        cos.append(float(g32 @ g8 / d))
        ratio.append(float(np.linalg.norm(g8) / (np.linalg.norm(g32) + 1e-30)))
    return {"cosine_mean": float(np.mean(cos)), "cosine_min": float(np.min(cos)),
            "ratio_mean": float(np.mean(ratio))}


def adapt_and_score(loss_path: Path, official_onnx: Path, art_dir: Path,
                    noisy_full: np.ndarray, input_len: int, steps: int = 30,
                    threads: int | None = 1):
    """Adapt ConvFSENet's mask head with `loss_path` gradients; score with the
    OFFICIAL model. The only number that means anything."""
    import onnxruntime as ort

    from export_demo_artifacts import export_head_graphs
    from ondevice_train import NpAdam, OnDeviceEnhancer

    noisy = noisy_full[:input_len].astype(np.float32)
    conv_T = len(noisy) // 256 + 1
    loop_dir = art_dir / f"loop_T{conv_T}"
    loop_dir.mkdir(parents=True, exist_ok=True)
    for name in ("convfsenet_trunk_int8.onnx", "convfsenet_trunk_fp32.onnx"):
        src = art_dir / name
        if src.exists() and not (loop_dir / name).exists():
            (loop_dir / name).write_bytes(src.read_bytes())
    export_head_graphs(256, conv_T, loop_dir)

    eng = OnDeviceEnhancer(loop_dir, loss_path, use_int8_trunk=True, threads=threads)
    official = ort.InferenceSession(str(official_onnx), providers=["CPUExecutionProvider"])

    def official_scores(x: np.ndarray) -> np.ndarray:
        y = x
        while len(y) < OFFICIAL_LEN:
            y = np.concatenate([y, y])
        return official.run(None, {"input_1": y[:OFFICIAL_LEN][None].astype(np.float32)})[0][0]

    X, mag = eng.analyse(noisy)
    h = eng.run_trunk(mag)
    Wm = np.zeros((eng.n_features, 192), dtype=np.float64)
    b = np.full(eng.n_features, 4.0, dtype=np.float64)
    opt = NpAdam([Wm.shape, b.shape], lr=3e-3)

    base = official_scores(noisy)
    for _ in range(steps):
        mask = eng.mask_of(h, Wm, b)
        enhanced, _ = eng.synthesise(X, mask, len(noisy))
        _, g = eng.dnsmos(enhanced, W_VEC)
        sisnr, g_si = dsp.sisnr_and_grad(enhanced, noisy.astype(np.float64))
        gw = g.astype(np.float64)
        if sisnr < 6.0:
            gw -= 0.5 * g_si
        gY = dsp.istft_adjoint(gw, T=X.shape[1])
        dmask = dsp.mask_grad(X, gY)[: eng.n_features][None]
        dW, db = eng.head_grads(dmask, mask, h)
        uW, ub = opt.step([dW, db])
        Wm -= uW
        b -= ub

    mask = eng.mask_of(h, Wm, b)
    enhanced, _ = eng.synthesise(X, mask, len(noisy))
    final = official_scores(enhanced)
    sisnr, _ = dsp.sisnr_and_grad(enhanced, noisy.astype(np.float64))
    return base, final, sisnr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--weights", type=Path,
                    default=HERE.parents[1] / "models" / "dnsmos_transplanted.pt")
    ap.add_argument("--official", type=Path,
                    default=HERE.parents[1] / "models" / "sig_bak_ovr.onnx")
    ap.add_argument("--cache", type=Path, default=HERE / "artifacts" / "vbd_cache.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "artifacts_crop")
    ap.add_argument("--demo-artifacts", type=Path, default=HERE / "artifacts",
                    help="directory holding the exported ConvFSENet trunk graphs")
    ap.add_argument("--calib-clips", type=int, default=8)
    ap.add_argument("--eval-clips", type=int, default=8)
    ap.add_argument("--ladder", type=int, nargs="*", default=[0, 1, 2],
                    help="backward-conv exclusion depths to try (deepest first)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=1,
                    help="ORT intra-op threads; 1 makes the int8 numbers reproducible")
    ap.add_argument("--skip-loop", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cropped_config(args.window_s)
    print(f"cropped config: {cfg.n_frames} frames, {cfg.n_bins} bins, "
          f"{cfg.input_len} samples ({cfg.input_len/SR:.2f} s)")
    print(f"  analytic peak activation {cfg.peak_activation_elems/2**20:.2f} MB int8 / "
          f"{cfg.peak_activation_elems*4/2**20:.2f} MB fp32")

    model = load_cropped(args.weights, cfg)

    # ---- fp32 loss graphs -------------------------------------------------
    tag = f"crop{args.window_s:g}s".replace(".", "p")
    fp32 = export_loss_graph(DnsmosLossGraph(model, mode="device", io_layout="flat"),
                             args.out_dir / f"{tag}_fp32.onnx")
    rows = export_loss_graph(DnsmosLossGraph(model, mode="device", io_layout="rows"),
                             args.out_dir / f"{tag}_fp32_stm32n6.onnx")
    problems = check_device_constraints(rows, max_opset=13, allowed=DNSMOS_DEVICE_OPS)
    print(f"\nfp32 loss graph {fp32.name} ({fp32.stat().st_size/1e6:.2f} MB)")
    print(f"rows-layout device artifact {rows.name} | "
          f"STM32N6 lint: {'OK' if not problems else problems}")

    # Calibration and gradient evaluation use disjoint clips from the front of
    # the cache; the adaptation clip is the cache's LAST one -- the same clip
    # validate_student.py adapts on, so the two scripts' functional rows are
    # directly comparable.
    clips = clips_from_cache(args.cache)
    calib = clips[: args.calib_clips]
    evalc = clips[args.calib_clips: args.calib_clips + args.eval_clips]

    # ---- int8 exclusion ladder -------------------------------------------
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = args.out_dir / f"{tag}_pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    import onnx

    convs = [n.name for n in onnx.load(str(pre)).graph.node if n.op_type == "Conv"]
    if len(convs) != 14:
        raise RuntimeError(f"expected 14 convs (7 fwd + 7 bwd), got {len(convs)}")
    bwd_convs = list(reversed(convs[7:]))     # backward convs, deepest first
    calib_feeds = [{"wav": c[: cfg.input_len].astype(np.float32)[None], "w": W_VEC}
                   for c in calib]

    print(f"\nint8 QDQ ladder (calibrated on {len(calib)} real VBD clips, MinMax, "
          f"{args.threads} ORT thread{'s' if args.threads != 1 else ''}):")
    results = {}
    for depth in args.ladder:
        out = quantize_qdq(pre, args.out_dir / f"{tag}_int8_d{depth}.onnx", calib_feeds,
                           extra_exclude=bwd_convs[:depth], preprocessed=True)
        rep = grad_cosines(fp32, out, evalc, threads=args.threads)
        results[depth] = (out, rep)
        print(f"  exclude={depth} bwd convs  {out.stat().st_size/1e6:5.2f} MB  "
              f"cos_mean {rep['cosine_mean']:.3f}  cos_min {rep['cosine_min']:.3f}  "
              f"|g8|/|g32| {rep['ratio_mean']:.2f}")

    print("\nreproducibility of the gradient itself (why --threads matters):")
    determinism_probe(fp32, results[args.ladder[0]][0], evalc)

    if args.skip_loop:
        return

    # ---- the honest test --------------------------------------------------
    # Every ladder depth is adapted, not just the cosine-best one: cosine is an
    # agreement metric and agreement does not rank steering quality (that is the
    # whole lesson of validate_student.py). The depth to ship is the one with
    # the best dOVRL, which need not be the one with the best cosine.
    if not (args.demo_artifacts / "convfsenet_trunk_int8.onnx").exists():
        print("\n(skipping adaptation: run export_demo_artifacts.py first)")
        return
    print(f"\nadapting ConvFSENet's mask head for {args.steps} steps, "
          "scored with the OFFICIAL 9.01 s model:")
    runs = [("fp32 crop", fp32)] + [
        (f"int8 crop (exclude={d})", results[d][0]) for d in args.ladder
    ]
    gains = {}
    for i, (label, path) in enumerate(runs):
        base, final, sisnr = adapt_and_score(path, args.official, args.demo_artifacts,
                                             clips[-1], cfg.input_len, args.steps,
                                             threads=args.threads)
        if i == 0:
            print(f"  {'noisy':<28} SIG {base[0]:.3f} BAK {base[1]:.3f} OVRL {base[2]:.3f}")
        gains[label] = final[2] - base[2]
        print(f"  {label:<28} SIG {final[0]:.3f} BAK {final[1]:.3f} OVRL {final[2]:.3f}"
              f"  dOVRL {final[2]-base[2]:+.3f}  SI-SNR {sisnr:.1f} dB")

    int8_gains = {k: v for k, v in gains.items() if k.startswith("int8")}
    best_label = max(int8_gains, key=int8_gains.get)
    cos_best = max(args.ladder, key=lambda d: results[d][1]["cosine_mean"])
    print(f"\n  ship: {best_label}  (dOVRL {int8_gains[best_label]:+.3f}, "
          f"{int8_gains[best_label]/gains['fp32 crop']*100:.0f}% of fp32)")
    if not best_label.endswith(f"exclude={cos_best})"):
        print(f"  note: cosine would have picked exclude={cos_best} "
              f"({results[cos_best][1]['cosine_mean']:.3f}) -- it does not rank steering.")


if __name__ == "__main__":
    main()
