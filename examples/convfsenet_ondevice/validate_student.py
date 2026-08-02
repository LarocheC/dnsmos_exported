#!/usr/bin/env python3
"""Does a shrunken, short-window DNSMOS still work as an on-device training signal?

The feasibility sweep (`explore_feasibility.py`) says which configs *fit*. This
script answers the question the arithmetic cannot: whether a student that small,
seeing that short a window, still produces gradients that improve the REAL
metric.

Pipeline, on real VoiceBank-DEMAND audio (the corpus eco8-neaixt evaluates on):

  1. score clips with the official full-size DNSMOS  -> distillation targets
  2. train the student on short crops to predict them
  3. report agreement (Pearson / Spearman) on held-out clips
  4. export the student's fused fwd+bwd loss graph, lint it for STM32N6
  5. adapt ConvFSENet's mask head using ONLY the student's gradients
  6. score the result with the OFFICIAL model -- the honest test

Step 6 is the one that matters: optimizing a proxy is only useful if the true
metric moves too.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import dsp  # noqa: E402
from dnsmos_trainable.backward import DnsmosLossGraph  # noqa: E402
from dnsmos_trainable.constants import SR, DnsmosConfig  # noqa: E402
from dnsmos_trainable.model import DnsmosModel  # noqa: E402

OFFICIAL_LEN = 144160


def load_vbd(parquet: Path, n_clips: int, seed: int = 0):
    """Real noisy speech, tiled/cropped to the official 9.01 s segment."""
    import pyarrow.parquet as pq
    import soundfile as sf

    table = pq.ParquetFile(str(parquet)).read()
    noisy_col = table.column("noisy")
    idx = np.random.default_rng(seed).permutation(len(noisy_col))[:n_clips]
    out = []
    for i in idx:
        raw = noisy_col[int(i)].as_py()["bytes"]
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        assert sr == SR
        while len(audio) < OFFICIAL_LEN:
            audio = np.concatenate([audio, audio])
        out.append(audio[:OFFICIAL_LEN].astype(np.float32))
    return np.stack(out)


def teacher_scores(official_onnx: Path, clips: np.ndarray, chunk: int = 4) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(official_onnx), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(clips), chunk):
        out.append(sess.run(None, {"input_1": clips[i:i + chunk]})[0])
    return np.concatenate(out)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def train_student(cfg, clips, targets, steps, batch, lr, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DnsmosModel(cfg)
    # DFT-like init for the learned frontend: a sane starting basis beats noise.
    with torch.no_grad():
        n = np.arange(cfg.win)
        k = np.arange(cfg.n_bins)[:, None] * (cfg.n_bins and 1)
        freqs = np.linspace(0, cfg.win // 2, cfg.n_bins)[:, None]
        model.frontend.stft.w_re.copy_(torch.tensor(
            np.cos(2 * np.pi * freqs * n / cfg.win) * np.hanning(cfg.win), dtype=torch.float32))
        model.frontend.stft.w_im.copy_(torch.tensor(
            -np.sin(2 * np.pi * freqs * n / cfg.win) * np.hanning(cfg.win), dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    tgt = torch.from_numpy(targets)
    span = OFFICIAL_LEN - cfg.input_len
    model.train()
    t0 = time.time()
    for step in range(steps):
        sel = rng.integers(0, len(clips), batch)
        starts = rng.integers(0, max(span, 1), batch) if span > 0 else np.zeros(batch, int)
        crops = np.stack([clips[s][o:o + cfg.input_len] for s, o in zip(sel, starts)])
        raw, _ = model(torch.from_numpy(crops))
        loss = torch.nn.functional.mse_loss(raw, tgt[sel])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:4d}  mse {loss.item():.4f}  ({time.time()-t0:.0f}s)")
    model.eval()
    return model


@torch.no_grad()
def student_eval(model, cfg, clips, targets):
    """Centre-crop prediction vs teacher, per output."""
    off = (OFFICIAL_LEN - cfg.input_len) // 2
    crops = np.stack([c[off:off + cfg.input_len] for c in clips])
    preds = []
    for i in range(0, len(crops), 8):
        preds.append(model(torch.from_numpy(crops[i:i + 8]))[0].numpy())
    preds = np.concatenate(preds)
    return preds, {
        name: (pearson(preds[:, j], targets[:, j]), spearman(preds[:, j], targets[:, j]))
        for j, name in enumerate(("SIG", "BAK", "OVRL"))
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vbd-parquet", type=Path, required=True)
    ap.add_argument("--official", type=Path, default=HERE.parents[1] / "models" / "sig_bak_ovr.onnx")
    ap.add_argument("--n-clips", type=int, default=160)
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--width", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--cache", type=Path, default=HERE / "artifacts" / "vbd_cache.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    ap.add_argument("--skip-loop", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache.exists():
        z = np.load(args.cache)
        clips, targets = z["clips"], z["targets"]
        print(f"cache: {len(clips)} clips")
    else:
        print(f"loading {args.n_clips} VoiceBank-DEMAND clips...")
        clips = load_vbd(args.vbd_parquet, args.n_clips)
        print("scoring with the official DNSMOS (teacher)...")
        t0 = time.time()
        targets = teacher_scores(args.official, clips)
        print(f"  {time.time()-t0:.0f}s")
        np.savez_compressed(args.cache, clips=clips, targets=targets)
    print(f"teacher raw OVRL: {targets[:,2].min():.2f}..{targets[:,2].max():.2f} "
          f"(mean {targets[:,2].mean():.2f}, sd {targets[:,2].std():.2f})")

    n_frames = int(round(args.window_s * SR / 160)) - 1
    input_len = 320 + (n_frames - 1) * 160
    base = (128, 64, 64, 32, 32, 32, 64)
    ch = tuple(max(4, int(round(c * args.width))) for c in base)
    fc = tuple(max(4, int(round(c * args.width))) for c in (128, 64))
    cfg = DnsmosConfig(input_len=input_len, n_bins=args.bins, channels=ch, fc=fc)
    print(f"\nstudent: {cfg}")
    print(f"  peak activation {cfg.peak_activation_elems/2**20:.2f} MB int8 "
          f"({cfg.peak_activation_elems*4/2**20:.2f} MB fp32)")

    tr, ho = clips[:-args.holdout], clips[-args.holdout:]
    tr_t, ho_t = targets[:-args.holdout], targets[-args.holdout:]
    print(f"\ntraining on {len(tr)} clips, {args.steps} steps")
    model = train_student(cfg, tr, tr_t, args.steps, args.batch, args.lr)

    _, corr = student_eval(model, cfg, ho, ho_t)
    print(f"\nheld-out agreement with the official teacher ({len(ho)} clips):")
    for name, (p, s) in corr.items():
        print(f"  {name:<5} Pearson {p:+.3f}   Spearman {s:+.3f}")

    ckpt = args.out_dir / "dnsmos_student.pt"
    torch.save({"state_dict": model.state_dict(),
                "cfg": dict(input_len=cfg.input_len, n_bins=cfg.n_bins,
                            channels=cfg.channels, fc=cfg.fc)}, ckpt)
    print(f"\nsaved {ckpt.name}")

    from dnsmos_trainable.export import export_loss_graph
    from dnsmos_trainable.verify import DNSMOS_DEVICE_OPS, check_device_constraints
    lg = export_loss_graph(DnsmosLossGraph(model, mode="device", io_layout="rows"),
                           args.out_dir / "dnsmos_student_loss_stm32n6.onnx")
    flat = export_loss_graph(DnsmosLossGraph(model, mode="device", io_layout="flat"),
                             args.out_dir / "dnsmos_student_loss.onnx")
    problems = check_device_constraints(lg, max_opset=13, allowed=DNSMOS_DEVICE_OPS)
    print(f"student loss graph: {lg.stat().st_size/1e6:.2f} MB | "
          f"STM32N6 lint: {'OK' if not problems else problems}")
    print(f"(flat variant for the demo loop: {flat.name})")

    if args.skip_loop:
        return
    run_adaptation(args, cfg, flat, clips[-1])


def run_adaptation(args, cfg, student_loss: Path, noisy_full: np.ndarray, steps: int = 30):
    """Adapt ConvFSENet's mask head with the STUDENT's gradients, then score
    the result with the OFFICIAL model. This is the claim that matters."""
    import onnxruntime as ort

    from export_demo_artifacts import export_head_graphs
    from ondevice_train import NpAdam, OnDeviceEnhancer

    trunk_art = args.out_dir / "convfsenet_trunk_int8.onnx"
    if not trunk_art.exists():
        print("\n(skipping loop: run export_demo_artifacts.py first)")
        return

    noisy = noisy_full[: cfg.input_len].astype(np.float32)
    conv_T = len(noisy) // 256 + 1
    print(f"\nadapting ConvFSENet on a {len(noisy)/SR:.2f} s window "
          f"({conv_T} ConvFSENet frames) using ONLY the student's gradients")
    # Head graphs are window-specific; write them beside the trunk in a private
    # directory so a student run never clobbers the demo's shared T=564 pair.
    loop_dir = args.out_dir / f"loop_T{conv_T}"
    loop_dir.mkdir(parents=True, exist_ok=True)
    for name in ("convfsenet_trunk_int8.onnx", "convfsenet_trunk_fp32.onnx"):
        src = args.out_dir / name
        if src.exists() and not (loop_dir / name).exists():
            (loop_dir / name).write_bytes(src.read_bytes())
    export_head_graphs(256, conv_T, loop_dir)

    eng = OnDeviceEnhancer(loop_dir, student_loss, use_int8_trunk=True)
    official = ort.InferenceSession(str(args.official), providers=["CPUExecutionProvider"])

    def official_ovrl(x: np.ndarray) -> np.ndarray:
        y = x
        while len(y) < OFFICIAL_LEN:
            y = np.concatenate([y, y])
        return official.run(None, {"input_1": y[:OFFICIAL_LEN][None].astype(np.float32)})[0][0]

    X, mag = eng.analyse(noisy)
    h = eng.run_trunk(mag)
    W = np.zeros((eng.n_features, 192), dtype=np.float64)
    b = np.full(eng.n_features, 4.0, dtype=np.float64)
    opt = NpAdam([W.shape, b.shape], lr=3e-3)
    w_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    base = official_ovrl(noisy)
    print(f"  official raw scores, noisy input   SIG {base[0]:.3f} BAK {base[1]:.3f} OVRL {base[2]:.3f}")
    for step in range(steps):
        mask = eng.mask_of(h, W, b)
        enhanced, _ = eng.synthesise(X, mask, len(noisy))
        _, g = eng.dnsmos(enhanced, w_vec)                    # STUDENT gradients
        sisnr, g_si = dsp.sisnr_and_grad(enhanced, noisy.astype(np.float64))
        gw = g.astype(np.float64)
        if sisnr < 6.0:
            gw -= 0.5 * g_si
        gY = dsp.istft_adjoint(gw, T=X.shape[1])
        dmask = dsp.mask_grad(X, gY)[: eng.n_features][None]
        dW, db = eng.head_grads(dmask, mask, h)
        uW, ub = opt.step([dW, db])
        W -= uW; b -= ub

    mask = eng.mask_of(h, W, b)
    enhanced, _ = eng.synthesise(X, mask, len(noisy))
    final = official_ovrl(enhanced)
    sisnr, _ = dsp.sisnr_and_grad(enhanced, noisy.astype(np.float64))
    print(f"  official raw scores, after {steps:2d} steps SIG {final[0]:.3f} "
          f"BAK {final[1]:.3f} OVRL {final[2]:.3f}   SI-SNR {sisnr:.1f} dB")
    print(f"  TRUE OVRL change: {final[2]-base[2]:+.3f}  "
          f"(BAK {final[1]-base[1]:+.3f}, SIG {final[0]-base[0]:+.3f})")
    verdict = "the proxy's gradients move the real metric" if final[2] > base[2] + 0.05 \
        else "NO reliable improvement in the true metric"
    print(f"  => {verdict}")


if __name__ == "__main__":
    main()
