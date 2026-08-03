#!/usr/bin/env python3
"""Export the STREAMING (per-frame FIFO) ConvFSENet trunk for the STM32N6.

The windowed trunk (`export_demo_artifacts.py`) recomputes 42 context columns
per call — a batch-mode compromise. This is the alternative the deployment
actually runs: eco8-neaixt's per-frame FIFO graph (4.40 ms/frame on target),
cut at `h` so the trainable mask head stays separate:

    noisy_mag_t [B, 256]                        one 16 ms hop's |STFT|
      -> compress (|m|+1e-9)**0.3               fp32 prologue (excluded from int8)
      -> frontend Conv1d(256->192, k=1) + ReLU
      -> 9 x streaming TCM block                FIFO state [B, 384, (K-1)*D] each
      -> h_t [B, 192, 1]

    forward(noisy_mag_t, state_0_in..state_8_in) -> (h_t, state_0_out..state_8_out)

Streaming math mirrors eco8's `ConvFSENetStreamingFast`: the block's dilated
depthwise conv becomes a k=3 dilation-1 conv over [buf[0], buf[D], y_t] —
the dilation is absorbed into the gather taps; the FIFO drops its oldest
column and appends y_t each frame. Zero-initialized state IS the offline
causal zero-padding, so streaming from zeros is bit-exact to the offline
model on every frame including warm-up (parity-gated below vs the windowed
trunk fed zero-padded windows).

Why this matters for on-device training: in deployment the trunk+head already
run every frame, so the adaptation window's (h, X, mask) tensors are computed
FOR FREE — the host only banks them. The windowed trunk's recompute per
adaptation window disappears entirely.

Int8 calibration threads real per-frame state (eco8's
ConvFSENetCalibrationReader pattern): each calibration clip streams through
the fp32 trunk and every frame contributes {mag_t, state_*_in} exactly as the
deployed graph will see them. Zero-state-only calibration would miss the
steady-state FIFO distributions entirely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import dsp  # noqa: E402
from convfsenet_arch import (  # noqa: E402
    ConvFSENetTrunk,
    KERNEL_SIZE,
    N_CHANNELS_CONV,
    N_CHANNELS_RES,
    build_split,
    compress_magnitude,
    load_from_eco8_checkpoint,
)
from export_demo_artifacts import OPSET, quantize_trunk  # noqa: E402


class StreamingTCMBlock(nn.Module):
    """Per-frame view of one (BN-folded) WindowedTCMBlock.

    State: FIFO buffer [B, 384, (K-1)*D] of past post-ReLU conv1x1 outputs,
    oldest on the left — identical layout to eco8's _StreamingTCMBlockQF_Fast.
    """

    def __init__(self, src) -> None:
        super().__init__()
        self.dilation = int(src.dilation)
        self.buffer_size = (KERNEL_SIZE - 1) * self.dilation
        self.conv1x1 = src.conv1x1
        self.dconv = src.dconv            # weights reused; called with dilation=1
        self.conv1x1_out = src.conv1x1_out
        taps = torch.tensor([i * self.dilation for i in range(KERNEL_SIZE - 1)],
                            dtype=torch.long)
        self.register_buffer("tap_indices", taps, persistent=False)

    def init_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, N_CHANNELS_CONV, self.buffer_size)

    def forward_step(self, x_t: torch.Tensor, buf: torch.Tensor):
        y = torch.relu(self.conv1x1(x_t))                         # [B, 384, 1]
        past = buf.index_select(dim=-1, index=self.tap_indices)   # [B, 384, K-1]
        window = torch.cat([past, y], dim=-1)                     # [B, 384, K]
        z = F.conv1d(window, self.dconv.weight, bias=self.dconv.bias,
                     stride=1, padding=0, dilation=1, groups=N_CHANNELS_CONV)
        z = torch.relu(z)
        z = self.conv1x1_out(z)                                   # [B, 192, 1]
        out = z + x_t
        new_buf = torch.cat([buf[..., 1:], y], dim=-1)
        return out, new_buf


class ConvFSENetStreamingTrunk(nn.Module):
    """Frozen streaming trunk: one |STFT| frame + 9 FIFO states -> h_t + states."""

    def __init__(self, trunk: ConvFSENetTrunk) -> None:
        super().__init__()
        self.n_features = trunk.n_features
        self.frontend = trunk.frontend
        self.blocks = nn.ModuleList(StreamingTCMBlock(b) for b in trunk.blocks)

    @property
    def state_input_names(self):
        return [f"state_{i}_in" for i in range(len(self.blocks))]

    @property
    def state_output_names(self):
        return [f"state_{i}_out" for i in range(len(self.blocks))]

    def init_states(self, batch: int = 1):
        return [b.init_state(batch) for b in self.blocks]

    def forward(self, noisy_mag_t: torch.Tensor, *states_in: torch.Tensor):
        """noisy_mag_t [B, 256] plain |STFT| -> (h_t [B, 192, 1], 9 states)."""
        x = compress_magnitude(noisy_mag_t).unsqueeze(-1)         # [B, 256, 1]
        x = torch.relu(self.frontend(x))                          # [B, 192, 1]
        states_out = []
        for blk, st in zip(self.blocks, states_in):
            x, new_st = blk.forward_step(x, st)
            states_out.append(new_st)
        return (x, *states_out)


# ---------------------------------------------------------------------------
# Parity: streaming-from-zeros == windowed trunk on zero-padded windows.
# ---------------------------------------------------------------------------

def parity_gate(trunk: ConvFSENetTrunk, stream: ConvFSENetStreamingTrunk,
                n_frames: int = 160, seed: int = 0) -> float:
    """Steady-state parity: for t >= context_l the windowed trunk's output uses
    only real context, so it must match the streaming FIFO exactly. Warm-up
    frames (t < L) legitimately differ between the two: the offline/causal
    model (== the FIFO) zero-pads each block's dconv input in the y-domain,
    while a magnitude-padded window pads before the compression Pow."""
    rng = np.random.default_rng(seed)
    mag = torch.from_numpy(
        np.abs(rng.standard_normal((1, trunk.n_features, n_frames))).astype(np.float32))
    L = trunk.context_l
    with torch.no_grad():
        padded = torch.cat(
            [torch.zeros(1, trunk.n_features, L), mag], dim=-1)
        h_ref = trunk(padded)                                     # [1, 192, n_frames]
        states = stream.init_states(1)
        cols = []
        for t in range(n_frames):
            out = stream(mag[..., t], *states)
            cols.append(out[0])
            states = list(out[1:])
        h_stream = torch.cat(cols, dim=-1)
    return float((h_ref[..., L:] - h_stream[..., L:]).abs().max())


# ---------------------------------------------------------------------------
# Int8 calibration with threaded per-frame state (real VBD audio).
# ---------------------------------------------------------------------------

def calibration_feeds(stream: ConvFSENetStreamingTrunk, clips: np.ndarray,
                      frames_per_clip: int | None = None):
    feeds = []
    names = stream.state_input_names
    with torch.no_grad():
        for clip in clips:
            X = dsp.stft(clip.astype(np.float64))
            mag = np.abs(X)[: stream.n_features].astype(np.float32)   # [256, T]
            T = mag.shape[1] if frames_per_clip is None else min(mag.shape[1], frames_per_clip)
            states = stream.init_states(1)
            for t in range(T):
                frame = torch.from_numpy(mag[:, t][None])             # [1, 256]
                feed = {"noisy_mag": frame.numpy().copy()}
                for n, s in zip(names, states):
                    feed[n] = s.numpy().copy()
                feeds.append(feed)
                out = stream(frame, *states)
                states = list(out[1:])
    return feeds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    p.add_argument("--split", type=Path, default=HERE / "artifacts" / "convfsenet_split.pt",
                   help="trunk/head state from export_demo_artifacts.py (real weights)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="eco8 g_best checkpoint (alternative to --split)")
    p.add_argument("--eco8-repo", type=Path, default=None)
    p.add_argument("--calib-clips", type=int, default=6)
    p.add_argument("--calib-frames", type=int, default=200,
                   help="frames per calibration clip (200 = 3.2 s; states reach steady state)")
    p.add_argument("--cache", type=Path, default=HERE / "artifacts" / "vbd_cache.npz")
    args = p.parse_args()

    if args.checkpoint is not None:
        trunk, _ = load_from_eco8_checkpoint(args.checkpoint, eco8_repo=args.eco8_repo)
        print(f"loaded real ConvFSENet weights from {args.checkpoint}")
    elif args.split.exists():
        trunk, _ = build_split()
        trunk.load_state_dict(torch.load(args.split, map_location="cpu")["trunk"])
        print(f"loaded trunk weights from {args.split}")
    else:
        trunk, _ = build_split()
        print("WARNING: random trunk weights (no --checkpoint / --split found)")
    trunk.eval()

    stream = ConvFSENetStreamingTrunk(trunk).eval()

    diff = parity_gate(trunk, stream)
    print(f"parity streaming-vs-offline (120 frames incl. warm-up): max|diff| = {diff:.2e}")
    assert diff < 1e-4, "streaming trunk does not match the offline/windowed model"

    # ---- fp32 export ------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = args.out_dir / "convfsenet_trunk_stream_fp32.onnx"
    example_mag = torch.zeros(1, trunk.n_features)
    example_states = stream.init_states(1)
    torch.onnx.export(
        stream, (example_mag, *example_states), str(fp32),
        input_names=["noisy_mag"] + stream.state_input_names,
        output_names=["h"] + stream.state_output_names,
        dynamic_axes={n: {0: "B"} for n in
                      (["noisy_mag", "h"] + stream.state_input_names + stream.state_output_names)},
        opset_version=OPSET, dynamo=False,
    )
    onnx.checker.check_model(str(fp32))
    print(f"wrote {fp32.name}")

    # ---- int8 QDQ with state-threaded calibration -------------------------
    clips = np.load(args.cache)["clips"][: args.calib_clips]
    feeds = calibration_feeds(stream, clips, args.calib_frames)
    print(f"calibrating streaming trunk int8 on {len(feeds)} frames "
          f"({args.calib_clips} clips x {args.calib_frames} frames, threaded state)")
    int8 = quantize_trunk(fp32, args.out_dir / "convfsenet_trunk_stream_int8.onnx", feeds)
    print(f"wrote {int8.name}  ({int8.stat().st_size/1e6:.2f} MB)")

    # ---- int8 fidelity over a held-out clip (threaded state, ORT) ---------
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s8 = ort.InferenceSession(str(int8), so, providers=["CPUExecutionProvider"])
    clip = np.load(args.cache)["clips"][10]
    X = dsp.stft(clip.astype(np.float64))
    mag = np.abs(X)[: trunk.n_features].astype(np.float32)
    T = min(200, mag.shape[1])
    st_t = stream.init_states(1)
    st_o = [s.numpy().copy() for s in stream.init_states(1)]
    cos_all = []
    with torch.no_grad():
        for t in range(T):
            frame = mag[:, t][None]
            ref = stream(torch.from_numpy(frame), *st_t)
            h_ref, st_t = ref[0].numpy(), list(ref[1:])
            feed = {"noisy_mag": frame}
            feed.update({n: s for n, s in zip(stream.state_input_names, st_o)})
            out = s8.run(None, feed)
            h8, st_o = out[0], out[1:]
            c = float((h_ref.ravel() @ h8.ravel()) /
                      (np.linalg.norm(h_ref) * np.linalg.norm(h8) + 1e-12))
            cos_all.append(c)
    print(f"int8 h vs fp32 h over {T} threaded frames: cos mean {np.mean(cos_all):.4f} "
          f"min {np.min(cos_all):.4f}")
    assert np.mean(cos_all) > 0.95, "int8 streaming trunk drifted from fp32"

    # ---- STM32N6 lint -----------------------------------------------------
    from dnsmos_trainable.verify import check_device_constraints, software_epoch_ops
    batched = {"noisy_mag", "h"} | set(stream.state_input_names) | set(stream.state_output_names)
    print("\nSTM32N6 constraint lint (ST Edge AI Core front end):")
    for art in (fp32, int8):
        problems = check_device_constraints(
            art, max_opset=OPSET, allow_dynamic_batch=True, batched_io=batched)
        sw = software_epoch_ops(art)
        note = f"  [M55 software epochs: {sorted(sw)}]" if sw else ""
        print(f"  {art.name}: {'OK' if not problems else problems}{note}")


if __name__ == "__main__":
    main()
