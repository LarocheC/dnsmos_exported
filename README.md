# dnsmos-trainable

A trainable PyTorch reimplementation of Microsoft's **DNSMOS P.835** speech-quality
model (SIG/BAK/OVRL), exported to ONNX in two complementary forms for **on-device**
use with **stock ONNX Runtime** or the **STM32N6** (Neural-ART NPU via ST Edge AI):

1. **Int8 QDQ forward** — cheap inference of the metric.
2. **Fused forward+backward "loss graph"** — DNSMOS as a **frozen differentiable
   perceptual loss**: a plain inference ONNX graph whose outputs are the scores
   *and* `dL/d(waveform)`, so an upstream model (e.g. a speech enhancer) can be
   trained on device against the DNSMOS score. No training runtime required —
   the backward pass is hand-written as ordinary ONNX ops.

## Artifact contract

| Artifact | I/O | Purpose |
|---|---|---|
| `dnsmos_fwd_fp32.onnx` | `wav [B,144160]` → `raw [B,3]`, `mos [B,3]` | fp32 reference metric (opset 20, dynamic batch) |
| `dnsmos_fwd_int8_qdq.onnx` | same | int8 QDQ metric, desktop |
| `dnsmos_loss_fp32.onnx` | `wav [B,144160]`, `w [3]` → `raw`, `mos`, `grad_wav [B,144160]` | perceptual loss + exact input gradients |
| `dnsmos_fwd_fp32_stm32n6.onnx` | `wav [1,901,160]` → `raw`, `mos` | device forward (opset 13, static shapes) |
| `dnsmos_fwd_int8_qdq_stm32n6.onnx` | same | device int8 metric (Neural-ART fast path) |
| `dnsmos_loss_fp32_stm32n6.onnx` | `wav [1,901,160]`, `w [3]` → `raw`, `mos`, `grad_wav [1,901,160]` | device loss graph, float |
| `dnsmos_loss_int8_qdq.onnx` | flat I/O, as `dnsmos_loss_fp32.onnx` | int8 loss graph (quantized forward *and* backward), desktop |
| `dnsmos_loss_int8_qdq_stm32n6.onnx` | rows I/O, as `dnsmos_loss_fp32_stm32n6.onnx` | int8 loss graph, device |

- `wav`: 16 kHz float32, 9.01 s (144160 samples). The STM32N6 layout is
  `[1, 901, 160]` — one row per 10 ms hop (`wav.reshape(1, 901, 160)`) — because
  the ST front-end rejects tensor dims ≥ 65536.
- `w [3]`: loss weights over `(SIG, BAK, OVRL)` **mapped** scores; the graph
  computes `L = Σ_batch w · mos` and `grad_wav = dL/dwav`. Use `w = [0, 0, -1]`
  and *subtract* the gradient to ascend OVRL.
- `raw` are the network outputs; `mos` applies the official polynomial mapping
  (differentiable, in-graph).

## Quickstart

```bash
pip install -e .[dev]
python scripts/download_official.py        # official sig_bak_ovr.onnx (sha256-pinned)
python scripts/transplant_weights.py       # exact PyTorch port + parity gates
python scripts/export_all_fp32.py          # 4 fp32 artifacts
python scripts/qat_finetune_int8.py        # QAT for the int8 forward (see below)
python scripts/export_int8.py              # int8 forward artifacts + delta gates
python scripts/export_int8_loss.py         # int8 loss artifacts + gradient gates
python examples/ondevice_demo.py           # PyTorch-free end-to-end proof
pytest -m "not slow"                       # full verification suite
```

The demo (`onnxruntime` + `numpy` only) optimizes a waveform against the loss
graph with an SI-SNR anchor: OVRL typically rises ~1.3 → 3.6 in 60 steps while
SI-SNR stays ≈ 29 dB, then reports the int8 "deployed" score.

## How it works

**Exact transplant, not retraining.** The official `sig_bak_ovr.onnx` embeds its
featurization in the graph, and its `stft-real`/`stft-imag` kernels are
*trained* matrices: close to — but not exactly — a hann-windowed DFT
(per-bin cosine similarity 0.93–0.98 on mid-band bins; they look
DFT-initialized and mildly trained end-to-end). Exact parity requires the
verbatim weights, so they are ported as-is, never re-derived. Parity gate: in
float64 the port matches the official model to 5.8e-6, which is ORT's own
fp32 rounding; the framing is bit-exact. A distillation harness (`scripts/train_distill.py`,
teacher = official ONNX) exists for fine-tuning, architecture changes, and the
compact student — the transplant itself needs no training.

**Hand-written backward.** `DnsmosLossGraph` recomputes the forward and chains
explicit vector-Jacobian products — no autograd anywhere — so the "training"
graph is a plain inference graph any runtime can execute (ST's Neural-ART is
inference-only; ONNX Runtime's on-device training API is deprecated; automated
joint-graph export to ONNX has unresolved gaps). Verified against
`torch.autograd.grad` to < 1e-4 relative in both op-vocabulary modes:

- `device` (default): STM32N6-safe — conv input-grads as plain `Conv` with
  pre-flipped constant weights, equality-mask max routing with concat-interleave
  upsampling, `Equal`-based masks, `Reciprocal` instead of tensor division, tie
  counts via `AvgPool`/`ReduceMean` + `Clip`. No `ScatterElements`,
  `ConvTranspose`, `Expand`, `Where`, `Greater`, `ReduceSum`, `Resize`.
- `ort`: exact autograd tie semantics via index scatter, for desktop.

**Quantization (ss/sa, ST-compatible).** Weights per-channel symmetric int8
(zero-point 0), activations per-tensor asymmetric int8. Two hard-won findings:

- *Percentile calibration is poison for this model*: DNSMOS max-pools at every
  stage, so clipping activation ceilings collapses the very peaks the pooling
  selects (and at int8, tie-saturates them). Ranges are min/max.
- *Recalibration after QAT throws the training away*: the int8 forward is
  produced by `qdq_writer.write_qdq_from_sim`, which stamps the QAT sim's own
  quantization grids into the graph instead of re-calibrating, so the deployed
  artifact inherits the sim's measured deltas exactly.
- *Float tail*: the dense head (+ the a7 activation) stays float — it is
  ~0.4% of the MACs but gates the heavy-tailed SIG error through the global
  max. Pass `--full-int8` to `scripts/export_int8.py` to quantize it anyway.

Measured deltas vs the fp32 reference (100 synthetic segments): mean |ΔMOS|
0.037 / 0.016 / 0.017 and p95 0.145 / 0.054 / 0.061 for SIG / BAK / OVRL.
Gates: mean < 0.05 everywhere; p95 < 0.10 for BAK/OVRL, < 0.18 for SIG
(intrinsically the noisiest output under int8 — its error enters through the
global-max routing, which propagates worst-case quantization noise).

**Int8 gradient reality.** An int8-quantized forward is a different (staircase)
function from the fp32 model; even PyTorch's own straight-through-estimator
gradient of a fake-quant model has mean cosine ≈ 0.77 to the fp32 gradient.
Demanding near-1.0 cosine from an int8 loss graph is physically meaningless, so
the gates are:

- **Functional (hard):** optimizing a waveform with int8 gradients for 60 steps
  must raise the fp32-reference OVRL by ≥ +0.3. (Current artifact: **+1.61**,
  vs ≈ +2.3 with fp32 gradients.)
- **Cosine tripwire (hard):** mean cosine ≥ 0.4 vs fp32 gradients.

If you need maximum gradient fidelity, use the fp32 loss graph — on the N6 it
runs as software epochs on the Cortex-M55 (Helium), slower but exact.

## STM32N6 notes (ST Edge AI Core / X-CUBE-AI)

- Compile with `stedgeai generate --model <artifact> --target stm32n6
  --st-neural-art` and add **`--native-float`** for any artifact that keeps
  float ops (all of them: the log frontend is float by design — one contiguous
  software epoch). Use `--mapping-recap` to inspect NPU/CPU placement and
  `--no-outputs-allocation` to place the 564 KB gradient buffer yourself.
- Every device artifact is linted by `verify.check_device_constraints` at
  export time (the fp32 exports lint inside `export_forward`/
  `export_loss_graph`; the int8 scripts lint their outputs): opset ≤ 13,
  static shapes, batch 1, **all** tensor dims < 65536 (interior tensors
  included, via shape inference), and an op vocabulary restricted to the
  documented ST Neural-ART mapping table.
- Memory: the full-size loss graph peaks ≈ 18.5 MB of int8 activations →
  external PSRAM territory (STM32N6570-DK: 32 MB hexa-SPI PSRAM; the Nucleo
  board has **no** external RAM). For internal-SRAM-only deployment, distill
  the compact student (`configs/student_small.py`, ≈ 21k body parameters,
  first-conv stride 2 → ≈ 0.6 MB peak activations):
  `python scripts/train_distill.py --student small --train-list ... --val-list ...`
  — the script enforces the quality gate (best-epoch Pearson r ≥ 0.9 vs
  teacher OVRL; `--no-gate` for smoke runs) before the student is considered
  exportable.

## Caveats

- **Metric hacking is real.** Optimizing any no-reference metric produces
  adversarial artifacts if unanchored (see EUSIPCO 2024, "Hallucination in
  Perceptual Metric-Driven Speech Enhancement"): the demo raises OVRL by +2.3
  with a small perturbation. Always pair the DNSMOS loss with an anchor
  (SI-SNR / L1 to the input or a reference), as the demo does.
- Gradients explode near digital silence (`1/p` in the log backward, floored at
  `p = 1e-12`); clip gradients in your optimizer (the demo's Adam does).
- The loss graphs recompute the forward internally; peak memory on desktop is
  ≈ 150 MB/sample fp32 — keep batches ≤ 4.
- Exact-tie gradient routing differs from autograd by design (ties are split
  evenly; autograd picks the first index for max-pool). Measure-zero in fp32;
  ubiquitous and correctly handled at int8.
- QAT weights (`models/dnsmos_qat_int8.pt`) are used **only** for int8
  artifacts; fp32 artifacts always carry the exact transplanted weights.

## License / attribution

Code is MIT. Transplanted and derived weights originate from the official
DNSMOS P.835 model in [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)
(CC BY 4.0) — see `LICENSE` for the required attribution. If you use DNSMOS,
cite: Reddy, Gopal, Cutler, *"DNSMOS P.835: A non-intrusive perceptual
objective speech quality metric to evaluate noise suppressors"*, ICASSP 2022.
