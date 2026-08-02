# ConvFSENet + DNSMOS: on-device training on the STM32N6

A minimal, working demonstration of training a speech enhancer **on the chip**,
with DNSMOS as the quality signal — no training runtime, no autograd, no clean
reference audio.

The enhancer is [ConvFSENet](https://github.com/LarocheC/eco8-neaixt) (the causal
192/384 model already deployed on the STM32N6570-DK at RTF 0.275). The gradient
source is this repo's DNSMOS loss graph. Everything the device executes is an
ordinary ONNX **inference** session, because ST's Neural-ART is inference-only
("no training mode is supported") and ONNX Runtime's on-device training API is
deprecated.

```bash
python export_demo_artifacts.py     # build the 4 ConvFSENet ONNX graphs
python ondevice_train.py --steps 60 # run the loop: onnxruntime + numpy only
python budget.py                    # STM32N6 memory / MAC accounting
```

## What actually runs, per 9.01 s adaptation window

| # | step | where | artifact |
|---|---|---|---|
| 1 | `STFT(noisy)` → `X [257,564]`, `|X|` → 256 bins | M55 / CMSIS-DSP | `dsp.py` |
| 2 | trunk, 9 windowed calls → `h [1,192,564]` | **NPU, int8** | `convfsenet_trunk_int8.onnx` |
| 3 | `mask = sigmoid(W·h + b)` | M55 | `convfsenet_head_fwd.onnx` |
| 4 | `Y = X·mask`, `ISTFT` → enhanced | M55 / CMSIS-DSP | `dsp.py` |
| 5 | scores **and** `dL/d(enhanced)` | **NPU, int8** | `dnsmos_loss_int8_qdq_stm32n6.onnx` |
| 6 | SI-SNR anchor gradient | M55 | `dsp.sisnr_and_grad` |
| 7 | ISTFT-adjoint → mask VJP → `dL/dmask` | M55 | `dsp.istft_adjoint` |
| 8 | `dW [256,192]`, `db [256]` | M55 (both operands dynamic) | `convfsenet_head_bwd.onnx` |
| 9 | Adam step on `(W, b)` | M55 | `NpAdam`, ~20 lines |

## The idea that makes it fit: train the last layer only

ConvFSENet ends with `backend = Conv1d(192→257, k=1) + Sigmoid`, and the mask it
produces is a **real gain applied to the noisy complex STFT**. Two consequences:

1. A gradient arriving at the waveform reaches that layer's weights through
   nothing but an ISTFT-adjoint, a multiply, and a sigmoid derivative — it
   **never enters the 9 TCM blocks**. No trunk activations are stashed; no
   backward pass traverses the trunk.
2. The trainable set is **49,408 parameters** (193 KiB fp32) out of 1.44 M. The
   frozen 1.39 M-parameter trunk keeps running int8 on the NPU exactly as it
   does today.

Adding on-device training to the existing ConvFSENet deployment therefore costs
**0.57 MB** of new persistent state for the head and its Adam moments (1.91 MB
once the DNSMOS graph's own weights are counted), rather than the ~17 MB of
gradients and optimizer state that full-network training would need.

Weights `W` and `b` are **runtime graph inputs**, not baked-in initializers, so
the optimizer updates them in RAM with no regeneration or reflash. The cost:
ST maps `MatMul` to hardware only when the second operand is constant, so the
two head matmuls run as M55 software epochs (27.7 MMAC each, once per window).

## Verified, not asserted

`pytest tests/test_convfsenet_demo.py` (24 tests) gates every link:

| claim | gate |
|---|---|
| vendored architecture == upstream | **bit-exact** (`max\|diff\| = 0.0`) vs `ConvFSENetWindowedONNX` with shared weights |
| parameter count | exactly 1,443,136 = upstream's windowed drop_nyquist model |
| `dsp.stft` / `dsp.istft` == `torch.stft` / `torch.istft` | < 1e-9 |
| **ISTFT adjoint** == autograd | < 1e-9 (it is *not* the forward STFT — see `dsp.py`) |
| mask VJP, SI-SNR gradient == autograd | < 1e-10 / 1e-7 |
| head backward == autograd | < 1e-12 relative |
| **whole chain** (ISTFT-adjoint → mask VJP → head backward) == autograd through (head → mask → complex mul → ISTFT) | **< 1e-9 relative** |
| ONNX artifacts == torch modules | < 1e-5 / 1e-3 |
| head backward reduces over the batch | vs autograd at B = 1, 2, 3 |
| int8 trunk tracks fp32 | cosine > 0.95 |
| compression prologue stays fp32 | graph input reaches `Add`→`Pow` before any `QuantizeLinear` |
| `--checkpoint` loader | bit-exact vs upstream from a synthetic eco8 checkpoint |
| the loop itself (`OnDeviceEnhancer`) | runs, and OVRL rises in 8 steps |
| every artifact | passes the ST Edge AI front-end lint |

The two rows marked bit-exact need the upstream clone at `/workspace/eco8-neaixt`
and are skipped without it; the artifact rows need
`export_demo_artifacts.py` to have been run.

## Result

With a **randomly initialized frozen trunk** and the head initialized to
pass-through (`mask ≈ 0.982`, i.e. output = input), 60 steps of DNSMOS-driven
adaptation on one 9.01 s window:

```
noisy input        SIG 1.873  BAK 1.282  OVRL 1.329
step   0           OVRL 1.319   SI-SNR 44.3 dB   mask[0.98, 0.98]
step  20           OVRL 2.083   SI-SNR 20.3 dB   mask[0.35, 1.00]
step  59           OVRL 2.339   SI-SNR 15.9 dB   mask[0.00, 1.00]
enhanced (trained) SIG 3.517  BAK 2.156  OVRL 2.342   SI-SNR 15.9 dB
OVRL: 1.329 -> 2.342  (+1.013)
deployed int8 DNSMOS OVRL: 2.319
```

The head learns a frequency-selective gain from scratch, driven only by
gradients that came out of an inference graph.

**Is the SI-SNR anchor doing anything?** At the default `--sisnr-floor 6` the
hinge never engages in 60 steps — SI-SNR settles at 15.9 dB on its own, so
nothing needs constraining. Raising the floor shows the anchor is functional
rather than decorative:

```
--sisnr-floor 25 :  OVRL 1.329 -> ~2.2    SI-SNR pinned at 25-26 dB
--sisnr-floor  6 :  OVRL 1.329 -> 2.342   SI-SNR drifts to 15.9 dB
```

The hinge binds exactly when asked to and still permits most of the DNSMOS
gain. Set the floor to whatever distortion budget your deployment tolerates.

**Read this honestly.** The trunk is random here because the published
ConvFSENet weights are in a gated HuggingFace repo. This demonstrates that the
*mechanism* is correct and effective, not that it improves a well-trained
enhancer. With real weights:

```bash
python export_demo_artifacts.py --checkpoint /path/to/cp_convfsenet/g_best \
                                --eco8-repo /path/to/eco8-neaixt
```

The head then starts from the trained `backend` layer instead of pass-through,
and the same loop performs genuine on-device adaptation.

## STM32N6 accounting (`budget.py`, computed from the artifacts)

```
Persistent          ConvFSENet trunk int8 weights          1.45 MB
                    DNSMOS loss graph int8 weights         1.35 MB
                    head W,b + Adam moments                0.57 MB
                    (persistent NEW for training: 1.91 MB, incl. the DNSMOS weights)
Working buffers     STFT, mask, gradients (9.01 s window)  5.01 MB
Activation peak     DNSMOS fused fwd+bwd graph            70.75 MB   <- dominates
                    ConvFSENet trunk (emit_T=64 window)     0.16 MB
                    head fwd / bwd                          0.55 MB
Compute             trunk 1,094 MMAC + head 55 MMAC + DNSMOS 38,823 MMAC
                    = 39,972 MMAC per window; 17-121 s extrapolated from
                      eco8's measured 0.33-2.41 GMAC/s on target
```

**The full-size DNSMOS loss graph does not fit the STM32N6 — not even with the
DK's 32 MB PSRAM.** A single live activation is 70.75 MB: DNSMOS's first
convolution expands a 900×161 spectrogram to 128 channels, and the backward's
elementwise tail — `Equal → Cast → Sub → Mul`, none of which ORT quantizes by
default — carries that tensor in **fp32**, at 128 × 900 × 161 × 4 B. The int8
body is not the problem. Both halves of that are fixable and both matter: the
window sets the tensor's size, and `--elementwise` sets its dtype.

Shrinking DNSMOS is therefore a **requirement** for this target, not an
optimization — but *how* you shrink it decides the project.
`explore_feasibility.py` maps the frontier, `validate_student.py` distills a
student, `crop_study.py` tests the alternative and `adaptation_eval.py` puts
error bars on both; see **[FEASIBILITY.md](FEASIBILITY.md)**. The headline: a
distilled student small enough to fit reaches OVRL Spearman 0.84 against the
official model and still gets *exploited* by the optimizer, making the true
metric **worse on 77 % of clips** (−0.175 over 60 measured adaptations).
Cropping the window on the **official weights** instead costs nothing to build
and improves the true metric on **100 %** of them. Memory is not the hard part;
keeping the proxy honest is.

The deployable result is a **1 s window crop of the published weights, int8
QDQ with the backward's peak-setting elementwise ops quantized**: a 1.95 MB
activation peak that fits the 2.8 MB `n6-noextmem` pool, passing the STM32N6
lint, at +0.340 [+0.295, +0.386] true OVRL. Build it with
`python crop_study.py --window-s 1.0 --elementwise`.

The enhancer half is 7.57 MB — also above the 2.8 MB `n6-noextmem` pools, but
that is dominated by the 5.01 MB of 9.01 s working buffers, which scale down
linearly if you shorten the adaptation window.

### Why `emit_T=64`

Windowing recomputes the 42-column receptive-field context on every call, so
the cost is strongly non-linear in `emit_T` (measured with `budget.macs`):

| trunk configuration | MMAC per 9.01 s |
|---|---:|
| windowed, `emit_T=1` (upstream's Track 1 default) | 19,282 |
| windowed, `emit_T=8` | 3,117 |
| **windowed, `emit_T=64` (this demo)** | **1,094** |
| windowed, `emit_T=564` (whole segment at once) | 815 |
| streaming FIFO, `T=1` (what eco8 deploys today) | 829 |

`emit_T=64` is 17.6× cheaper than `emit_T=1` and within 1.32× of the stateful
streaming graph, while keeping Track 1's benefit — no per-frame
`Slice`/`Concat`/`Gather` state plumbing, which is ConvFSENet's M55 floor on the
Neural-ART. Larger `emit_T` is cheaper still but grows the activation window.

## Compiling for the chip

Following eco8-neaixt's `deploy/stm32n6/scripts/generate.sh` exactly — the
artifacts here are opset 17, `dynamo=False`, static except batch, QDQ int8 with
**signed** activations, per-channel weights, MinMax calibration, and the
magnitude-compression prologue excluded from quantization:

```bash
cd "$N6DIR"                                    # profile resolves ./my_mpools/*.mpool relatively
STEDGEAI=~/stedgeai/install/4.0/Utilities/linux/stedgeai

for m in convfsenet_trunk_int8 convfsenet_head_fwd convfsenet_head_bwd; do
  $STEDGEAI generate -m artifacts/$m.onnx --target stm32n6 \
    --st-neural-art n6-allmems-O3@user_neuralart.json \
    --fix-parametric-shapes "{'B':1}" --native-float \
    -n $m -o /tmp/gen_$m
done

# DNSMOS gradient graph (this repo's artifacts/)
$STEDGEAI generate -m ../../artifacts/dnsmos_loss_int8_qdq_stm32n6.onnx \
  --target stm32n6 --st-neural-art n6-allmems-O3@user_neuralart.json \
  --fix-parametric-shapes "{'B':1}" --native-float -n dnsmos_loss -o /tmp/gen_dnsmos
```

Note `--native-float` is **not** in upstream's `generate.sh`, and upstream
compiles the same partially-quantized ConvFSENet topology (fp32 Pow/Add
prologue + int8 body) without it. Try it without the flag first; add it only if
the compiler objects to the DNSMOS gradient graph, whose float tail is a much
larger fraction of the network. Read `network_generate_report.txt` for the epoch
count and HW/hybrid/SW split before writing any runtime code — that is
upstream's own de-risking pattern (`deploy/stm32n6/scripts/build_gate0d_probe.py`).

Runtime integration extends `deploy/stm32n6/app/ai_dpu_se_stream.c`: it already
does multi-I/O with `memcpy` feedback between sessions, which is the mechanism
the gradient buffers need. `--no-outputs-allocation` lets you place the 564 KiB
`dL/d(enhanced)` buffer yourself.

## Caveats

- **DNSMOS is a no-reference metric and will reward artefacts** if optimized
  without a constraint (EUSIPCO 2024, "Hallucination in Perceptual
  Metric-Driven Speech Enhancement"). On device there is no clean reference, so
  the demo hinges on SI-SNR against the *noisy input* — loose enough to permit
  real noise removal, tight enough to forbid muting or hallucinating. Tune
  `--sisnr-floor` for your deployment; do not remove it.
- Int8 gradients are noisier than fp32 ones by construction (see the main
  README's "int8 gradient reality"). The demo defaults to the fp32 DNSMOS loss
  graph; pass `--dnsmos-loss ../../artifacts/dnsmos_loss_int8_qdq.onnx` for the
  quantized one.
- One update per 9.01 s window is not real-time training; it is opportunistic
  background adaptation. The enhancement path itself is unaffected — the trunk
  and head forward are the same graphs the device already runs.
- Only `stedgeai validate --mode target` and `npu_profiler.py` give real
  latency and mapping numbers. Everything in `budget.py` is an extrapolation
  from artifact contents plus eco8's published on-target measurements.

## Files

| file | role |
|---|---|
| `convfsenet_arch.py` | vendored ConvFSENet (MIT), split into frozen trunk + trainable head |
| `dsp.py` | STFT / ISTFT / **ISTFT adjoint** / mask VJP / SI-SNR, numpy (the M55 half) |
| `head.py` | head forward + hand-written backward, both ONNX-exportable |
| `export_demo_artifacts.py` | builds the 4 graphs, quantizes the trunk, lints for STM32N6 |
| `ondevice_train.py` | the loop — `onnxruntime` + `numpy` only, asserts PyTorch is never imported |
| `budget.py` | memory / MAC accounting read out of the artifacts |
