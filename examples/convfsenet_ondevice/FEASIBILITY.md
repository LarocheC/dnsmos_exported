# How small must it be to train on-chip in real time?

Reproduce with `python explore_feasibility.py` and `python validate_student.py`.

**Short answer.** Memory and compute are the easy part — plenty of configurations
fit, and the enhancer is not the bottleneck. The binding constraint is something
the arithmetic does not show: a distilled DNSMOS proxy small enough to fit is
*exploited by the very optimizer it is meant to guide*, and makes the true
metric worse. Fixing that, not shrinking further, is the critical path.

---

## 1. The budget

One 9.01 s adaptation window, ConvFSENet int8 at eco8's measured 4.40 ms/frame:

| | |
|---|---:|
| wall-clock per window | 9.01 s |
| enhancement NPU time (564 frames × 4.40 ms) | 2.48 s |
| **left for adaptation** | **6.53 s** |
| → MAC budget at eco8's measured 0.33–2.41 GMAC/s | **2.2 – 15.7 GMAC** |
| current DNSMOS fused fwd+bwd | 38.8 GMAC → **2.5–18× too slow** |
| current activation peak | 70.8 MB vs 2.8 MB on-chip / 32 MB PSRAM |

## 2. The enhancer is not the constraint

Latency model for width-scaling the same topology, anchored on eco8's own split
of the 4.40 ms into 1.26 ms NPU core + ~3.14 ms M55 epoch/state plumbing (their
documented "N6 floor", which does not scale with width):

| ConvFSENet | MMAC/frame | ms/frame | RTF |
|---|---:|---:|---:|
| 192/384 ×9 (deployed) | 1.44 | 4.40 | 0.275 |
| 128/256 ×9 (published) | 0.66 | 3.72 | 0.233 |
| 64/128 ×9 | 0.18 | 3.30 | 0.206 |
| 32/64 ×6 | 0.04 | 3.18 | 0.199 |

**Shrinking the enhancer 36× in MACs buys ~7 % of the frame period.** Every
variant is already comfortably real-time; the floor is epoch-bound. So size the
enhancer for *quality*, not for the training budget — and if you want more NPU
headroom, attack the epoch count (Track 1 windowing) rather than the width.

## 3. How small must DNSMOS get?

Peak activation is `c1 × frames × bins`, so window length, bins and first-conv
width multiply. Selected rows (full sweep in `explore_feasibility.py`):

| window | frames | bins | c1 | params | peak int8 | loss GMAC | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 9.01 s | 900 | 161 | 128 | 287 k | 17.7 MB | 38.8 | PSRAM, too slow |
| 9.01 s | 900 | 64 | 32 | 53 k | 1.76 MB | 1.06 | **on-chip, fits** |
| 4.00 s | 399 | 64 | 32 | 53 k | 0.78 MB | 0.47 | on-chip, fits |
| 2.00 s | 199 | 64 | 32 | 53 k | 0.39 MB | 0.23 | on-chip, fits |
| 1.00 s | 99 | 64 | 32 | 53 k | 0.19 MB | 0.12 | on-chip, fits |

Window length is the strongest lever because it shrinks the activation peak,
the host working buffers **and** the MAC count simultaneously. A 9.01 s /
64-bin / quarter-width student is already 40× under the memory limit and 2–15×
under the time limit — the design space is not tight.

## 4. The result that actually matters

Two students distilled from the official model on real VoiceBank-DEMAND audio
(160 clips, 400 steps, held-out 40):

| student | peak int8 | OVRL Pearson | OVRL Spearman | BAK Pearson |
|---|---:|---:|---:|---:|
| 2.00 s, 64 bins, ¼ width | 0.39 MB | +0.799 | +0.822 | +0.884 |
| 9.01 s, 64 bins, ¼ width | 1.76 MB | **+0.858** | **+0.837** | +0.905 |

Good agreement. But agreement is not what a training signal needs. Adapting
ConvFSENet's mask head for 30 steps on one real clip, then scoring the result
with the **official** model:

| gradient source | true SIG | true BAK | true OVRL | ΔOVRL |
|---|---:|---:|---:|---:|
| (noisy input) | 3.927 | 2.838 | 2.923 | — |
| full official DNSMOS (70.8 MB) | 4.001 | 2.994 | 3.074 | **+0.151** |
| 9.01 s student (1.76 MB) | 3.707 | 2.690 | 2.745 | **−0.178** |

**The student makes the true metric worse.** The mechanism is unambiguous —
tracking both scores through the same optimization:

```
step   student OVRL   TRUE OVRL     gap
   0          2.478       2.925    -0.446
  10          2.549       2.885    -0.336
  20          2.644       2.859    -0.216
  30          2.744       2.745    -0.001
```

The student's own score rises monotonically while the truth falls monotonically.
Gradient descent is not following speech quality; it is walking into the
student's blind spots. Rank correlation of 0.84 says nothing about the local
gradient field, and 30 optimization steps is enough to find the gap.

This is exactly the failure MetricGAN exists to prevent, and eco8-neaixt already
has the machinery: `common/discriminator.py` retrains its PESQ predictor on the
generator's *current* outputs every step. A frozen, offline-distilled proxy has
no such defence.

## 5. The fix is not a better student — it is no student at all

The failure above is *not* the distribution shift MetricGAN addresses. Measuring
`cos(grad_student, grad_official)` on the very clips the student was trained on,
before any adaptation:

```
mean cosine +0.0002        (int8-vs-fp32 gradients of the SAME model score 0.41)
```

The gradient is **orthogonal** to the truth. Matching `f` does not constrain
`grad f`: score distillation transfers values, not derivatives. Adding an
explicit Sobolev term (`--sobolev 3.0`, cosine loss against cached teacher
gradients) moved it from 0.003 to 0.006 in 300 steps — matching a direction in
144,160 dimensions is hard, and near-chance is ~0.0026.

So stop distilling. **The official DNSMOS topology is fully convolutional up to
a *global* max pool, so the exact published weights accept any window length**
≥ 8 frames. Cropping the window costs nothing to build and keeps the gradient
field intact:

| window | peak int8 | GMAC | OVRL corr vs 9.01 s | grad cosine |
|---:|---:|---:|---:|---:|
| 9.01 s | 17.69 MB | 38.8 | 1.000 | 1.0000 |
| 4.00 s | 7.84 MB | 17.2 | 0.998 | **0.495** |
| **2.00 s** | **3.91 MB** | **8.6** | **0.910** | **0.452** |
| 1.00 s | 1.95 MB | 4.3 | 0.734 | 0.255 |
| 0.50 s | 0.96 MB | 2.1 | 0.193 | 0.113 |

Functionally, on the same clip and protocol as the student:

| gradient source | ΔTRUE OVRL |
|---|---:|
| distilled 9.01 s student (1.76 MB) | **−0.178** |
| full official, 9.01 s (17.7 MB int8) | +0.151 |
| **official weights, 2 s crop (3.91 MB int8)** | **+0.494** |

A 2 s crop of the real model is 4.5× cheaper than the full one, fits the compute
budget (8.6 of 2.2–15.7 GMAC), and steers *better* than either alternative.

## 6. Recommendation

1. **Do not distill a student.** Crop the window on the official weights
   instead. Recommended operating point: **2 s window, official weights, int8**
   — 3.91 MB activation peak, 8.6 GMAC, gradient cosine 0.45, and the best
   measured effect on the true metric of anything tested.
2. **MetricGAN is not needed for this problem, but is still the answer to the
   next one.** On-policy refresh fixes proxy *drift*; it does not create a
   gradient field that was never there. Once you are steering with real
   weights, drift and metric-hacking become the live risks — note that even the
   full official model drives the mask to 0.05 and SI-SNR downward. That is
   where co-training (`common/discriminator.py` in eco8-neaixt) earns its
   keep.
3. **Keep the SI-SNR anchor and make it tight.** It is what stops both the
   proxy and the true metric from being gamed; in the runs above the mask
   reaches 0.05 (near-total gating of some bins) before the anchor engages.
4. **Validate against the true metric, never the proxy.** `validate_student.py`
   step 6 is the only number that means anything.
5. **Shrink the enhancer for quality/power reasons only** — it is not what
   limits on-device training.

## Caveats

- The ConvFSENet trunk here is randomly initialized (the published weights are
  gated), so absolute deltas are not representative of adapting a trained
  enhancer. The *relative* comparison — full model +0.151 vs student −0.178
  under identical conditions — is the meaningful part, and the exploitation
  trace is independent of the trunk.
- Timing is MAC-derived from eco8's measured throughput span, not measured on
  target. Only `stedgeai validate --mode target` / `npu_profiler.py` settle it.
- Students were trained for 400 steps on 120 clips. More data would raise
  correlation; whether it removes the exploitation is exactly the open question,
  and correlation alone will not tell you.
