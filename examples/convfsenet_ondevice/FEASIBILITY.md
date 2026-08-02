# How small must it be to train on-chip in real time?

Reproduce with `python explore_feasibility.py`, `python validate_student.py`,
`python crop_study.py` and — for anything with error bars — `python
adaptation_eval.py`.

**Short answer.** Memory and compute are the easy part — plenty of configurations
fit, and the enhancer is not the bottleneck. The binding constraint is something
the arithmetic does not show: a distilled DNSMOS proxy small enough to fit is
*exploited by the very optimizer it is meant to guide*, and makes the true
metric worse. Fixing that, not shrinking further, is the critical path. The fix
is to stop shrinking the model and shrink the *window* instead: cropping the
official weights keeps the gradient field intact. Over 660 measured adaptations
(§5d, §5e), a **1 s crop, int8, elementwise-quantized** improves the true
metric on 100 % of clips by +0.340 [+0.295, +0.386] and fits the 2.8 MB
on-chip pool. The distilled student, on the same clips, is −0.175 and makes
77 % of them worse.

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

| window | peak int8 (analytic) | GMAC | OVRL corr vs 9.01 s | grad cosine |
|---:|---:|---:|---:|---:|
| 9.01 s | 17.69 MB | 38.8 | 1.000 | 1.0000 |
| 4.00 s | 7.84 MB | 17.2 | 0.998 | **0.495** |
| **2.00 s** | **3.91 MB** | **8.6** | **0.910** | **0.452** |
| 1.00 s | 1.95 MB | 4.3 | 0.734 | 0.255 |
| 0.50 s | 0.96 MB | 2.1 | 0.193 | 0.113 |

Functionally, on the same clip and protocol as the student (single-clip
figures — see §5d for the same comparison with error bars):

| gradient source | ΔTRUE OVRL |
|---|---:|
| distilled 9.01 s student (1.76 MB) | **−0.178** |
| full official, 9.01 s (17.7 MB int8) | +0.151 |
| **official weights, 2 s crop** | **+0.475** |

A 2 s crop of the real model is 4.5× cheaper than the full one, fits the compute
budget (8.6 of 2.2–15.7 GMAC), and steers *better* than either alternative.

## 5b. Does it survive int8? Yes — at 83 % efficiency

The crop above is fp32. Quantizing it (`crop_study.py`, QDQ int8, MinMax,
calibrated on 8 real VBD clips) with an exclusion ladder over the deepest
backward convs. ΔOVRL is the paired multi-clip number from §5d, not the
single-clip one this script prints:

| int8 variant | size | grad cos vs its own fp32 | ‖g₈‖/‖g₃₂‖ | ΔTRUE OVRL (n=60) |
|---|---:|---:|---:|---:|
| fp32 2 s crop | 1.99 MB | 1.000 | 1.00 | **+0.460** |
| int8, exclude 0 bwd convs | 0.98 MB | 0.440 | 0.51 | +0.355 |
| int8, exclude 2 | 1.20 MB | 0.499 | 0.54 | **+0.381** |

**Training with a quantized DNSMOS works**: the quantized graph retains 83 % of
the fp32 crop's true-metric gain, improves the true metric on 100 % of clips,
and beats the distilled student (−0.175) by 0.56 MOS.

Two things in that table are worth more than the headline:

- **Holding the two deepest backward convs in float is worth +0.027 MOS**
  (paired, 95 % CI [+0.011, +0.042]) for 0.22 MB. Cosine ranks the ladder the
  same way, so here the cheap proxy happens to agree with the metric — unlike
  §4, where cosine 0.0002 sat next to Spearman 0.84. Note that the single-clip
  run ranked these two rungs the *other* way; that ranking was noise, and it is
  why §5d exists.
- **The int8 gradient is not a function of its input.** Rerunning the same
  graph on the same audio across fresh sessions and thread counts, 5–50 % of
  executions diverge — the rate climbs with machine load, because thread
  scheduling is what sets reduction order — and when they do the gradient can
  be nearly unrelated to the previous one (worst observed cosine 0.036,
  ‖g′‖/‖g‖ 27×) while the *score* stays within 0.07 MOS. fp32 never does this:
  it differs only in the last ULP, cosine 1.000. The mechanism is tie routing —
  int8 collapses many activations onto the same code, so a max pool's argmax
  becomes a large tie set, and the backward decides membership with an `Equal`
  against the pooled value; one ULP of reduction-order difference moves
  elements in and out and redirects gradient mass wholesale. This is also why
  int8 cosine plateaus near 0.5 no matter how much of the graph is held float.
  Pin `--threads 1` for reproducible runs, and expect on-target gradients to
  differ from any desktop simulation by a comparable margin.

Adam over 30 steps absorbs the occasional corrupted step, which is why the
functional result survives. A first-order optimizer with no averaging would
not be a safe choice here.

Memory, measured on the shipped artifacts rather than analytically: all crop
variants peak at **15.64 MB** (`[1,128,199,161]`, conv1's output), because
`op_types_to_quantize` covers only Conv/Gemm/MatMul — the elementwise
mask/multiply ops of the backward keep that tensor in fp32. Quantizing those
too would take it to 3.91 MB.

**Neither number fits the 2.8 MB `n6-noextmem` pool**, so the 2 s crop is a
PSRAM part however it is quantized. Window length is the only free dimension
on official weights — the 161 bins and 128 first-conv channels are fixed by
the published tensors — so the on-chip frontier is set purely by frames:

| window | frames | peak int8 | fits 2.8 MB pool? |
|---:|---:|---:|---|
| 2.00 s | 199 | 3.91 MB | no |
| **1.00 s** | **99** | **1.95 MB** | **yes** |
| 0.50 s | 49 | 0.96 MB | yes |

So there are two deployable targets, not one: the 2 s crop in PSRAM, or the
1 s crop on-chip (§5c).

## 5c. The 1 s crop, and the protocol running out of resolution

Measured at 1 s, `crop_study.py` said the int8 build beat its own fp32 build by
17 % (+0.328 vs +0.281). Quantization cannot improve the gradient it
approximates, so that was a sign the **single-clip, single-seed, 30-step
protocol had a noise floor comparable to the effects it was ranking**. It did.
Everything in §5d supersedes the single-clip numbers; §5b's ladder ranking was
one of the casualties.

## 5d. With error bars

`adaptation_eval.py`: 7 gradient sources × 30 clips × 2 head initializations =
420 adaptations, every arm on the identical (clip, init) list.

| gradient source | mean ΔTRUE OVRL | sd | 95 % CI | clips improved |
|---|---:|---:|---:|---:|
| **fp32 2 s crop** | **+0.460** | 0.139 | [+0.426, +0.495] | 100 % |
| int8 2 s, exclude 2 | +0.381 | 0.141 | [+0.346, +0.418] | 100 % |
| int8 2 s, exclude 0 | +0.355 | 0.148 | [+0.318, +0.393] | 100 % |
| fp32 1 s crop | +0.446 | 0.231 | [+0.394, +0.507] | 100 % |
| int8 1 s, exclude 2 | +0.385 | 0.247 | [+0.329, +0.452] | 100 % |
| int8 1 s, exclude 0 | +0.347 | 0.178 | [+0.305, +0.393] | 100 % |
| distilled student | **−0.175** | 0.310 | [−0.257, −0.105] | **23 %** |

Paired against the fp32 2 s crop — same clips, same inits, so the clip-to-clip
spread cancels:

| | mean diff | 95 % CI | |
|---|---:|---:|---|
| fp32 1 s crop | −0.014 | [−0.067, +0.044] | **indistinguishable** |
| int8 2 s, exclude 2 | −0.079 | [−0.102, −0.055] | worse, resolvably |
| int8 1 s, exclude 2 | −0.075 | [−0.135, −0.009] | worse, resolvably |
| distilled student | −0.635 | [−0.718, −0.561] | worse, enormously |

What this settles:

- **int8 costs 0.08 MOS, and that cost is real** — the CI excludes zero, where
  the single clip could not tell. Retention is 83 % (2 s) and 86 % (1 s), not
  the 69 % the single clip suggested.
- **Halving the window costs nothing measurable.** −0.014 [−0.067, +0.044] in
  fp32, +0.004 [−0.055, +0.070] in int8. The 1 s crop is half the memory and
  half the MACs for no detectable quality loss — and it is the one that fits
  on-chip.
- **The student result was never noise.** −0.175, and it makes 77 % of clips
  *worse*; every real-weight arm improves 100 % of them. That was always the
  finding the project turned on, and 60 pairs confirm it at 0.56 MOS of
  separation.
- **The ladder ranking flipped.** With one clip, `exclude=0` looked best; with
  60 pairs, `exclude=2` is better by +0.027 [+0.011, +0.042]. §5b now reflects
  the paired number.

One confound is not settled and should not be read past. The official judge
needs 9.01 s, so a shorter crop is tiled up to it — 4.5× for the 2 s arms, 9×
for the 1 s arms. **Within-window comparisons are clean** (both arms hand the
judge an identically-built signal); the 1 s-vs-2 s rows confound window length
with tiling. Settling that means scoring the learned mask applied to the full
clip, which also tests whether adaptation on a short window generalizes off it.

## 5e. Getting it on-chip: quantizing the backward's elementwise tail

Everything above still needed PSRAM, because the activation peak was fp32.
`relu_mask` materializes `Equal → Cast → Sub → Mul` at conv1's full size, and
none of those are in ORT's QDQ registry — so the largest tensor in the graph
was carried at 4 bytes/element. That is not an ORT accounting artifact: those
tensors sit outside any Q/DQ region, so a QDQ-*fusing* backend materializes
them as float too.

ORT will quantize `Mul`/`Sub` if they are registered against `QDQOperatorBase`,
but doing it wholesale is a trap. Gradient norm on the 1 s crop, against 11.07
for the unquantized-elementwise baseline:

| registered | ‖g‖ | |
|---|---:|---|
| `+Sub` | 11.07 | harmless |
| `+Add` | 0.54 | severe degradation |
| **`+Mul`** | **0.00** | **destroys the gradient outright** |

`Mul` is simultaneously the op that matters for memory and the one that
silently zeroes the gradient. `large_elementwise_nodes` therefore quantizes
only the nodes that *set* the peak — output larger than `peak/4`, 6 of 45 here
— and leaves the small Muls carrying the surviving gradient in float, where
they cost nothing.

| 1 s crop, int8, exclude 2 | activation peak (fused) | ΔTRUE OVRL (n=60) |
|---|---:|---:|
| elementwise in float | 7.78 MB — **PSRAM** | +0.397 [+0.344, +0.458] |
| **elementwise quantized** | **1.95 MB — fits `n6-noextmem`** | **+0.340 [+0.295, +0.386]** |

The 4× memory win costs **−0.057 MOS** (paired, [−0.095, −0.023] — resolvable,
not noise). It is a trade, not a free lunch: you are buying the on-chip build
with about an eighth of the fp32 crop's headroom. The quantized artifact still
improves 100 % of clips and retains 76 % of the fp32 1 s crop, and the
rows-layout export passes the STM32N6 lint at 1.47 MB of weights.

Two accounting notes, because they are easy to quote wrongly:

- **ORT's peak stays 7.78 MB.** ORT CPU materializes the float tensors between
  Q/DQ pairs; a fusing backend does not. `budget.peak_activation` reports both,
  and only the fused number decides the on-chip fit. It is analytic —
  `stedgeai validate --mode target` is what would confirm it.
- Once both of a MatMul's inputs are quantized, ORT fuses `DQ+DQ+MatMul` into
  `QLinearMatMul` at session load and then rejects the per-channel weight
  zero-point. That is an ORT CPU kernel limit, not a defect in the exported
  graph, but it is why these artifacts use per-node exclusion.

## 6. Recommendation

1. **Ship the 1 s crop of the official weights, int8 QDQ, `exclude=2`.**
   4.3 GMAC, improves 100 % of clips, passes the STM32N6 lint. Pick the
   variant by which memory you have:
   - **on-chip** (`--elementwise`): 1.95 MB fused peak, fits the 2.8 MB
     `n6-noextmem` pool. **+0.340 [+0.295, +0.386]**, 1.47 MB of weights.
   - **PSRAM**: 7.78 MB peak, +0.397 [+0.344, +0.458], 1.21 MB of weights.
     Worth +0.057 MOS if you have the memory anyway.

   The 2 s crop is statistically indistinguishable from the 1 s one
   (−0.014 [−0.067, +0.044]) at 2× the memory and MACs, so take it only if the
   §5d tiling confound resolves in its favour.

   **Do not distill a student.** −0.175, worse on 77 % of clips, 0.5 MOS
   behind every crop variant.
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
4. **Validate against the true metric, never the proxy, and with error bars.**
   Correlation is actively misleading (§4: Spearman 0.84, cosine 0.0002, true
   effect negative). Gradient cosine happens to rank the int8 ladder correctly
   (§5b) but nothing guarantees that — it is the paired ΔOVRL that decided it.
   And a single clip cannot rank anything here: it got the ladder backwards and
   produced an int8 build that "beat" fp32 (§5c). `adaptation_eval.py` is the
   instrument; treat `crop_study.py`'s own ΔOVRL line as a smoke test.
5. **Shrink the enhancer for quality/power reasons only** — it is not what
   limits on-device training.
6. **Use an averaging optimizer with the int8 graph.** Its gradient is
   intermittently corrupted by tie routing (§5b); Adam over 30 steps absorbs
   that, plain SGD would not.

### What to do next, in order

1. **Confirm the memory on target.** `stedgeai validate --mode target` on
   `crop1s_int8_stm32n6.onnx`. The 1.95 MB peak is analytic and assumes the
   compiler fuses Q/DQ — true of Neural-ART, but the whole on-chip claim rests
   on it. The same run replaces the MAC-derived latency model, which the
   2.2–15.7 GMAC budget also rests on.
2. **Close the tiling confound** (§5d) by scoring the learned mask on the full
   9.01 s clip rather than a tiled crop. It decides 1 s vs 2 s properly and
   answers a better question — whether adapting on a short window generalizes
   off it.
3. **Recover some of the −0.057.** The elementwise cost (§5e) comes from
   quantizing 6 nodes with one global calibration; per-node range selection,
   or holding the mask path in float while quantizing only the gradient
   tensors, are both untried.
4. **Then co-training** (item 2 above). It answers metric-hacking, which only
   becomes the live risk once the steering signal is settled.

## Caveats

- The ConvFSENet trunk here is randomly initialized (the published weights are
  gated), so absolute deltas are not representative of adapting a trained
  enhancer. The *relative* comparisons are the meaningful part, and they are
  all paired against a common reference under identical conditions.
- Timing is MAC-derived from eco8's measured throughput span, not measured on
  target. Only `stedgeai validate --mode target` / `npu_profiler.py` settle it.
- §5d/§5e carry error bars; **§1–§5c do not** — their ΔOVRL figures are
  single-clip and, as §5c shows, at or below the noise floor. Where the two
  disagree, the paired numbers win. The 30-step budget and the lr=3e-3 Adam
  schedule are fixed across all arms but were never tuned; a different budget
  could reorder arms that §5d calls indistinguishable.
- Those CIs cover clip and initialization sampling, **not** execution
  nondeterminism. Re-running an identical int8 arm moved its mean by 0.012
  (+0.385 → +0.397) — small against a ±0.06 interval, and it does not reorder
  anything, but int8 arms carry that on top. Do not run the eval under
  concurrent load; §5b's divergence rate climbs with it.
- The adaptation loop runs the int8 ConvFSENet trunk, which carries the same
  intermittent nondeterminism as §5b. `crop_study.py` and `adaptation_eval.py`
  both pin `--threads 1`; the §4/§5 rows predate that pin and move by ~0.02
  OVRL without it.
- Students were trained for 400 steps on 120 clips. More data would raise
  correlation; whether it removes the exploitation is exactly the open question,
  and correlation alone will not tell you.
