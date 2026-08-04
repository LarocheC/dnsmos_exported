# First on-board run of the active-learning artifacts (STM32N6570-DK, 2026-08-03)

> Updated the same day with the **streaming variant** (per-frame FIFO trunk +
> banked 1 s DNSMOS windows) — see the section at the bottom.

First real `stedgeai` / on-target contact for this repo's artifacts, using
eco8-neaixt's measurement flow (`n6_loader` + `stedgeai validate` +
`ai_runner`), ST Edge AI Core 4.0.1, board @ MCU 800 MHz / NPU 1 GHz.
The repo's export-time lint (`verify.check_device_constraints`) had never been
cross-checked against an actual compile; it under-approximates. Every graph
needed patching (all patches parity-checked bit-exact in ORT before use), and
the fused loss graph ultimately hit a firmware-level hang.

## Measured

| artifact | build | on-target result |
|---|---|---|
| `convfsenet_trunk_int8` (real `cp_convfsenet/g_best` weights) | `n6-noextmem` | **31.75 ms** per emit_T=64 window = 0.50 ms per emitted 16 ms frame (RTF 0.031); X-cross cos **0.988**; 126.3 MMAC/window |
| DNSMOS **forward** 1 s crop, int8 rows (`crop1s_fwd_int8_rows`) | `n6-allmems-O3` (5.0 MB act → hyperRAM; noextmem alloc fails) | **528.2 ms**/window (σ 0.04 ms over 10; RTF 0.53); 35 epochs (14 HW / 2 hybrid / 19 SW) |
| DNSMOS **loss graph** 1 s crop int8 d2 + elementwise (the shipped adaptation artifact) | compiles under `n6-allmems-O3` *and* `-O1`: 204 epochs (33 HW / 46 hybrid / 125 SW), 1.53 MB weights (octoFlash), **23.3 MB activations** (20.6 MB hyperRAM) | **hangs on-device** — see blocker #1 |
| `convfsenet_head_fwd` | `n6-noextmem`, compiles clean | not measured (ST-LINK wedge ate the last slot); pure M55 SW matmul |
| `convfsenet_head_bwd` | — | **does not compile** — see blocker #2 |

Forward-scorer accuracy note: on-target `validate` vs the host int8 ONNX gives
mos cos 0.996 but a **systematic +0.23 mean bias** (mae 0.245 on 10 random
samples; a real VBD clip showed |Δmos| up to 0.46). ST's realization of the
int8 body + float frontend is not bit-compatible with ORT's QDQ semantics;
through the global-max routing the difference becomes a bias. Anyone using
on-device scores as an absolute metric must recalibrate against the deployed
artifact, not the host one.

## Blocker #1 — the fused loss graph hangs in a SW `DequantizeLinear`

Reproducible on both `-O3` and `-O1` builds: per-epoch streaming shows the
first 64 epochs (report IDs `epoch_1`..`epoch_64`, last one `Sub(float)`)
completing in **2.5–4.9 s total**, then the firmware never returns from
**`epoch_65` of 204 — a pure-SW `DequantizeLinear`** in the backward's first
`Equal/Cast/Sub/Mul` mask block (the peak-setting fp32 region, hyperRAM
resident). (Runtime callbacks are 0-based, report epoch IDs 1-based; quoted
here in report numbering.) Waited >10 min twice; identical stop point across five loads. The
graph itself is fine (identical outputs in ORT; first 63 epochs run on target),
so the blocker is the ll_aton SW kernel and/or its hyperRAM buffer placement.
Escalation path: ST bug report with `n6_gen/dnsmos_loss_crop1s_int8_d2` +
this reproduction; or eliminate the giant fp32 elementwise region (quantize the
mask chain itself, or crop ≤0.25 s so the fp32 tensors fit on-chip SRAM).

Also note: the "1.95 MB fits `n6-noextmem`" claim from `budget.py` does not
survive the real allocator — atonn reports 116 MB unallocatable on the on-chip
pools and needs 23.3 MB across pools even at int8+elementwise. `budget.py`'s
fused-QDQ peak is a lower bound, not an allocation.

## Blocker #2 — `head_bwd` breaks the ST front end

`INTERNAL ERROR: Mismatch in channel position` on the Neural-ART path *and*
the plain-M55 path (`--target stm32n6` without `--st-neural-art`). The graph
is 12 nodes (dynamic×dynamic MatMul + Transpose + ReduceSum/Mean, 2-D
outputs). Practical fix: hand-write the two matmuls in C (CMSIS-DSP), which
the README's runtime-integration section anticipates anyway.

## Graph patches required by ST Edge AI 4.0.1 (apply before `generate`)

All were needed on top of artifacts that already pass this repo's device lint;
each verified bit-exact vs the unpatched graph in ORT:

1. **Slice sentinels**: ends of `-1` / INT64_MAX → explicit in-range bounds
   ("Error in computation of shapes").
2. **Rank-1 graph input**: `w [3]` + in-graph `Reshape` to `[1,3]` → make the
   input `[1,3]` and drop the Reshape (same error).
3. **dynamo exporter artifacts** (flat/opset-20 graphs only): `allowzero=1`
   Reshapes with Shape-subgraph shape inputs → static shape initializers +
   dead-code elimination; `Shape(end=…)`/`Expand` (`ones_like`) → constants.
   The rows/opset-13 torchscript export avoids this class entirely.
4. **`Pad` ops**: atonn rejects the scalar (dims `[]`) `constant_value`
   initializer the front end itself re-emits ("missing shape or size for
   value=Pad_*_constant_value"). Replace every zero-pad with `Concat` of a
   zeros initializer; in quantized regions the zeros must be **pre-quantized**
   (int8 zeros at the sibling tensor's zero-point + shared-scale
   `DequantizeLinear`) or the compiler dies on a dtype "Conversion".
5. **`Clip` bounds**: ORT's quantizer rewires shared scalar constants through
   `DequantizeLinear`; ST needs Clip min/max to be plain constants → fold the
   dequantized value back to an initializer ("None has type NoneType…").
6. **Quant JSON**: the generated `*_Q.json` includes BOOL tensors (from
   `Equal`/`Cast`) which atonn's parser rejects wholesale, silently dropping
   ALL quantization ("invalid value \"BOOL\" for ContainerType" then
   "Ignore Quantization JSON file") → strip BOOL entries from the JSON between
   the front end and atonn (we used a temporary wrapper around the `atonn`
   binary; it has been removed — re-shim if you re-run `generate`/`validate`
   on the loss graph).

## Host/workflow gotchas (this laptop)

- `usbipd.exe attach --wsl --busid 1-1` after every board replug.
- The gdbserver port default 61234 collides with a VS Code listener on
  Windows (mirrored networking); `n6_utils_pkg/compilers.py` is patched to
  61245 in this stedgeai install.
- ST-LINK wedges (`DEV_USB_COMM_ERR` / "Loading memories failed") after a
  killed-mid-inference run **or sometimes after a validate run**: only a
  physical USB replug clears it. Don't kill runners mid-inference.
- `ai_runner` under numpy≥2 needs `np.fromstring` → `np.frombuffer` patched;
  the serial protocol's 50 s per-message timeout needs raising for slow graphs
  (`AiPbMsg._waiting_answer`).
- The compiler may transpose input layouts per-graph (the loss graph wanted
  `wav` as `[1,160,100]`; the fwd graph kept `[1,100,160]`) — read the
  device's I/O descriptors via `ai_runner.get_info()`, don't assume; this
  layout change is also why `stedgeai validate --mode target` errors on the
  loss graph (host-side feeder builds `(1,100,1,160)`).

## Reproduction pointers

- Patched deployable artifacts: `artifacts_crop/crop1s_int8_d2_rows_dev2.onnx`
  (loss, = shipped `crop1s_int8_d2` grids in rows layout + patches 1/2/4/5),
  `artifacts_crop/crop1s_fwd_int8_rows.onnx` (forward).
- Compiled outputs: `n6_gen/<name>/` (network.c + reports + xSPI2 blobs).
- Board flow: eco8-neaixt `deploy/stm32n6/ONBOARD_MEASUREMENT.md`, with
  `n6_loader.py --config config.json -nf <gen>/network.c -bc N6-DK`.

---

## Streaming variant (same day): enhance per frame, bank 1 s, adapt

`export_streaming_trunk.py` + `streaming_adapt.py`. Instead of the windowed
emit_T=64 trunk, the deployment-real per-frame FIFO trunk (eco8's form) runs
every 16 ms hop; the host banks the `(X, h, mask)` tensors the enhancement
path already computes (three memcpys — zero extra compute) and every 63
frames (~1.01 s) runs the DNSMOS 1 s crop loss → ISTFT-adjoint → mask VJP →
`head_bwd` (T=63) → Adam. Trunk FIFO states flow through window boundaries:
no per-window cold start, and the windowed demo's 1,094 MMAC/9 s of trunk
recompute disappears.

**Artifacts** (in `artifacts/`): `convfsenet_trunk_stream_{fp32,int8}.onnx`
(per-frame, 9 state tensors as I/O; int8 calibrated with *threaded* per-frame
state à la eco8's ConvFSENetCalibrationReader), `stream_head/` (head graphs at
T=63). Parity gates: steady-state streaming == offline 1.1e-05; int8 h cos
0.998 over 200 threaded frames (host ORT). NOTE the warm-up frames (t<42)
legitimately differ from the *windowed* trunk: the causal model (== the FIFO)
zero-pads each block's dconv input in the y-domain, while the windowed host
ring buffer pads magnitudes.

**On-target**: compiles for `n6-noextmem` (68 epochs: 29 HW / 27 hybrid /
12 SW, 1.63 MMAC/frame), runs at **4.236 ms/frame (RTF 0.265)** — matching
eco8's full streaming graph at 4.40 ms (backend conv removed). `validate`
X-cross cos 0.843 under random-state feeds (the same harsh-caveat eco8 hit:
random FIFO states are far outside any calibrated distribution).

**Host-threaded on-device states are NOT a valid fidelity probe.** Feeding
states over the serial validation protocol and reading them back diverges
from the compiled graph specifically when state buffers sit at the int8
zero-point (cos 0.73 at constant -128 vs 0.95-0.97 at random/0 constants,
same real frame; the OE graph in ORT matches the deployable artifact at
0.998 everywhere). In deployment states never cross the serial link — they
persist in the aton runtime's in-place buffers — so the deployment-relevant
fidelity number is the threaded-state host measurement (0.998). Root cause
of the -128 anomaly (ai_runner packing vs in-place aliasing in the
validation firmware) is an open question; reproduction is scripted in the
session notes.

**Adaptation results** (real trained trunk+head from `cp_convfsenet/g_best`,
official 9.01 s DNSMOS raw OVRL, single streaming pass = 8 updates):

| clip | noisy | deployed baseline | adapted | d |
|---|---|---|---|---|
| 159 | 2.923 | 2.320 | 2.565 | **+0.245** |
| 100 | 3.377 | 1.476 | 1.526 | +0.050 |
| 120 | 3.971 | 1.177 | 1.778 | **+0.600** |
| 140 | 1.594 | 1.386 | 1.341 | -0.045 |

Two findings:

1. ~~**The trained ConvFSENet scores far below noisy on official DNSMOS**~~ —
   **RETRACTED, see the CORRECTION section at the end of this file.** This was
   a missing RMS normalization in the harness, not a property of the model.
   With the fix the enhancer scores PESQ 3.43 vs noisy 2.48. The gains in the
   table above are therefore measured from a crippled baseline.
2. **Multi-pass replay is live metric hacking**: over 4 passes on the same
   clip the on-device crop proxy rises monotonically (1.57 -> 1.74) while
   the official score falls (2.565 -> 2.339). The proxy cannot early-stop
   itself. Single-pass over fresh audio — which is what a streaming
   deployment does anyway — is the honest default (`--passes 1`).

**On-board budget for the full streaming loop** (all measured): enhancement
4.24 ms/frame within the 16 ms hop (RTF 0.265, 73 % idle); per 1 s window
the DNSMOS forward costs 528 ms (RTF 0.53 of the banked second) — both fit
opportunistic background adaptation. The missing on-target piece remains the
fused loss graph (epoch-64 hang, above) and `head_bwd` (front-end crash,
above); on-device gradients still need those resolved or hand-written C.

---

## Unblocking round (same day, later): both blockers attacked

### head_bwd: FIXED — compiles and measured

Op-by-op bisection isolated the `INTERNAL ERROR: Mismatch in channel position`
to the in-graph `Transpose` of a runtime tensor. Everything else — the
elementwise chain, dynamic x dynamic MatMul, the rank-2 reducing sum, the
ReduceMean db path — compiles fine in isolation. Fix: `HeadBackwardHT`
(head.py) takes `hT [B, T, 192]` pre-transposed as an input; banking h in
transposed layout is the same per-frame memcpys with different strides, so the
change is free on device. Bit-exact parity with `HeadBackward`.

**On-target: 34.2 ms per update** (T=63 window, `n6-noextmem`, 10 epochs:
2 HW / 8 SW) — negligible against a once-per-second update cadence.
Artifact: `artifacts/stream_head/convfsenet_head_bwd_ht.onnx`, compile in
`n6_gen/head_bwd_ht/`.

### Loss graph: hang FIXED by the 0.25 s crop; a numerics fault remains

`crop_study.py --window-s 0.25 --elementwise` produces a *better* proxy than
the 1 s crop on every host axis: int8 grad cos 0.71 (vs 0.39), adaptation
dOVRL +0.518 at exclude=2 (163 % of its fp32 crop), fp32 elementwise peak
1.89 MB. `patch_for_stedgeai.py` (new, consolidates all graph patches +
parity gates) produces `artifacts_crop/crop0p25s_int8_d2_rows_dev.onnx`.

**The 1 s epoch-64 hang does not occur at 0.25 s**: all 205 epochs execute,
**4.80 s device time per window** (activations 6.7 MB: 2.8 on-chip + 3.97
hyperRAM). The hang is therefore size/placement-dependent, strengthening the
ll_aton-kernel diagnosis for the 1 s graph.

**But the on-device numerics are wrong, and it is an ST runtime/codegen
fault, not a graph fault.** Evidence chain:

- ST's own optimized-export (`*_OE_*.onnx`) run in ORT matches the deployable
  artifact: mos within 0.03, grad cos 0.80 — with the input in the
  compiler's transposed `(1,160,25)` layout. The compile is semantically
  sound.
- The device returns near-constant outputs — mos `[1.909 1.448 1.143]`
  byte-identical across different clips and across FOUR builds (`-O3`, `-O1`,
  no `--Ocache-opt`/`--cache-maintenance`, `--Opreserve-inputs`); the
  gradient output overflows (norm 100-300x host). The input barely
  influences the result; both plausible input byte-orders were tested.
- The `-O3`/`-O1` builds alias the `wav` input and `grad_wav` output at the
  same address (0x342e0000, from the on-device I/O descriptors);
  `--Opreserve-inputs` separates them and changes nothing, ruling the
  aliasing out as the root cause.
- The forward-only network (untransposed input layout) runs with only a
  +0.23 mos bias — the corruption is specific to the multi-output
  loss-graph builds.

Escalation package for ST: `n6_gen/dnsmos_loss_crop0p25s_*` (4 builds),
the OE-vs-device comparison above, and the 1 s hang reproduction. Local
next step if ST stalls: eco8's `firmware_router` app path instead of the
`NPU_Validation` stack, in case the fault is validation-firmware-specific.

Toolchain state after this round: real `atonn` binary restored; the BOOL-strip
shim is preserved at `atonn_shim.sh` (install by moving `atonn` ->
`atonn.real` and dropping the shim in its place); `user_neuralart.json` gained
`n6-dbg-nocache` / `n6-dbg-preserve` debug profiles.

### The numerics fault is NOT the validation stack — bare-runtime confirmation

Built a standalone firmware (`firmware_loss/`, modelled on eco8's
`firmware_router` project-patching + UART flow) that drives the SAME compiled
network through the **bare LL_ATON runtime**: input baked into flash as a const
array, no protobuf, no ai_runner, no serial feeding. Verdict:

```
-- clip 0: fed wav[0..3] = 0xbc620000 0x3b950000 0x3b800000 0x3cd94000   (correct)
   MOS  SIG=0x3ff465d9  BAK=0x3fb959d2  OVRL=0x3f925cd8   (5329 ms)
   GRAD n=4000 nonfinite=176   grad[0..7] = 0x41a472f0 0x808080f2 0x808080f2 ...
-- clip 1: fed wav[0..3] = 0xbd148000 0xbd730000 0x3bcd0000 0x3ce1c000   (different!)
   MOS  SIG=0x3ff465d9  BAK=0x3fb959d2  OVRL=0x3f925cd8   (IDENTICAL BITS)
   GRAD n=4000 nonfinite=491   grad[0..7] = ... 0x7fc00000 0x7fc00000 (NaN)
```

- The scores are **bit-identical across two different clips**, and identical to
  what the validation stack reported (1.9094 / 1.4481 / 1.1435). Two completely
  independent host→device paths, same wrong constant ⇒ **the validation stack is
  exonerated**; the fault is in the generated code or the NPU runtime.
- The fed input is verifiably correct at the buffer: printed back from the
  network's own input buffer, and separately dumped over SWD after a run
  (bit-identical to what the host sent).
- The constant is **not** what a degenerate input produces: feeding zeros /
  all-0x80 / tiny constants to the same graph in ORT gives 2.395/2.743/1.756,
  not 1.909/1.448/1.143. So it is not simply "input never arrives" — the device
  is computing something else entirely.
- The gradient contains **`0x808080f2`-style words and NaNs**: 0x80 is the int8
  **zero-point** byte. Raw int8 tensor bytes are being read as float32 — i.e. a
  dequantize/type boundary is mis-wired in the SW-epoch path. That is the same
  region that *hangs* at the 1 s crop (epoch 64 = a SW `DequantizeLinear`), so
  both symptoms are one bug: **ST's int8↔float SW-epoch boundary in this graph**.
- Only 2 of 277 OE tensors are legitimately input-independent in ORT (two
  constant equality masks), so the device's constant output cannot be explained
  by graph structure.

Bug-report package for ST: `firmware_loss/` (bare-runtime reproduction),
`n6_gen/dnsmos_loss_crop0p25s_*` (four builds), the OE-in-ORT reference, and
the 1 s hang. This is now a clean vendor-side defect with a minimal reproducer
and no remaining host-side suspects.

`firmware_loss/README.md` documents the two gotchas that cost time: this
project does not link newlib float printf (`%f` silently wedges the UART
mid-line — print IEEE-754 bits instead), and a bare runner must call
`LL_ATON_RT_RuntimeInit()` or the first `RunEpochBlock` parks in
`LL_ATON_OSAL_WFE()` forever.

---

## Could the loss graph run fully in int8? (`int8_coverage_study.py`)

Motivation: the ST defect lives at the int8↔float boundary of *software*
epochs, and a graph without such a boundary cannot trip it — plus every tensor
moved to int8 is 4× less memory on the part where the fp32 tail sets the peak.

**Short answer: not strictly, but the irreducible part is tiny — and the
blocker for the rest is structural, not a quantizer setting.**

### Where the float traffic actually is (0.25 s graph, shipped recipe)

Counting only tensors that truly materialize as fp32 on device (a node whose
every consumer is a `QuantizeLinear` gets folded into an int8 kernel):
**22.2 MB across 184 nodes.**

| op | fp32 MB | share | can it be int8? |
|---|---:|---:|---|
| `Concat` | 5.64 | 25 % | yes — pure data movement |
| `Equal` | 5.05 | 23 % | yes — int8 equality is **exact** when both sides share a scale |
| `Reshape` + `Unsqueeze` | 5.54 | 25 % | yes — pure data movement |
| `Mul` | 1.57 | 7 % | partly — quantizing *every* `Mul` zeroes the gradient |
| `Cast` | 1.28 | 6 % | yes — bool→int8 instead of bool→float |
| `Sub` / `Clip` / `Relu` | 1.89 | 9 % | yes — standard QDQ ops |
| **`Reciprocal` + `Log`** | **0.65** | **3 %** | **no — division / transcendental; needs a LUT or fixed-point rewrite** |

So ~75 % of the float traffic does **no arithmetic at all** — it is data
movement and comparisons — and only 3 % is genuinely irreducible.

### Why the quantizer cannot reach it

ORT inserts QDQ only where a quantized tensor already flows. The backward's
mask chain is rooted at `Equal`, which consumes `DequantizeLinear` *outputs*
(float) and emits **bool** — a type with no quantization grid. Everything
downstream (`Cast` → `Sub` → `Mul`, plus the `Concat`/`Reshape` interleave
upsampling) is therefore float-rooted and unreachable, no matter which op types
are registered.

Closing that gap needs a **graph rewrite**, not a setting: have `Equal` compare
the raw int8 tensors directly (exact integer comparison — arguably better than
the fp32 one it replaces) and emit an int8 0/1 mask, so the chain stays int8
end to end. The repo's `repair_mask_consistency` already does half of this
work, rebuilding the max-side chain rooted at the exact tensor instance.

### What is achievable today, measured

Registering the data-movement and standard pooling ops
(`Reshape`/`Unsqueeze`/`Squeeze`/`Slice`/`Transpose` + `MaxPool`/`AveragePool`/
`Relu`/`Clip`) does compile and lint clean, on-chip pools included:

| recipe | SW epochs | grad cos (mean/min) | \|g8\|/\|g32\| | adaptation dOVRL |
|---|---:|---:|---:|---:|
| shipped | 125 | 0.658 / 0.513 | 0.71 | **+0.518** |
| +movement+pool | **119** | **0.781 / 0.580** | **0.99** | +0.377 |

The agreement metrics improve markedly — the gradient norm ratio going from
0.71 to ~1.0 means the int8 gradient no longer systematically shrinks — **but
the functional adaptation on the one test clip is worse**. That is this repo's
recurring lesson (`crop_study.py`: "cosine does not rank steering"), and a
single clip is exactly the kind of evidence an earlier commit had to retract as
noise. **Ranking these two recipes needs the paired multi-clip protocol in
`adaptation_eval.py`; it is currently unresolved.**

### Would it dodge the ST defect?

Only the full rewrite would meaningfully shrink the failing region, and even
then `Log`/`Reciprocal` keep a small int8↔float boundary. So this is a
worthwhile memory and (possibly) quality optimization, **not a reliable
workaround** for the vendor bug.

---

## Feeding DNSMOS from the enhancer's spectrogram (`spectral_adapter.py`)

Idea (user's): the enhancer already computes an STFT, and DNSMOS computes
another one internally. Drop the second, and the ISTFT between them.

It is better founded than it first appears, because the enhancer applies a
**real gain**: the enhanced magnitude is exactly `|Y| = |X| · mask` in the
enhancer's own domain, so no synthesis is needed to obtain it, and the weight
gradient collapses to

    dL/dmask = dL/d|Y| · |X|          (one elementwise multiply)

replacing the ISTFT-adjoint and the complex mask VJP. Two facts checked before
building anything:

* **Consistency**: `|STFT(ISTFT(X·mask))|` vs `|X·mask|` — cosine **0.9999**,
  1.3 % relative error, for both smooth and hard-gate masks. A magnitude-domain
  metric sees essentially what DNSMOS sees.
* **Phase**: normally the fatal objection to a magnitude-only metric. Not here —
  a real gain *cannot* alter phase, so nothing is lost **for this loop**. It
  would be lost for a phase-modifying enhancer.

Measured savings: **46× less host DSP per adaptation window** (1.763 ms →
0.038 ms), `istft_adjoint` — the hardest-verified function in `dsp.py` — leaves
the backward entirely, **59 of 329 graph nodes (18 %)** disappear, and 2 of the
6 irreducible-float ops go with them (the `Log` and one `Reciprocal` live in
the STFT frontend/adjoint).

### Two surprises from the experiment

**1. A plain bilinear resample already carries most of the signal.** With no
training at all, feeding the frozen official body a bilinear-resampled
enhancer spectrogram gives, on 40 held-out clips:

| | pearson | spearman | mean abs err |
|---|---:|---:|---:|
| SIG | 0.719 | 0.498 | 0.238 |
| BAK | **0.827** | **0.860** | 0.383 |
| OVRL | **0.802** | 0.753 | 0.272 |

(An earlier estimate of 0.27–0.57 in this session was **wrong** — it used
nearest-neighbour time indexing and a stray ×10 on the log scale. Corrected
here.)

**2. Regressing the feature map is actively harmful.** Training the adapter to
match the official `Frontend(wav)` output — ~145k densely supervised values per
clip instead of 3 — cuts eval feature MSE **6.4× (6.04 → 0.94)** while dropping
BAK rank correlation from **0.860 to 0.154** and OVRL from 0.753 to 0.510. The
body reads a **global max** over the spectrogram; MSE smooths exactly the peaks
that max selects, so being closer on average is worse where it counts.

That is the third independent instance in this repo of the same lesson — an
agreement metric that improves while the thing you care about degrades (after
`crop_study.py`'s "cosine does not rank steering" and the int8-coverage
recipe). The default objective is therefore `--loss score`: the frozen body is
differentiable, so the 5.7k-parameter adapter is regressed on the three mapped
scores it must preserve, with the metric itself frozen and bit-exact so the
learned part cannot reshape what "quality" means.

### Score-space training, and a clean anti-correlation

Regressing the three mapped scores through the frozen body instead
(`--loss score`, 5,665 trainable parameters, 30 epochs) gives the mirror image
of the feature-MSE result:

| variant | eval feature MSE | OVRL pearson | OVRL spearman | BAK spearman |
|---|---:|---:|---:|---:|
| bilinear, no training | 6.04 | 0.802 | 0.753 | 0.860 |
| trained on **features** | **0.94** (6.4x better) | 0.654 | 0.510 | **0.154** |
| trained on **scores** | 8.34 (1.4x worse) | **0.926** | **0.887** | **0.929** |

The two objectives are close to anti-correlated: whichever one is optimized,
the other degrades. Training longer overfits — at 100 epochs the training loss
keeps falling (0.054 -> 0.027) while held-out OVRL correlation drops 0.926 ->
0.863 and SIG collapses 0.824 -> 0.483, so the 30-epoch checkpoint is the one
to use with this 112-clip training set.

### Verdict: the correlation is real, the steering is not — and the adapter is exploited

`adapter_adapt_eval.py`, 8 held-out clips, 40 steps, paired, scored with the
official 9.01 s DNSMOS:

| | mean dOVRL | better on |
|---|---:|---|
| waveform path (today) | **+1.536** | 8/8 clips |
| spectral path (adapter) | +0.589 | 0/8 clips |

Paired difference **-0.948 [95% CI ±0.522]** — the interval excludes zero, so
this is *resolved*, not a sample-size problem. The adapter delivers 38 % of the
gain, and on 2 of 8 clips essentially none (+0.036, -0.074).

**Why**: the optimizer exploits it. Re-scoring the adapter's own optimized
output against the truth:

| clip | adapter believes | official truth | optimism |
|---|---:|---:|---:|
| 0 | 4.213 | 2.550 | **+1.663** |
| 1 | 4.272 | 1.802 | **+2.470** |
| 2 | 4.492 | 1.160 | **+3.331** |
| 3 | 4.167 | 2.689 | **+1.478** |

**mean optimism +2.24 MOS.** The same adapter agrees with truth at Spearman
0.89 on *natural* clips — and is off by more than two MOS on the adversarial
spectrograms the optimizer itself steers into. Rank correlation on the natural
distribution says nothing about behaviour off it, which is exactly the
distribution a gradient source is dragged into.

This reproduces `FEASIBILITY.md`'s student result — Spearman 0.84, true metric
worse on 77 % of clips — with a completely different and much more constrained
surrogate: 5,665 trainable parameters in front of a frozen, bit-exact official
body. That the failure survives *this* much constraint is the finding. The
metric's own trained STFT front end is apparently not a detachable
implementation detail; it is part of what makes DNSMOS hard to fool.

### Where that leaves the idea

The architecture argument is untouched and still attractive — 46x less DSP, no
`istft_adjoint`, 18 % fewer nodes, 2 of 6 irreducible-float ops gone. What is
refuted is the *cheap* route to it (resample + small learned correction). Paths
that remain open, in increasing cost:

1. **Adversarial / iterated training** — retrain the adapter on the
   spectrograms the optimizer actually produces, not just natural ones, and
   iterate. Directly targets the measured failure.
2. **Port the trained STFT instead of approximating it.** DNSMOS's kernels are
   fixed matrices; a 512/256 -> 320/160 re-analysis could be done exactly in
   the spectral domain rather than learned, keeping the metric bit-exact and
   still avoiding the ISTFT round trip.
3. **Keep the waveform path** and take the memory win elsewhere.

Option 2 is the one that preserves what makes the metric robust, and is the
natural next experiment.

---

## CORRECTION: a normalization bug invalidated several adaptation numbers above

ConvFSENet is trained on **RMS-normalized** input (`eco8-neaixt/convfsenet/
inference_onnx.py`: `norm_factor = sqrt(len(x) / sum(x**2))`, un-scaled after
the iSTFT). Its first operation is a power-law compression `(|X|+1e-9)**0.3`,
so the input *level* sets the range every downstream conv sees. Feeding raw
audio does not degrade the mask gracefully — it destroys it:

| VBD test segments (6 s) | PESQ |
|---|---:|
| noisy | 2.482 |
| enhanced, **no** RMS norm (the bug) | 1.097 |
| enhanced, **with** RMS norm | **3.430** |

Confirmed against upstream's own deployed `g_best.onnx`, which scores the same
1.058 when fed unnormalized — so the vendored split was always faithful; the
harness was wrong. Fixed via `dsp.rms_normalize`, applied in
`streaming_adapt.py` and `pesq_adapt_eval.py`.

**What this invalidates, above:**

* The claim that "the trained ConvFSENet scores far below noisy on official
  DNSMOS" — **wrong**. That was this bug, not a metric disagreement. The mask
  statistics check that seemed to confirm it (mean 0.142 vs upstream 0.144) was
  too weak: per-element the masks differed by up to 0.42.
* The streaming single-pass adaptation gains (+0.245 / +0.050 / +0.600 /
  -0.045) — measured from a crippled baseline, so they largely reflect
  recovery from the bug rather than genuine adaptation.
* The DNSMOS waveform-vs-spectral magnitudes (+1.536 / +0.589). That
  comparison was *paired* — both arms shared the broken enhancer — so the
  ranking (spectral worse, 0/8) plausibly survives, but the numbers do not.

The int8/graph/on-target findings (latencies, epoch counts, the ST defect, the
compile patches) are unaffected: none of them depend on the enhancer's audio
quality.

## The PESQ predictor: honest architecture, still exploited

`pesq_predictor.py` / `pesq_adapt_eval.py`. Trained on 2,800 PESQ-labelled
candidates from VBD train (8 deliberately varied masks per utterance: ideal
ratio, under/over-suppression, spectral holes, band damage, random smooth),
eco8's 181,650-parameter `MetricDiscriminator` architecture — 8x smaller than
the DNSMOS body, `spectral_norm` throughout (Lipschitz-bounded), reading
`(noisy_mag, enhanced_mag)` on the enhancer's own compressed magnitude.

**As a predictor it is excellent**: held-out pearson **0.964**, spearman
**0.961**, MAE **0.18 PESQ** — better than the DNSMOS adapter on a harder,
broader distribution, with no overfitting at 40 epochs.

**As a gradient source it fails, harder than the DNSMOS adapter.** On VBD
*test* utterances with real clean references and the normalization fixed:

| | value |
|---|---:|
| true PESQ gain | **-0.722 [95% CI ±0.128]**, improved on **0/7** |
| optimism (believed − true) | **+5.33 PESQ** (DNSMOS adapter: +2.24 MOS) |

The predictor reports **7.8–7.9** where PESQ's maximum is 4.5 — the optimizer
drove it past the top of its own output range, i.e. clean off the distribution
it was fit on, while true quality fell from ~3.5 to ~2.6.

### What this establishes

Every structural defence we could apply — an intrusive training target, the
noisy signal as a second input so the metric judges a *relationship*, a
Lipschitz bound by construction, and training data that deliberately includes
the optimizer's favourite artefacts — was **not sufficient**. Learned metrics
with 0.96 correlation on natural data are still exploitable off it, and the
gap between correlation and steering keeps widening the harder we try:

| surrogate | correlation | optimism under optimization |
|---|---:|---:|
| DNSMOS distilled student (FEASIBILITY.md) | spearman 0.84 | true metric worse on 77 % of clips |
| DNSMOS spectral adapter | spearman 0.89 | +2.24 MOS |
| PESQ predictor (this) | spearman **0.96** | **+5.33 PESQ** |

Correlation on natural data is not merely insufficient — across these three it
is **anti**-correlated with steering safety, because a better fit on the
natural manifold buys nothing off it.

The practical consequence for this project: **use the real metric as the
gradient source, not a learned stand-in.** The full DNSMOS loss graph — with
its own trained STFT, quantized and cropped — remains the only source in this
repo that has demonstrably improved the true metric. That is an argument for
finishing the on-device DNSMOS path (i.e. the ST defect) rather than replacing
it.

---

## Deploying the PESQ predictor: the ST defect is avoided at compile time

Rationale (user's): full ownership of the metric matters for deployment — a
181,650-parameter model trained in-house beats a transplanted third-party one
— and the predictor's *architecture* is a direct test of the ST diagnosis. The
DNSMOS loss graph fails inside ST's int8<->float **software**-epoch boundary,
which exists because its hand-written backward carries a large fp32 elementwise
mask region between int8 conv blocks. This model has no such region.

`export_pesq_predictor.py`. Two export details mattered:

* **`spectral_norm` uses the old hook API**, so `.eval()` does not disable it
  and `remove_parametrizations` does not see it. Exported live, the power
  iteration itself is traced into the graph — 12 `MatMul` + 6 `Div` of pure
  normalization arithmetic. `remove_spectral_norm` folds it once and the graph
  drops from those 18 stray ops (940 kB int8) to none (**200 kB**).
* `Flatten` -> `Reshape`; after a global max-pool to `[B,C,1,1]` they are identical.

Result: torch-vs-ONNX parity **1.8e-07**, int8 costs **0.020 PESQ** mean
(max 0.041), and it **compiles for the on-chip `n6-noextmem` profile**:

| | epochs | HW / hybrid / SW | weights | MACC |
|---|---:|---|---:|---:|
| DNSMOS loss graph (0.25 s) | 201 | 35 / 41 / **125** | 2.24 MB | 1,183 M |
| **PESQ predictor** | **35** | **12 / 0 / 23** | 727 kB | 26 M |

Zero hybrid epochs, 45x fewer MACs, and it fits the on-chip pools that the
DNSMOS loss graph could never fit. The repo's lint flags
`InstanceNormalization` as outside its op vocabulary — that list is an
approximation, and the compiler accepted it, consistent with everything else
this session found about lint-vs-compiler.

### On-target: it runs correctly — the defect is confirmed localized

Loaded to `n6-noextmem` and fed six candidates spanning the PESQ range:

| candidate | true PESQ | host int8 | **device** | \|d\| |
|---|---:|---:|---:|---:|
| 3 | 3.19 | 3.27 | **3.27** | 0.000 |
| 17 | 4.14 | 3.49 | **3.52** | 0.028 |
| 42 | 2.14 | 3.19 | **3.19** | 0.000 |
| 88 | 1.10 | 1.10 | **1.11** | 0.005 |
| 123 | 3.79 | 3.66 | **3.66** | 0.000 |
| 260 | 3.49 | 2.82 | **2.82** | 0.000 |

* **device vs host int8: mean 0.0056 PESQ, max 0.028** — the device reproduces
  the host artifact.
* **Output spread 1.11–3.66.** The DNSMOS loss graph's failure signature is a
  bit-identical constant across different inputs; here the output tracks the
  input, correlating +0.84 with true PESQ over these six.
* **37.6 ms per inference**, entirely on-chip.

Against the DNSMOS loss graph on the same board and toolchain: **4,800 ms and
input-independent garbage** versus **37.6 ms and correct**. That is a 128x
speedup and the difference between unusable and deployable — and it is direct
evidence for the bug report's diagnosis, since the only structural difference
that matters is the absence of the fp32 elementwise mask region that creates
the int8<->float software-epoch boundary.

**A metric the project owns, running correctly on the target.** What remains
for a full on-device adaptation loop is the predictor's *backward* pass, which
would be built the same way as the DNSMOS one (hand-written ONNX ops) but over
a far simpler graph — no `Equal`/`Cast` mask chains, so no fp32 elementwise
region, so no exposure to the ST defect.

Note on the negative PESQ steering result above: the user's point that
ConvFSENet is a PESQ metric-GAN trained on this very dataset is well taken and
partly explains it — baseline PESQ was already 3.3-3.9, so there was little to
climb and the optimizer could only descend. That does not explain the +5.33
optimism (the predictor reporting 7.8 where PESQ maxes at 4.5), which is
off-distribution behaviour independent of headroom. A fairer steering test
would start from a *weaker* enhancer, where real headroom exists.

## The PESQ backward: written, verified, and it isolates the ST defect further

`pesq_backward.py` — a fused `(noisy_mag, enh_mag) -> (score, dScore/d(enh_mag))`
graph with every VJP spelled out as inference ops (LearnableSigmoid, two
Linears, PReLU, global max-pool with tie splitting, InstanceNorm's three-term
backward, and stride-2 `Conv` input-grads via zero-insertion upsampling —
`ConvTranspose` is outside the device vocabulary).

* **Matches `torch.autograd` to 3.9e-07 relative**, first run.
* Exports at opset 13 with torch-vs-ONNX parity **6.0e-08**.
* Compiles: 169 epochs (13 HW / 18 hybrid / **138 SW**), 2.70 MB weights,
  3.49 MB activations. It does *not* fit `n6-noextmem` — the zero-insertion
  upsample materializes a 1.0 MB intermediate — so it needs `n6-allmems-O3`.

### On-target: the score is exact, the gradient is garbage

| candidate | score host | score **device** | gradient cosine | ratio |
|---|---:|---:|---:|---:|
| 3 | 3.292 | **3.292** | -0.022 | 1.6e9 |
| 88 | 1.104 | **1.104** | 0.002 | 1.7e10 |
| 123 | 3.647 | **3.647** | -0.005 | 6.8e10 |

The score output is reproduced **exactly**; the gradient output is corrupt by
nine to eleven orders of magnitude.

### Why this matters for the bug report

This is a **much cleaner reproducer than the DNSMOS one, and it is pure fp32** —
no quantization anywhere. That kills the "int8<->float boundary" framing as the
*sole* explanation and replaces it with something sharper:

| graph | outputs | result |
|---|---|---|
| ConvFSENet trunk | 1 (+9 states) | correct |
| DNSMOS forward (int8) | 2, both tiny | correct (+0.23 bias) |
| **PESQ predictor forward** | **1 tiny** | **exact** |
| **PESQ loss (fp32)** | 2: tiny + **64.7 kB** | **tiny exact, large corrupt** |
| DNSMOS loss (int8) | 3: two tiny + 64 kB | all corrupt |

The pattern across every graph measured this session: **a large tensor written
as a graph output by a software epoch comes back wrong, while small outputs of
the same inference are exact.** That holds in fp32, so it is not a quantization
bug; quantization appears to widen the blast radius rather than cause it.

That is a far more actionable statement for ST than "the int8/float boundary
misbehaves", and the PESQ loss graph is a better attachment than the DNSMOS one:
fp32-only, 169 epochs instead of 204, no hand-quantization, no BOOL-in-JSON
workaround, and a self-checking pass/fail (score exact + gradient wrong in the
same inference).

---

## On-device adaptation DOES help — under domain shift, with the metric clamped

`reverb_adapt_eval.py`. The negative PESQ result on VoiceBank-DEMAND was the
wrong test: ConvFSENet is a PESQ metric-GAN *trained on VBD*, so it starts near
its optimum and an optimizer can only walk downhill. Adaptation is for
conditions the model never saw. Reverberant noisy speech is one — VBD contains
none.

Reference is the **reverberant clean** signal (reverb kept, noise removed): a
denoiser cannot dereverberate, so scoring against the anechoic clean would
charge it for a task it was never given.

The shift is real: it costs the shipped enhancer **1.040 PESQ** (3.3–3.9
anechoic → 1.95–2.63 reverberant). Headroom now exists.

| | mean gain | improved on | optimism |
|---|---:|---:|---:|
| VBD (no headroom, unclamped) | **−0.722** ±0.128 | 0/7 | +5.30 |
| reverb (headroom, unclamped) | −0.054 ±0.125 | 3/7 | +5.72 |
| **reverb, metric clamped at PESQ 4.5** | **+0.164 ±0.091** | **7/7** | +2.64 |

Two independent effects, cleanly separated:

* **Headroom stops the losses.** VBD's −0.72 (resolved) becomes −0.05 (not
  resolved) once the enhancer is off its training distribution. The user's
  hypothesis was right, and necessary — but not sufficient on its own.
* **Clamping unlocks the gains.** The predictor's `LearnableSigmoid(beta=2)`
  can emit up to **PESQ 8.0**, and the optimizer drove it to 7.8 — chasing a
  score the metric cannot mean. Capping it at PESQ 4.5 makes the gradient
  vanish once the claim becomes unjustifiable, which halves the optimism
  (+5.72 → +2.64) and turns the gain positive and unanimous.

**+0.164 PESQ on 7/7 clips, 95% CI excluding zero**, from adapting 49,408
parameters against a metric the project owns, on features the enhancer already
computes. That is the first resolved evidence in this repo that the on-device
adaptation loop does what it was built to do.

### Honest limits

* The predictor was trained only on **non-reverberant** VBD, so under reverb it
  is doubly off-distribution; the residual +2.64 optimism is unsurprising and
  retraining with reverberant candidates is the obvious next step.
* Synthetic RIRs (exponentially-decaying noise) create the shift but are not
  acoustically exact; real RIRs would strengthen the claim.
* 7 utterances, one RIR family, one rt60. The CI is honest for that sample but
  the sample is small.
* The clamp is a *brake*, not a fix — it stops the optimizer chasing impossible
  scores, but the metric is still over-reporting by 2.64 PESQ inside the
  allowed range.

### The LearnableSigmoid was the problem — a plain sigmoid replaces the clamp

eco8's `LearnableSigmoid1d(beta=2)` can emit **PESQ 8.0**, nearly double what
the metric can mean, and the optimizer went straight for it. Retrained with a
plain `nn.Sigmoid` (ceiling PESQ 4.5, structural rather than enforced):

| variant | correlation | adaptation gain | improved | optimism |
|---|---:|---:|---:|---:|
| lsig, unclamped | 0.964 / 0.961 | −0.054 ±0.125 | 3/7 | +5.72 |
| lsig + external clamp | 0.964 / 0.961 | **+0.164** ±0.091 | **7/7** | +2.64 |
| **plain sigmoid, no clamp** | 0.959 / 0.955 | **+0.120** ±0.080 | 6/7 | **+2.14** |

* **Correlation is unaffected** (0.959/0.955 vs 0.964/0.961 — within noise), so
  the extra output range was pure liability. The concern that `lsig`'s
  learnable slope also controls sharpness did not materialize.
* **The bound is now structural.** No clamp, no tuning parameter, and the
  gradient vanishes at the top of the valid range because there is nowhere
  further to go.
* **Lowest optimism of the three (+2.14)** — the architectural fix beats the
  external guard on the metric that matters for exploitation.
* The gain, +0.120 [±0.080], is statistically resolved and within noise of the
  clamped lsig's +0.164; 6/7 rather than 7/7, with the one regression at
  −0.019.

The clamp remains available (`--clamp`) but is no longer needed; the sigmoid
head is the default for new models.

Residual optimism of ~2 PESQ is real, and the obvious lever is to train the
predictor on reverberant candidates it has never seen. **Deliberately not
doing that**: this is a POC of on-device adaptation, and the gap in the
training data is the thing being adapted *for*. Close it and the adaptation
step has nothing left to recover — the experiment would then measure the
predictor's generalization, which is a different question. The optimism figure
is a property of the setup, not a defect to remove.

### The sigmoid head on target: same result, one epoch cheaper

Switching heads changed the backward, which hardcoded the LearnableSigmoid's
`slope` and would have crashed on the new checkpoint. `PesqLossGraph` now
branches on the head, and the plain sigmoid is the cheaper of the two: its
derivative is `y*(1-y)` in the forward output already on the wire, so the head
costs one `Sub` and one `Mul` and no parameter. Both heads gate against
autograd at **3.9e-07**.

Re-exported, recompiled for `n6-noextmem`, flashed, and run
(`run_pesq_on_target.py`, now tracked — the previous on-target run was ad hoc):

| | epochs | HW / hybrid / SW | total |
|---|---:|---|---:|
| lsig head | 35 | 12 / 0 / 23 | 727 kB |
| **sigmoid head** | **33** | **12 / 0 / 21** | 739 kB |

| candidate | true PESQ | host int8 | **device** | \|d\| |
|---|---:|---:|---:|---:|
| 3 | 3.19 | 2.95 | **2.95** | 0.000 |
| 17 | 4.14 | 3.54 | **3.52** | 0.015 |
| 42 | 2.14 | 2.84 | **2.86** | 0.018 |
| 88 | 1.10 | 1.13 | **1.13** | 0.003 |
| 123 | 3.79 | 3.29 | **3.29** | 0.000 |
| 260 | 3.49 | 2.39 | **2.39** | 0.000 |

**device vs host int8: mean 0.0060 PESQ, max 0.018, at 37.6 ms/inference** —
identical latency to the lsig build, and the output spans 1.13..3.52, so it is
tracking its input rather than returning the DNSMOS loss graph's constant. The
head swap costs nothing on target.

The gradient half of this graph still hits the ST large-output defect; that is
unchanged by the head and remains blocked on the ticket.

## The adaptation budget curve: the optimum is ~10 steps, not 40

`adapt_budget.py`. Every adaptation run before this used a fixed 40 steps,
chosen arbitrarily. MetricGAN's premise is that a frozen surrogate's gradient
"is only accurate for the first few learning iterations", and their fix —
retrain the surrogate alternately — **is unavailable on an MCU**: the predictor
is frozen in flash, there are no labels at the edge, and there is no room for a
second training loop. So the qualitative warning has to become a number.

150 reverberant VBD test clips (3 s, rt60 drawn per clip in [0.2, 0.6]),
true PESQ against the **reverberant clean** reference at each step count.

| step | true PESQ | gain | 95% CI | improved | exploitation |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.077 | — | — | — | +0.000 |
| 1 | 2.096 | +0.018 | ±0.006 | 77% | +0.516 |
| 4 | 2.129 | +0.051 | ±0.020 | 76% | +0.790 |
| **10** | **2.161** | **+0.083** | **±0.032** | **70%** | +0.812 |
| 20 | 2.152 | +0.075 | ±0.041 | 62% | +0.847 |
| 40 | 2.134 | +0.057 | ±0.046 | 57% | +0.876 |
| 100 | 2.113 | +0.036 | ±0.047 | 55% | +0.902 |

### Four findings

**1. The curve peaks and declines.** The optimum is ~10 steps. The 40-step
protocol used everywhere earlier gives up ~30% of the available gain (+0.057
vs +0.083) and drops the improved-clip rate from 70% to 57%. It is not
catastrophic — an earlier 30-clip run at 4 s suggested 40 steps was actually
*negative*, which the larger sample does not support — but it is clearly
suboptimal, and the peak location is now measured rather than assumed.

**2. Most of the "optimism" was never Goodhart.** Splitting it:

* **calibration error at step 0: +1.483 PESQ** — the predictor is wrong about
  reverberant audio it never saw, before any optimizer acts
* **exploitation: +0.90 PESQ at maximum budget** — the part adaptation created

The headline "+2.14 optimism" quoted earlier was ~62% calibration error. These
have different fixes (training data vs. frozen-critic control) and quoting the
sum overstated how compromised the metric is.

**3. Exploitation is near-instantaneous, not gradual.** +0.52 after a *single*
step, then a slow creep to +0.90 over the remaining 99. The natural reading of
MetricGAN's "accurate for the first few iterations" — as gradual degradation —
does not describe what happens here. The first step moves the output off the
predictor's manifold and the metric pins near its ceiling (4.50 of a possible
4.50) almost immediately; what declines afterwards is true quality drifting
while the metric has nothing left to say.

**4. A fixed budget captures about half of what is available.**

| stopping rule | gain |
|---|---:|
| best fixed budget (10 steps) | +0.083 |
| oracle per-clip | **+0.178** |
| headroom for a better rule | +0.094 |

Per-clip optimum: median 10, IQR 4–30, 10–90% **0–81**. The 10th percentile
being 0 means for at least a tenth of clips the correct action is *not to adapt
at all*. A cheap on-device gating/stopping rule has ~+0.09 PESQ to compete for
— more than the fixed-budget gain itself.

### The trust region raises the peak but does not create it

Ablating the SI-SNR floor (`--no-sisnr`), same 150 clips:

| | peak step | gain at peak | gain at 100 | exploitation@100 | SI-SNR drift |
|---|---:|---:|---:|---:|---|
| trust region ON | 10 | **+0.083** ±0.032 | +0.036 | +0.902 | 7.91 → 9.30 |
| trust region OFF | 10 | +0.051 ±0.028 | −0.009 | +0.947 | 7.91 → 7.28 |

* **The peak survives the ablation at the same step count**, so the budget
  effect is the metric's own behaviour, not an artifact of the constraint. The
  budget is the real control variable.
* The trust region is worth **+0.032 PESQ** at the peak and keeps the tail from
  going negative.
* It barely changes *exploitation* (+0.90 vs +0.95) — it constrains the output,
  not the metric's belief. It limits the damage, not the fooling.

### Honest limits

* **+0.083 PESQ is statistically resolved but perceptually marginal** — below
  the ~0.1–0.2 usually taken as a PESQ JND. 30% of clips still get worse at the
  optimal budget. The oracle +0.178 would matter; the fixed budget arguably
  does not. Capturing that headroom is the thing that decides whether this
  line is worth pursuing.
* RIRs remain synthetic exponentially-decaying noise. Varying rt60 per clip is
  a broader condition set than the single value used before, but it is one
  family, not real rooms.
* Single enhancer, single seed, one shift type.

## The oracle headroom is not reachable: the metric saturates into a constant

`adapt_gate.py`. The budget curve left +0.094 PESQ between the best fixed
budget (+0.083) and oracle per-clip stopping (+0.178). The hypothesis was that
the metric's own trajectory — how fast it pins to its ceiling — would flag
clips being exploited, at zero extra compute.

**It does not, and no other device-visible signal does either.** Admissible
signals are only those the edge actually has: the predictor's output, and
SI-SNR against the *noisy* input (not clean, but free). True PESQ is oracle-only.

| signal | r vs optimal step | r vs oracle gain |
|---|---:|---:|
| believed at step 0 | +0.023 | +0.140 |
| belief rise by step 2 | +0.003 | −0.117 |
| belief rise by step 10 | −0.018 | −0.148 |
| SI-SNR at step 0 | −0.079 | −0.180 |
| SI-SNR drift by step 10 | +0.058 | +0.163 |

Nothing correlates with where a clip should stop. Five rules, each with its
threshold **cross-validated** (fit on training folds, scored held-out, the
fixed budget given the same treatment so the comparison is fair):

| rule | gain | 95% CI | improved | vs fixed |
|---|---:|---:|---:|---:|
| fixed budget | +0.082 | ±0.033 | 69% | — |
| belief-rise cap | +0.030 | ±0.032 | 61% | −0.052* |
| belief saturation | +0.067 | ±0.035 | 71% | −0.015* |
| headroom gate | +0.083 | ±0.032 | 69% | +0.001 |
| SI-SNR drift cap | +0.048 | ±0.038 | 60% | −0.034* |
| never adapt | +0.000 | — | 0% | −0.082 |
| *oracle (not a rule)* | *+0.178* | *±0.029* | *88%* | *+0.096* |

`*` = paired difference resolves at 95%. The best rule captures **1%** of the
headroom — the headroom gate simply degenerates to the fixed budget, since its
fitted threshold (4.15) fires for nearly every clip.

### Why: the signal's variance collapses while the outcome's does not

| step | believed: mean | **std across clips** | frac > 4.45 |
|---:|---:|---:|---:|
| 0 | 3.560 | **0.2640** | 0% |
| 1 | 4.094 | 0.1559 | 0% |
| 4 | 4.402 | 0.0424 | 9% |
| 10 | 4.455 | 0.0350 | 76% |
| 25 | 4.487 | 0.0078 | 100% |
| 100 | 4.498 | 0.0010 | 100% |

The metric's spread across clips falls **250x**, to 0.001 PESQ, while the true
outcome at step 10 still has std 0.202 and ranges −0.78 to +0.59. By step 25
every clip sits within 0.05 of the 4.50 ceiling. The predictor has become a
constant function of its input.

That is the same mechanism as the instantaneous exploitation seen in the budget
curve, and it forecloses a whole family of solutions: **a saturating metric
cannot serve as its own stopping criterion**, because at the moment you need it
to discriminate, it has stopped discriminating. 12% of clips have oracle gain
<= 0 — they should never be adapted — and nothing visible identifies them.

### What this implies

Realistic ceiling for this configuration is the fixed-budget **+0.083 PESQ**,
which is below the ~0.1–0.2 PESQ JND. Adaptation works, is statistically
resolved, and is perceptually marginal. Making it matter needs a metric that
*retains resolution off its training manifold* — a different design criterion
from the correlation-on-natural-data that metrics are normally selected for,
and one that this project's earlier surrogate comparisons never tested.

There is a concrete route that stays compatible with a frozen on-device critic.
MetricGAN's replay buffer fixes exactly this, but *online*, which an MCU cannot
do. The same augmentation can be applied **offline**: generate exploited
outputs by running this adaptation loop during dataset construction, label them
with real PESQ, and train the predictor to score them correctly. The critic is
still frozen at deployment; the anti-exploitation training simply happens
before the flash. That is untested here and is the obvious next experiment.

## Offline replay does not substitute for online replay — the experiment failed

`pesq_adv_dataset.py`. The saturation finding suggested a fix compatible with a
frozen on-device critic: MetricGAN's replay buffer solves exactly this problem
but *online*, which an MCU cannot do, so do the same augmentation **offline**.
Run the adaptation loop during dataset construction, capture the masks it
produces, label them with real PESQ, and train the predictor to score them
correctly. Critic still frozen at deployment; only the anti-exploitation
training moves before the flash.

Built: 2,240 adversarial candidates from 224 anechoic VBD *train* utterances
(5 capture points x 2 trust-region settings), appended to the original 2,800.
Reverb was deliberately excluded — it is the held-out shift the whole
adaptation experiment exists to recover. The exploits were exactly the region
the hand-designed `candidate_masks` misses:

| capture step | true PESQ | v1 believed | gap |
|---:|---:|---:|---:|
| 1 | 2.75 | 4.07 | +1.32 |
| 10 | 2.41 | 4.45 | +2.04 |
| 100 | 2.09 | 4.50 | **+2.40** |

### It did not work

| | v1 (natural) | v2 (+ replay) |
|---|---:|---:|
| **belief std across clips, step 0** | 0.2640 | 0.3477 |
| **belief std across clips, step 100** | 0.0010 | 0.0015 |
| **std collapse** | **276x** | **230x** |
| frac > 4.45 at step 100 | 100% | 100% |
| calibration error at step 0 | +1.483 | **+1.010** |
| exploitation at max budget | +0.902 | **+1.371** |
| best fixed budget | **+0.083** ±0.032 | +0.063 ±0.031 |
| oracle stopping | +0.178 | +0.156 |
| best gating rule (CV) | 1% of headroom | 4% of headroom |
| natural-data pearson (jointly held out, n=83) | **+0.961** | +0.951 |

The mechanism is untouched: v2 still collapses to a constant, still pins every
clip above 4.45 by step 100. The adaptation gain went *down*, exploitation went
*up*, and natural-data accuracy paid a small price (MAE 0.182 -> 0.220).

### Why — and this is the useful part

**Exploitation is defined relative to the critic being attacked.** Training v2
on v1's exploits inoculates it against v1's blind spots; the optimizer then
finds v2's, which are different ones. v2 starts *lower* (3.087 vs 3.560, the
calibration win) and still climbs to the same 4.495 ceiling, so it actually has
further to travel — hence more measured exploitation, not less.

This makes MetricGAN's alternation **essential rather than incidental**. The
target moves as the critic changes, so a fixed number of offline rounds cannot
converge on a moving target; only tracking it can. One round was never going to
be enough, but the failure is not "needs more rounds" — each round redefines
what an exploit is.

The one thing offline replay did buy is real and worth keeping: **baseline
calibration improved from +1.483 to +1.010**, and it did so on *reverberant*
audio that the augmentation never contained. That gives a clean dichotomy:

* **static errors** — being wrong about a distortion type never seen — are
  fixable offline, and generalize across shifts
* **adaptive errors** — being wrong about whatever an optimizer is currently
  producing — are not, because the error is a function of the attacker

### What this means for the direction

The frozen-critic constraint is **fundamental, not an engineering
inconvenience**. It cannot be trained away offline. So on-device metric-driven
adaptation with a learned critic is capped near the fixed-budget **+0.083 PESQ**
measured earlier — below the PESQ JND — unless the objective itself changes to
something that cannot be exploited by construction (a non-learned signal-domain
criterion), or the device gains the ability to update its critic online, which
is the thing the hardware forbids.

That is a negative result about the approach, established at n=150 with
cross-validated rules and a mechanism that explains it, rather than an
engineering gap waiting to be closed.

## Gate 0: the classical baseline — and the static preset that ends the premise

`classical_baseline.py`. Before building further on the +0.083 learned-critic
result, the question that should have been asked first: can classical adaptive
DSP match it? Same 150 clips, same RIRs (identical rng draw order — every
number paired), all methods starting from the enhancer's shipped output.

| method | gain | 95% CI | improved |
|---|---:|---:|---:|
| decision-directed Wiener post-filter | −0.545 | ±0.046 | 0% |
| rescale grid, SI-SNR-floor pick | −0.302 | ±0.072 | 20% |
| rescale grid, v1 zeroth-order pick | −0.327 | ±0.042 | 7% |
| rescale grid, v2 zeroth-order pick | −0.142 | ±0.045 | 33% |
| rescale grid, **oracle pick** | **+0.687** | ±0.051 | 100% |
| *(reference: gradient loop, fixed 10 steps)* | *+0.083* | *±0.032* | *70%* |
| *(reference: gradient loop, oracle stop)* | *+0.178* | *±0.029* | *88%* |

Read as designed, Gate 0 *passes*: every device-pickable classical method is
negative, and the gradient loop's +0.083 survives as the only positive
device-visible adaptation. But the oracle column gives the game away twice.

**First: the grid oracle is +0.687 — 4x anything gradient adaptation can
reach even with perfect stopping.** And its picks are nearly constant:
`a=0.5, f=0.1` (soften the mask, floor the suppression) on 113/150 clips. So
apply that one setting globally, with no selection at all:

| static preset | gain | 95% CI | improved |
|---|---:|---:|---:|
| **a=0.5, f=0.1** | **+0.641** | ±0.062 | **95%** |
| a=0.65, f=0.1 | +0.534 | ±0.045 | 97% |
| a=0.5, f=0.05 | +0.606 | ±0.058 | 95% |

**A fixed global mask-softening captures 93% of the per-clip oracle and beats
the entire learned-critic apparatus 8-fold.** The reverb shift's damage is a
*systematic* miscalibration — the enhancer over-suppresses under reverb — and
one knob corrects it for 95% of clips. No labels, no critic, no gradient, no
per-clip decisions.

**Second: the learned critics rank the family backwards.** Zeroth-order
selection over 21 near-manifold candidates — no gradient, no saturation, the
setting where a critic should be safest — picks `a=2` (MORE suppression, 57 of
150 clips) and lands at −0.327. Truth wants `a=0.5`. The critics, trained on
anechoic VBD where suppression is good, prefer exactly the wrong direction
under reverb; the gradient loop's meagre +0.083 was won *against* its own
metric's preference, presumably via the trust region. This also kills the
"zeroth-order use is safe" hypothesis: the critic is not just exploitable
under optimization, it is *miscalibrated in ranking* off-distribution.

### What Gate 0 actually decides

The premise dies, but not the way the gate anticipated. Classical *adaptive*
methods lose; a classical *static correction* wins overwhelmingly. The honest
statement:

> On this testbed, the domain shift is too simple to justify adaptation. Any
> shift whose correction is expressible as a global preset will be won by the
> preset; test-time adaptation can only justify itself under shifts whose
> correction varies per clip, per user, or per device — and this one does not.

Which retroactively reframes the whole reverb line: the +0.083 was never
"adaptation working under domain shift"; it was adaptation clawing back a
sliver of a fix that a constant knob delivers 8x better. The correct
evaluation for on-device adaptation needs a shift with no global fix —
speaker-specific, device-specific, or multi-condition mixtures where the
preset that helps one clip hurts another.

## Exp 1 for the record: the judge stays awake, and it still doesn't matter

`adapt_budget.py --judge`. Run after Gate 0 had already decided the premise,
to complete the record: gradient from v1, v2 recorded per checkpoint but never
differentiated through.

The mechanism hypothesis was **confirmed**:

| step | attacked std | judge std | judge r(true) | judge−true |
|---:|---:|---:|---:|---:|
| 0 | 0.2640 | 0.3477 | +0.538 | +1.010 |
| 10 | 0.0350 | **0.4165** | +0.663 | +0.243 |
| 100 | 0.0010 | **0.4109** | +0.686 | **+0.021** |

While the attacked metric collapses 264x, the judge's spread across clips
*grows*, its correlation with truth *improves* (0.54 → 0.69), and its
calibration error vanishes (+1.01 → +0.02) — the trajectory walks INTO v2's
training distribution (v1-exploits are exactly what v2 was trained on).
Exploits being critic-specific cuts both ways, exactly as predicted.

And yet the cross-validated stopping rules built on it still lose:

| rule | gain | vs fixed |
|---|---:|---:|
| fixed budget | +0.082 ±0.033 | — |
| judge drop | +0.054 | −0.028* |
| divergence cap | +0.056 | −0.026* |
| judge accept/reject | +0.064 | −0.018 |

A discriminative, well-calibrated judge (r = 0.69) is still too weak an
instrument for a per-clip stopping decision whose signal is ±0.1 PESQ of
trajectory curvature. The oracle headroom (+0.096) remains uncaptured; the
mechanism was necessary but not sufficient.

The don't-differentiate-through-the-judge principle survives as the one
constructive finding: one round of offline replay produces a judge that stays
honest along the attacker's trajectory — it is just that on this testbed there
is nothing worth judging. Worth re-testing if a valid testbed (no global
preset fix) ever shows adaptation winning.

## The mixed-shift testbed: adaptation is finally needed — and the learned loop still can't deliver it

`mixed_adapt_eval.py`. The testbed the reverb study said was required: 150
clips, each drawing ONE of four shifts whose corrections conflict — reverb
(wants a softer mask), noise at 2-4x (wants a harder one), noise re-coloring,
telephone-ish bandlimit (wants ~identity). Null hypothesis: the best single
global preset, fit on the test set itself.

**Validity holds.** The null collapses from reverb-only's +0.641 to
**+0.087 ±0.043** (52% improved), and its per-kind signs conflict as designed
(+0.481 on reverb, −0.064 noise, −0.079 tilt, +0.002 bandlimit). Meanwhile the
per-clip grid oracle is +0.221 — for the first time, genuinely per-clip
decisions have real headroom (+0.135) over any constant.

| method | gain | 95% CI | improved | vs null |
|---|---:|---:|---:|---:|
| global preset (null, test-fit) | +0.087 | ±0.043 | 52% | — |
| global preset, cross-validated | +0.087 | ±0.043 | 52% | ±0.000 |
| per-kind preset table, cross-validated | **+0.166** | ±0.057 | 59% | **+0.079** |
| per-clip grid oracle | +0.221 | ±0.053 | 91% | +0.135 |
| v1 zeroth-order pick | −0.132 | ±0.032 | 19% | −0.219 |
| v2 zeroth-order pick | −0.045 | ±0.025 | 38% | −0.131 |
| **gradient v1, 10 steps** | **−0.170** | ±0.055 | 28% | −0.256 |
| gradient + oracle stop | +0.056 | ±0.017 | 78% | −0.030 |
| judge accept/reject | −0.054 | ±0.033 | 19% | −0.140 |

Per-kind, the gradient loop: reverb +0.106, noise **−0.420 (0% improved)**,
tilt −0.214, bandlimit −0.151.

### Three conclusions, and this closes the arc

**1. The learned-critic loop fails precisely where adaptation is finally
justified.** On the testbed with real per-clip headroom, gradient adaptation
is *harmful* overall (−0.170), harmful on three of four shifts, and its
ceiling with PERFECT per-clip stopping (+0.056) still loses to a single
global constant (+0.087). This is no longer "marginal gain, below JND" — it
is the wrong sign, including on the noise shift, which is the critic's own
training noise merely made louder.

**2. What actually works is discrete, not continuous.** A per-kind preset
table with honestly cross-validated presets (+0.166) captures 75% of the
per-clip oracle, needing only a 4-way domain decision — and the four shifts
are acoustically distinct enough that a trivial classifier is plausible.
Classify-then-preset beats optimize-against-a-judge everywhere we measured.

**3. The reverb result was not an unlucky testbed; it was the lucky one.**
Under reverb the critic's errors happened to point somewhere harmless enough
that the trust region could salvage +0.083. Under the noise shift the same
loop destroys −0.420. The learned critic is not a noisy compass; off its
training distribution it is a compass pointing wrong, with confidence.

### The study's final shape

Adaptation-by-learned-metric on a frozen edge device fails for three
independent, now-measured reasons: the critic saturates under optimization
(instant, 276x), its failures cannot be patched offline (moving target), and
its steering is miscalibrated off-distribution even zeroth-order (backwards
ranking). Where shifts are simple a preset wins; where shifts conflict a
classify-then-preset table wins; nowhere does the gradient loop win. The
salvageable positives: the deployment stack itself, the calibration/
exploitation decomposition, the bystander-judge mechanism, and the testbed
design criterion — a study whose value is the map of why, not a method.

## Capacity or direction? Two knobs settle it

`two_knob_adapt.py`. The mixed-shift failure invites the objection that
adapting only the 49k-parameter mask head was too little, or the wrong,
capacity. Strongest possible test of that objection: reduce adaptation to
TWO scalars, `m = sigmoid(t*z + c)` with the head's logits `z` frozen — a
family that *contains* the known reverb fix (t~0.5 softens like m^0.5, c>0
raises the floor). Nothing here can be "too small"; there is almost nothing
to exploit; the right answer is on the table. Same 150 paired reverberant
clips.

| within the same 2-parameter family | gain | 95% CI | improved |
|---|---:|---:|---:|
| fixed (t=0.5, c=0.5) | **+0.731** | ±0.078 | 94% |
| per-clip (t,c) oracle | +0.868 | ±0.080 | 99% |
| critic-steered, step 5 | −0.089 | ±0.047 | 35% |
| critic-steered, step 40 | **−0.244** | ±0.082 | 27% |

And where the critic drives the knobs (truth wants t~0.5, c>0):

| step | mean t | mean c |
|---:|---:|---:|
| 0 | 1.000 | +0.000 |
| 10 | 1.071 | −0.102 |
| 40 | **1.117** | **−0.345** |

**Both knobs, turned the wrong way.** Holding exactly the two parameters
that contain a +0.73 fix, the critic hardens the mask (t up) and deepens the
suppression floor (c down), losing −0.244 — monotonically worse with more
steps. Capacity is exonerated conclusively: the 49k-parameter loop never
lacked the winning direction (the preset is ~`W,b -> 0.5W, 0.5b`, inside its
space all along); it lacked a critic that knew which way "better" was.

Corollary for the "adapt more parameters" instinct: with a miscalibrated
compass, capacity multiplies damage — every extra dimension is another
direction to be steered wrong in. The mixed testbed already showed the 49k
version at −0.42 on the noise shift; two parameters merely lose −0.24.

## The first honest win: feature-routed presets beat the null, end to end

`knob_router.py`. Classify-then-preset had assumed the 4-way domain label;
this closes the gap with two fully device-visible routers, trained on the
per-preset gains already measured in `mixed_shift.npz`, everything (router
weights, presets, thresholds) fit on train folds only:

* **classify -> table** — multinomial logistic over 13 classical DSP features
  (envelope autocorrelation, modulation depth, noise-floor level/tilt,
  spectral rolloff, the enhancer's own mask statistics and SI-SNR) predicts
  the kind; the per-kind table is applied to the *predicted* label.
* **gain regression -> argmax** — ridge predicts each of the 21 presets'
  gains from the same features; apply the per-clip argmax. No discrete label;
  ceiling is the per-clip oracle rather than the table.

| method (all CV, n=150 mixed) | gain | 95% CI | improved | vs null |
|---|---:|---:|---:|---:|
| gradient loop (v1, 10 steps) | −0.170 | ±0.055 | 28% | −0.256 |
| global preset (null) | +0.087 | ±0.043 | 52% | — |
| classify → table, honest | +0.126 | ±0.058 | 55% | **+0.039\*** |
| **gain regression → argmax, honest** | **+0.129** | ±0.056 | 56% | **+0.043\*** |
| per-kind table, oracle labels | +0.176 | ±0.057 | 68% | +0.089\* |
| per-clip grid oracle | +0.221 | ±0.053 | 91% | +0.135\* |

`*` = paired difference vs the null resolves at 95%.

**This is the first fully honest, device-computable mechanism in the study
that beats the best global constant with statistical resolution** — and it is
thirteen hand-crafted DSP features and a ridge regression, fit offline
against real PESQ. Against the learned-critic gradient loop on the same
clips, the swing is +0.30 PESQ.

Details worth keeping:

* Router accuracy is 77% (reverb 87%, bandlimit 100%, noise/tilt confused —
  harmlessly, since those two kinds share the same best preset). The gap to
  the oracle-label table (+0.176) is pure router accuracy; the gap from there
  to +0.221 is within-kind refinement. Both are ordinary supervised-learning
  engineering — more features, more training clips — not research risk, which
  is the structural difference from the critic path.
* The most informative features are modulation depth and lag-4 envelope
  autocorrelation (reverb dynamics), the top-octave noise floor, and —
  notably — `sisnr0`, how far the enhancer already moved the signal: the
  enhancer's own behaviour is itself a usable domain sensor.
* A confidence fallback (commit to the kind preset only when the router is
  sure, else the global preset) adds nothing here (+0.127) — the fitted
  thresholds go low, i.e. committing is already the right call at 77%.
* Runtime cost: the features are a few statistics over the STFT the enhancer
  already computes, plus one 13x21 matrix multiply. No backward pass, no
  learned critic, the ST defect is irrelevant.
