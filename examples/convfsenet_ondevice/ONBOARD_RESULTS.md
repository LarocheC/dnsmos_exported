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
