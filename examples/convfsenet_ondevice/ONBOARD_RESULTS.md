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

1. **The trained ConvFSENet scores far below noisy on official DNSMOS** for
   these clips (its masks match upstream's deployed `g_best.onnx` — mean
   mask stats 0.142 vs 0.144 — so this is the real enhancer, not a port
   bug). DNSMOS-driven adaptation recovers part of the gap. (PESQ, the
   metric ConvFSENet was tuned for, tells the opposite story — 2.911 int8 on
   VBD; the two metrics genuinely disagree about this model.)
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
