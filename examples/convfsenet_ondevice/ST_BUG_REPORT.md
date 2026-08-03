# ST Edge AI Core 4.0.1 / Neural-ART: int8↔float software-epoch boundary produces wrong results (and hangs) on STM32N6570-DK

**Summary.** A quantized ONNX graph that contains a large fp32 elementwise
region between int8 convolution blocks compiles cleanly and executes on the
STM32N657 NPU, but returns results that **do not depend on the input**. Two
different input clips produce **bit-identical** outputs, and the fp32 output
tensor contains words such as `0x808080F2` — i.e. int8 zero-point bytes
(`0x80` = −128) appearing inside a float32 buffer — plus NaNs. A larger
instance of the same graph **hangs** on a pure-software `DequantizeLinear`
epoch and never returns.

ST's own optimized export of the same model (`*_OE_*.onnx`, emitted by
`stedgeai generate`) is **numerically correct** when executed on the host with
onnxruntime, so the ONNX graph and the front-end transformation are both
sound. The defect appears in code generation or in the LL_ATON runtime's
handling of the int8/float boundary in software epochs.

We have reproduced the failure through **two independent host→device paths**
(the `NPU_Validation` protobuf stack, and a bare LL_ATON runner with the input
baked into the firmware image), across **four compiler configurations**. A
minimal, self-contained reproducer is attached.

---

## 1. Environment

| item | version |
|---|---|
| ST Edge AI Core | **v4.0.1-20581** (`7ed50de05`) |
| Neural-ART runtime reported by target | LL_ATON API **v1.1.3**, `atonn-v1.1.3-275-g6770925d`, optimized SW lib `v12.0.1-92efada9 GCC` |
| Board | **STM32N6570-DK**, device ID `0x486`, rev B |
| Clocks | MCU 800 MHz, NPU 1 GHz, NOC 400 MHz, NIC 900 MHz |
| STM32CubeProgrammer | 2.22.0 |
| STM32CubeCLT | 1.21.0 |
| Arm GNU toolchain | 13.3.1 (13.3.Rel1) |
| Host | Ubuntu 22.04 on WSL2 |
| Boot mode | development boot, RAM-resident firmware loaded over ST-LINK/gdb |

## 2. The model

A fused forward + backward "loss graph": it computes speech-quality scores
**and** the gradient of those scores with respect to the input waveform. The
backward pass is written as ordinary ONNX inference operators (no training
runtime), which is why the graph mixes int8 convolution blocks with large fp32
elementwise regions (`Equal` → `Cast` → `Sub` → `Mul` masks used for max-pool
gradient routing).

Quantization is standard static QDQ, produced by onnxruntime:
per-channel symmetric int8 weights, per-tensor asymmetric **signed** int8
activations, MinMax calibration on real audio.

Two instances, identical topology, differing only in window length:

| artifact | I/O | epochs (HW / hybrid / SW) | weights | activations | symptom |
|---|---|---|---|---|---|
| `crop0p25s_int8_d2_rows_dev.onnx` | `wav [1,25,160]`, `w [1,3]` → `raw [1,3]`, `mos [1,3]`, `grad_wav [1,25,160]` | 201 (35 / 41 / **125**) | 2.24 MB | 6.67 MB | **wrong results** (§3) |
| `crop1s_int8_d2_rows_dev2.onnx` | `wav [1,100,160]`, same | 204 (33 / 46 / **125**) | 2.36 MB | 23.3 MB | **hang** (§4) |

Note the high pure-software epoch count (125): the fp32 elementwise region is
executed on the Cortex-M55, and that is exactly where both symptoms appear.

## 3. Defect A — output does not depend on the input (0.25 s instance)

### 3.1 Reproduction

```bash
cd "$N6DIR"                       # ST Edge AI N6_scripts dir (mpool paths are relative)
stedgeai generate -m crop0p25s_int8_d2_rows_dev.onnx --target stm32n6 \
  --st-neural-art n6-allmems-O3@user_neuralart.json -n network -o gen_025
```

Then run the generated network with **any** driver and feed two different
inputs. We used both:

* ST's `NPU_Validation` application + `ai_runner` over serial, and
* an attached bare LL_ATON runner (`firmware_loss/`) with the input baked into
  the image as a `const float[]` — no protobuf, no serial input path.

### 3.2 Observed (bare LL_ATON runner, verbatim UART output)

```
=== DNSMOS loss graph, bare LL_ATON runner ===
inputs=2 outputs=3 wav_len=4000 clips=2
buffers: wav@0x342e0000 w@0x342e3e80 mos@0x3434d910 grad@0x342e3e90

-- clip 0: fed wav[0..3] = w=-0.013 (bits=0xbc620000) w=0.004 (bits=0x3b950000) ...
   MOS  SIG=1.909 (bits=0x3ff465d9)  BAK=1.448 (bits=0x3fb959d2)  OVRL=1.143 (bits=0x3f925cd8)
        (4254344844 cycles, 5317 ms @800MHz)
   GRAD n=4000 nonfinite=460
   grad[0..7] bits = 0x4122dc11 0xc10eb6ee 0xc18d7b9e 0xc13d23b4 0xb43b7ff2 0x80807ff2 ...

-- clip 1: fed wav[0..3] = w=-0.036 (bits=0xbd148000) w=-0.059 (bits=0xbd730000) ...
   MOS  SIG=1.909 (bits=0x3ff465d9)  BAK=1.448 (bits=0x3fb959d2)  OVRL=1.143 (bits=0x3f925cd8)
        (4241945612 cycles, 5302 ms @800MHz)
   GRAD n=4000 nonfinite=1304
   grad[0..7] bits = 0x7fc00000 0x7fc00000 0x7fc00000 0x801d80f2 0x7fc00000 ...
```

* The two clips carry **different** input data (see the differing `wav[0..3]`
  words, read back from the network's own input buffer after `memcpy`).
* The three score outputs are **bit-identical** between them
  (1.9094 / 1.4481 / 1.1435).
* The float32 gradient buffer contains `0x808080F2`-pattern words. `0x80` is
  the int8 **zero-point** value used throughout this graph's activations; these
  look like raw int8 tensor bytes being consumed as float32. It also contains
  NaNs (`0x7FC00000`).
* **The two outputs fail differently.** The scores are *deterministically*
  wrong — the same constant, bit-for-bit, on every run and every build we have
  tried. The gradient is *non-deterministically* wrong: across two runs of the
  identical firmware and identical input, the non-finite count changed from
  176 → 460 (clip 0) and 491 → 1304 (clip 1), and the leading words changed
  entirely. Stable-wrong scores plus unstable-wrong gradient suggests a stale
  or uninitialized buffer being consumed, rather than a purely arithmetic
  error.

### 3.3 Expected

Host reference for the same two clips, using the deployable ONNX under
onnxruntime 1.28 (CPU EP):

| clip | expected SIG / BAK / OVRL | device (both clips) |
|---|---|---|
| 12 | 2.403 / 2.404 / 1.619 | 1.909 / 1.448 / 1.143 |
| 30 | 1.392 / 1.424 / 1.190 | 1.909 / 1.448 / 1.143 |

### 3.4 The compiled graph itself is correct

Running **ST's own optimized export**, `gen_025/crop0p25s_int8_d2_rows_dev_OE_3_3_1.onnx`,
on the host with onnxruntime — feeding the input in the layout the compiler
chose, `[1,160,25]` — reproduces the host reference:

| | SIG / BAK / OVRL | gradient cosine vs deployable ONNX |
|---|---|---|
| deployable ONNX (host) | 2.403 / 2.404 / 1.619 | 1.000 |
| **ST OE graph (host, ORT)** | **2.434 / 2.444 / 1.628** | **0.80** |
| ST OE graph (on target) | 1.909 / 1.448 / 1.143 | ≈ 0 (and non-finite) |

So the front-end transformation preserves the semantics; only the on-target
execution of that same graph does not.

## 4. Defect B — hang on a software `DequantizeLinear` (1 s instance)

The same topology at a 4× longer window hangs. Per-epoch progress from the
runtime's epoch callback:

* report epochs `epoch_1` … `epoch_64` complete normally, in **2.5–4.9 s total**
  (last completed epoch: `epoch_64`, `Sub(float)`);
* the firmware **never returns** from **`epoch_65` of 204**, a **pure-SW
  `DequantizeLinear`**, in the same `Equal`/`Cast`/`Sub`/`Mul` mask block that
  produces the corrupted floats in Defect A;
* waited > 10 minutes, twice; identical stop point across five loads and both
  `-O3` and `-O1` builds.

(The runtime's callback index is 0-based and the generate report's epoch IDs
are 1-based; epoch numbers above are quoted in **report** numbering.)

Given that both symptoms sit in the same fp32-elementwise / dequantize region,
we believe A and B are one defect, with the hang being the larger-buffer
manifestation.

## 5. What we have ruled out

| hypothesis | test | result |
|---|---|---|
| Bad ONNX / unsupported construct | export-time lint + `stedgeai generate` completes without error or warning | graph accepted |
| Front-end miscompilation | run `*_OE_*.onnx` in onnxruntime | **correct** (§3.4) |
| Host driver / protobuf I/O path | bare LL_ATON runner, input baked into the image | **same wrong constant, bit-for-bit** |
| Input never reaches the device | print input back from the network's input buffer; independently dump the buffer over SWD after a run | input is **byte-correct** on chip |
| Wrong input byte order | tested both `[1,25,160]` and the compiler's `[1,160,25]` layout | both wrong; OE-in-ORT confirms `[1,160,25]` is the right one |
| Optimization level | `--optimization 3` and `--optimization 1` | identical wrong output |
| Cache maintenance | build without `--Ocache-opt` / `--cache-maintenance`; explicit `LL_ATON_Cache_MCU_Clean_Invalidate_Range` on inputs and `..._Invalidate_Range` on outputs; `npu_cache_invalidate()` | identical wrong output |
| Input/output buffer aliasing (the allocator placed `wav` and `grad_wav` at the same address, `0x342e0000`) | rebuild with `--Opreserve-inputs`; buffers separated (`0x342e0000` / `0x342e3e90`) | identical wrong output |
| Degenerate input (e.g. all-zeros reaching the net) | feed zeros / all-`0x80` / small constants to the graph in ORT | gives 2.395 / 2.743 / 1.756 — **not** the device's constant |
| Graph is inherently input-independent | compare all 277 OE tensors across two inputs in ORT | only 2 are input-independent (two constant equality masks) |

A forward-only network derived from the same model (no backward region, hence
far fewer fp32 software epochs) runs on the same board with only a small
systematic bias, which is consistent with the fault being specific to the
large fp32 elementwise / dequantize region rather than to the model family.

## 6. Working hypothesis

The `0x808080F2` words in a float32 output buffer suggest that a tensor which
should be dequantized to float32 is instead being read as raw int8 (or that a
buffer descriptor's element type / stride is wrong) at the int8→float boundary
of a software epoch. That would explain both the input-independent scores
(downstream epochs computing on a constant, zero-point-filled tensor) and the
non-finite gradient. The hang in Defect B occurs at the first
`DequantizeLinear` of the same region, which fits the same root cause with a
larger buffer.

## 7. Attachments

| file | contents |
|---|---|
| `crop0p25s_int8_d2_rows_dev.onnx` | deployable ONNX, Defect A |
| `crop1s_int8_d2_rows_dev2.onnx` | deployable ONNX, Defect B (hang) |
| `gen_025_*/` | four `stedgeai generate` outputs (`-O3`, `-O1`, no-cache-opt, `--Opreserve-inputs`): `network.c`, `*_OE_*.onnx`, `*_Q.json`, `network_generate_report.txt` |
| `gen_1s/` | generate output for the hanging instance |
| `firmware_loss/` | bare LL_ATON reproducer: `orchestrator_loss.c`, `build_and_run.sh`, `gen_loss_input.py` |
| `uart_capture.txt` | verbatim UART output quoted in §3.2 |
| `host_reference.json` | expected scores + gradient statistics for the two clips, from onnxruntime |

## 8. Questions for ST

1. Is the int8→float32 boundary in **pure-software epochs** known to be
   affected in ST Edge AI Core 4.0.1 / LL_ATON v1.1.3, particularly for graphs
   with many (>100) software epochs and multi-MB fp32 intermediate tensors?
2. Is there a supported way to force the fp32 elementwise region onto a path
   that avoids this — a compiler flag, an epoch-splitting hint, or a different
   `--st-neural-art` profile?
3. Is a multi-output network (here 3 outputs, one of which is the same size as
   an input) subject to any constraint we may have violated?
4. Can you confirm whether the `epoch_65` `DequantizeLinear` hang (Defect B) is
   the same root cause, or should it be tracked separately?

## 9. Additional issues encountered (lower priority, not blocking)

Reported for completeness — each has a local workaround, and each cost
significant bring-up time:

1. **`atonn` rejects its own quantization JSON when it contains `BOOL`
   tensors.** For graphs with `Equal`/`Cast`, the `*_Q.json` emitted by the
   front end contains entries with `"type": "BOOL"`; `atonn` reports
   `INVALID_ARGUMENT:(tensors[N].value.format.type): invalid value "BOOL" for
   type ...GraphInfo.ContainerType`, then **`Ignore Quantization JSON file;
   unsupported version ''`** and proceeds to treat **all** float tensors as
   native floats — silently discarding the entire quantization. Workaround:
   strip `BOOL` entries from the JSON between the front end and `atonn`.
2. **`Pad` with a scalar `constant_value`.** The front end re-emits a
   zero-dimensional `constant_value` initializer that `atonn` then rejects with
   `Failed to build a valid graph: missing shape or size for
   value=Pad_*_constant_value`. Workaround: rewrite every zero-`Pad` as a
   `Concat` with a zeros initializer (and, inside quantized regions,
   pre-quantized int8 zeros at the sibling tensor's zero-point).
3. **`Transpose` of a runtime (non-constant) tensor** fails the front end with
   `INTERNAL ERROR: Mismatch in channel position`, on the Neural-ART path and
   on the plain `--target stm32n6` path alike. A dynamic × dynamic `MatMul` on
   its own compiles; only the `Transpose` breaks. Workaround: pre-transpose on
   the host and pass the transposed tensor in.
4. **`Clip` bounds fed through `DequantizeLinear`** (which is what
   onnxruntime's quantizer emits when a scalar constant is shared) fail with
   `INTERNAL ERROR: None has type NoneType, but expected one of: bytes,
   unicode`. Workaround: fold the dequantized value back into a plain
   initializer.
5. **`Slice` with INT64_MAX / negative `ends`** (standard ONNX exporter output)
   fails with `TOOL ERROR: Error in computation of shapes`. Workaround:
   normalize to explicit in-range bounds.

Items 1 and 2 are arguably the more serious of these five, because they are
*silent or misleading*: item 1 discards quantization while continuing to emit a
working-looking network, and item 2 is triggered by an initializer the tool
itself generated.
