# Bare-runtime DNSMOS loss-graph firmware (STM32N6570-DK)

Runs the compiled 0.25 s DNSMOS loss network (scores + `dL/d(wav)`) directly on
the LL_ATON runtime with the input **baked into the image** — no protobuf, no
`ai_runner`, no serial feeding. Built to answer one question: when the same
compiled network returns near-constant garbage under ST's `NPU_Validation`
stack while ST's own optimized-export graph is numerically correct in
onnxruntime, *which side is at fault?*

**Answer: not the validation stack.** This firmware reproduces the identical
wrong constant (bit-for-bit) for two different input clips. See
`../ONBOARD_RESULTS.md` for the full evidence chain.

## Run

```bash
python gen_loss_input.py                 # bakes 2 VBD clips into loss_input.h
./build_and_run.sh                       # patch + build + flash + load + capture
SKIP_FLASH=1 ./build_and_run.sh          # skip the octoFlash weight write
GEN=../n6_gen/<other_build> ./build_and_run.sh
```

Patches ST's `NPU_Validation` project in place with `.loss_bak` backups
(restore from those when done), exactly as eco8-neaixt's `firmware_router`
does.

## Two gotchas worth knowing

- **No float printf.** This project does not link newlib's `%f`/`%e`; using
  them wedges the UART mid-line with no error. Print IEEE-754 bits (and scaled
  integers) instead — the bits are what a host-side comparison wants anyway.
- **`LL_ATON_RT_RuntimeInit()` is mandatory** in a bare runner. Without it the
  first `LL_ATON_RT_RunEpochBlock()` parks in `LL_ATON_OSAL_WFE()` forever. The
  validation stack calls it inside `ai_wrapper_ATON.c`, which is easy to miss.

## Files

| file | role |
|---|---|
| `orchestrator_loss.c` | the runner: buffer lookup by size, cache maintenance, epoch loop, UART report |
| `gen_loss_input.py` | bakes VBD clips into `loss_input.h` in the compiled graph's device layout |
| `build_and_run.sh` | patch project → build → flash weights → gdb-load → capture UART |
