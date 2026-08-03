/* Standalone DNSMOS loss-graph runner for the STM32N6570-DK.
 *
 * Runs the compiled 0.25 s DNSMOS loss network (scores + dL/d(wav)) through the
 * bare LL_ATON runtime, with the input baked in as a const array — no protobuf,
 * no serial feeding, no ai_runner. That is the whole point: the same compiled
 * network returns near-constant garbage under ST's NPU_Validation stack while
 * ST's own optimized-export graph is numerically correct in onnxruntime, so
 * this firmware isolates *which side* is at fault. If the numbers come out
 * right here, the bug is in the validation stack's I/O path; if they come out
 * wrong here too, it is in the generated code or the NPU runtime itself.
 *
 * Prints over USART1 (ST-LINK VCP, /dev/ttyACM0 @ 921600):
 *   - the three MOS scores per clip,
 *   - gradient checksum statistics (mean / min / max / L2) — enough to compare
 *     against the host reference without shipping 16 KB over UART,
 *   - the first 8 gradient values,
 *   - DWT cycle count per inference.
 *
 * Modelled on eco8-neaixt's deploy/stm32n6/firmware_router/orchestrator_router.c
 * (same NPU_Validation project patching + UART capture flow), simplified to a
 * single non-relocatable network.
 *
 * Called from main.c in place of aiValidationInit()/aiValidationProcess().
 */
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "ll_aton_runtime.h"
#include "ll_aton_util.h"
#include "npu_cache.h"
#include "network.h"      /* LL_ATON_NETWORK_IN_NUM / OUT_NUM */
#include "loss_input.h"

LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(network)

/* ---- DWT cycle counter (same accessors eco8's router firmware uses) ---- */
static volatile uint32_t *const DWT_CYCCNT = (uint32_t *)0xE0001004;
static volatile uint32_t *const DWT_CTRL   = (uint32_t *)0xE0001000;
static volatile uint32_t *const DEMCR      = (uint32_t *)0xE000EDFC;
static void dwt_init(void) { *DEMCR |= (1u << 24); *DWT_CYCCNT = 0; *DWT_CTRL |= 1u; }
static inline uint32_t dwt(void) { return *DWT_CYCCNT; }

/* Buffers are matched by byte size: the compiled order need not match the ONNX
 * order, but 16000 / 12 / 3 are unique within this graph, so the lookup is
 * unambiguous and order-proof (eco8's in_by_sz/out_by_sz pattern). */
static const LL_Buffer_InfoTypeDef *buf_by_size(const LL_Buffer_InfoTypeDef *bufs,
                                                uint32_t want)
{
  for (const LL_Buffer_InfoTypeDef *b = bufs; b && LL_Buffer_addr_start(b); b++) {
    if (LL_Buffer_len(b) == want) return b;
  }
  return NULL;
}

/* This project does not link newlib's float printf (%f/%e silently wedge the
 * UART mid-line), so every float is printed as scaled integers plus its raw
 * IEEE-754 bits — the bits are what a host-side comparison actually wants. */
static void pf(const char *tag, float x)
{
  union { float f; uint32_t u; } c = { .f = x };
  long milli = (long)(x * 1000.0f);          /* value scaled by 1e3, printed back */
  const char *sign = (x < 0 && milli / 1000 == 0) ? "-" : "";
  printf("%s=%s%ld.%03ld (bits=0x%08lx)", tag, sign, milli / 1000,
         labs(milli) % 1000, (unsigned long)c.u);
}

static void print_f32_stats(const char *tag, const float *v, uint32_t n)
{
  double sum = 0.0, sq = 0.0;
  float mn = v[0], mx = v[0];
  uint32_t n_nan = 0;
  for (uint32_t i = 0; i < n; i++) {
    float x = v[i];
    if (isnan(x) || isinf(x)) { n_nan++; continue; }
    sum += x; sq += (double)x * x;
    if (x < mn) mn = x;
    if (x > mx) mx = x;
  }
  float mean = (float)(sum / (double)n), l2 = (float)sqrt(sq);
  printf("%s n=%lu ", tag, (unsigned long)n);
  pf("mean", mean); printf(" "); pf("l2", l2); printf(" ");
  pf("min", mn);    printf(" "); pf("max", mx);
  printf(" nonfinite=%lu\r\n", (unsigned long)n_nan);
}

void orchestratorInit(void)
{
  dwt_init();
  /* Global ATON runtime init (IRQ wiring, DMA, epoch scheduler). Without it the
   * first RunEpochBlock parks in LL_ATON_OSAL_WFE() forever — the validation
   * stack does this in ai_wrapper_ATON.c, so a bare runner must do it too. */
  LL_ATON_RT_RuntimeInit();
  printf("\r\n=== DNSMOS loss graph, bare LL_ATON runner ===\r\n");
  printf("inputs=%d outputs=%d wav_len=%d clips=%d\r\n",
         LL_ATON_NETWORK_IN_NUM, LL_ATON_NETWORK_OUT_NUM,
         LOSS_WAV_LEN, LOSS_N_CLIPS);
}

void orchestratorProcess(void)
{
  const LL_Buffer_InfoTypeDef *in_bufs  = LL_ATON_Input_Buffers_Info(&NN_Instance_network);
  const LL_Buffer_InfoTypeDef *out_bufs = LL_ATON_Output_Buffers_Info(&NN_Instance_network);

  const LL_Buffer_InfoTypeDef *b_wav  = buf_by_size(in_bufs,  LOSS_WAV_LEN * 4);
  const LL_Buffer_InfoTypeDef *b_w    = buf_by_size(in_bufs,  12);
  const LL_Buffer_InfoTypeDef *b_mos  = buf_by_size(out_bufs, 12);
  const LL_Buffer_InfoTypeDef *b_grad = buf_by_size(out_bufs, LOSS_WAV_LEN * 4);

  if (!b_wav || !b_w || !b_mos || !b_grad) {
    printf("FATAL: buffer lookup failed (wav=%p w=%p mos=%p grad=%p)\r\n",
           (void *)b_wav, (void *)b_w, (void *)b_mos, (void *)b_grad);
    while (1) {}
  }
  printf("buffers: wav@%p w@%p mos@%p grad@%p%s\r\n",
         (void *)LL_Buffer_addr_start(b_wav), (void *)LL_Buffer_addr_start(b_w),
         (void *)LL_Buffer_addr_start(b_mos), (void *)LL_Buffer_addr_start(b_grad),
         (LL_Buffer_addr_start(b_wav) == LL_Buffer_addr_start(b_grad))
             ? "  [wav/grad ALIASED]" : "");

  for (int c = 0; c < LOSS_N_CLIPS; c++) {
    float *wav  = (float *)LL_Buffer_addr_start(b_wav);
    float *wvec = (float *)LL_Buffer_addr_start(b_w);

    memcpy(wav, loss_wavs[c], LOSS_WAV_LEN * sizeof(float));
    memcpy(wvec, loss_w, sizeof(loss_w));
    printf("\r\n-- clip %d: fed wav[0..3] =", c);
    for (int i = 0; i < 4; i++) { printf(" "); pf("w", wav[i]); }
    printf("\r\n");

    /* Flush what the CPU wrote so the NPU sees it; invalidate the NPU cache. */
    LL_ATON_Cache_MCU_Clean_Invalidate_Range((uintptr_t)LL_Buffer_addr_start(b_wav),
                                             LL_Buffer_len(b_wav));
    LL_ATON_Cache_MCU_Clean_Invalidate_Range((uintptr_t)LL_Buffer_addr_start(b_w),
                                             LL_Buffer_len(b_w));
#ifdef USE_NPU_CACHE
    npu_cache_invalidate();
#endif

    uint32_t t0 = dwt();
    LL_ATON_RT_RetValues_t rt;
    LL_ATON_RT_Init_Network(&NN_Instance_network);
    do {
      rt = LL_ATON_RT_RunEpochBlock(&NN_Instance_network);
      if (rt == LL_ATON_RT_WFE) LL_ATON_OSAL_WFE();
    } while (rt != LL_ATON_RT_DONE);
    uint32_t cycles = dwt() - t0;
    LL_ATON_RT_DeInit_Network(&NN_Instance_network);

    /* Invalidate so the CPU reads what the NPU wrote, not stale cache lines. */
    LL_ATON_Cache_MCU_Invalidate_Range((uintptr_t)LL_Buffer_addr_start(b_mos),
                                       LL_Buffer_len(b_mos));
    LL_ATON_Cache_MCU_Invalidate_Range((uintptr_t)LL_Buffer_addr_start(b_grad),
                                       LL_Buffer_len(b_grad));

    const float *mos  = (const float *)LL_Buffer_addr_start(b_mos);
    const float *grad = (const float *)LL_Buffer_addr_start(b_grad);

    printf("   MOS  ");
    pf("SIG", mos[0]); printf("  "); pf("BAK", mos[1]); printf("  "); pf("OVRL", mos[2]);
    printf("   (%lu cycles, %lu ms @800MHz)\r\n",
           (unsigned long)cycles, (unsigned long)(cycles / 800000u));
    print_f32_stats("   GRAD", grad, LOSS_WAV_LEN);
    printf("   grad[0..7] bits =");
    for (int i = 0; i < 8; i++) {
      union { float f; uint32_t u; } c2 = { .f = grad[i] };
      printf(" 0x%08lx", (unsigned long)c2.u);
    }
    printf("\r\n");
  }
  LL_ATON_RT_RuntimeDeInit();
  printf("\r\n=== done ===\r\n");
  while (1) {}
}
