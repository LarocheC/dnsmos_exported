#!/usr/bin/env bash
# Build + load + run the standalone DNSMOS loss-graph firmware on the STM32N6570-DK.
#
# Patches ST's NPU_Validation project in place (with .loss_bak backups) to call our
# orchestrator instead of the protobuf validation stack, builds the standard N6-DK
# configuration, gdb-loads the RAM image and captures UART.
#
# Why: the same compiled network returns near-constant garbage under the validation
# stack while ST's own optimized-export graph is correct in onnxruntime. This run
# tells us whether the fault is the validation stack's I/O path or the generated
# code / NPU runtime itself.
#
# Follows eco8-neaixt's deploy/stm32n6/firmware_router/build_and_run.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ST=~/stedgeai/install/4.0
VAL="$ST/Projects/STM32N6570-DK/Applications/NPU_Validation"
TC=~/toolchains/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin
CLT=~/opt/st/stm32cubeclt_1.21.0
GEN="${GEN:-$HERE/../n6_gen/dnsmos_loss_crop0p25s_preserve}"
PORT="${PORT:-61245}"
export PATH="$ST/Utilities/linux:$TC:$PATH"

[ -f "$HERE/loss_input.h" ] || { echo "loss_input.h missing — run gen_loss_input.py"; exit 1; }
[ -f "$GEN/network.c" ] || { echo "no network.c in $GEN"; exit 1; }
echo "[loss-fw] network: $GEN"

MK="$VAL/armgcc/Makefile"; MAIN="$VAL/Core/Src/main.c"
for f in "$MK" "$MAIN"; do [ -f "$f.loss_bak" ] || cp "$f" "$f.loss_bak"; done

# our orchestrator + input header + the compiled network
rm -f "$VAL/Core/Src/orchestrator.c" "$VAL/Core/Src/orchestrator_router.c"
cp "$HERE/orchestrator_loss.c" "$VAL/Core/Src/"
cp "$HERE/loss_input.h" "$VAL/Core/Inc/"
cp "$GEN/network.c" "$GEN/network.h" "$VAL/X-CUBE-AI/App/"

# redirect main() to our entry points
grep -q orchestratorInit "$MAIN" || sed -i \
  -e 's/#include "aiValidation.h"/#include "aiValidation.h"\nextern void orchestratorInit(void); extern void orchestratorProcess(void);/' \
  -e 's/aiValidationInit();/orchestratorInit();/' -e 's/aiValidationProcess();/orchestratorProcess();/' "$MAIN"
grep -q orchestrator_loss.c "$MK" || sed -i \
  's#APP_SOURCES += $(APP_PATH)/main.c#APP_SOURCES += $(APP_PATH)/main.c\nAPP_SOURCES += $(APP_PATH)/orchestrator_loss.c#' "$MK"

cd "$VAL/armgcc"
make BUILD_CONF=N6-DK GCC_PATH="$TC" -j8 2>&1 | tail -5
ELF="$VAL/armgcc/build/N6-DK/Project.elf"
arm-none-eabi-size "$ELF"

# external-memory blobs (weights) still need flashing when the profile uses them
for hex in "$GEN"/network_atonbuf.xSPI2.hex; do
  [ -f "$hex" ] || continue
  [ "${SKIP_FLASH:-0}" = "1" ] && { echo "[loss-fw] SKIP_FLASH=1, weights assumed resident"; continue; }
  echo "[loss-fw] flashing $(basename "$hex")"
  ~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI \
    -c port=SWD mode=HOTPLUG -el ~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr \
    -w "$hex" 2>&1 | tail -2
done

# Break just before the orchestrator call, then detach — detaching resumes the
# core, so the printing loop runs free while we capture UART. (Detaching from a
# halted core without a breakpoint+continue leaves it halted and silent.)
cat > /tmp/loss_load.gdb <<EOF
target remote 127.0.0.1:$PORT
monitor halt
load $ELF
set \$sp = *(unsigned int*)0x34000000
set \$pc = *(unsigned int*)0x34000004
tbreak main.c:138
commands
  detach
  quit
end
continue
EOF

pkill -x ST-LINK_gdbserver 2>/dev/null || true; sleep 1
"$CLT/STLink-gdb-server/bin/ST-LINK_gdbserver" -d --frequency 2000 --apid 1 \
  --port-number "$PORT" -cp ~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin \
  > /tmp/loss_gdbserver.log 2>&1 &
sleep 3

# capture UART from before the reset so we don't miss the banner.
# pyserial, not `cat`: the tty layer needs the line configured at 921600.
/tmp/profenv/bin/python - > /tmp/loss_uart.txt 2>/dev/null <<'PY' &
import serial, sys, time
s = serial.Serial('/dev/ttyACM0', 921600, timeout=1)
end = time.time() + 90
while time.time() < end:
    d = s.read(4096)
    if d:
        sys.stdout.write(d.decode('utf-8', 'replace')); sys.stdout.flush()
        if 'done ===' in d.decode('utf-8', 'replace'):
            break
PY
CATPID=$!
sleep 2
"$CLT/GNU-tools-for-STM32/bin/arm-none-eabi-gdb" -batch --command=/tmp/loss_load.gdb "$ELF" 2>&1 | tail -3
wait $CATPID 2>/dev/null || true
pkill -x ST-LINK_gdbserver 2>/dev/null || true

echo "===== UART ====="
cat /tmp/loss_uart.txt
