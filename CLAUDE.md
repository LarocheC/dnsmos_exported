# dnsmos_exported: agent notes

## STM32N6 deployment

Before debugging anything on the STM32N6570-DK (ST Edge AI Core, the Neural-ART
NPU, signing and flashing, memory placement), read the hub:
[stm32n6-deployment-zoo/KNOWLEDGE.md](https://github.com/LarocheC/stm32n6-deployment-zoo/blob/main/KNOWLEDGE.md).
From a checkout of that repo, `uv run zoo atlas <words>` searches its failure atlas
and `uv run zoo atlas --classify <log>` matches a build or flash log against it.
Findings about the part, the toolchain or the board belong there, as atlas entries
with sources; findings about this repo's models stay here.
