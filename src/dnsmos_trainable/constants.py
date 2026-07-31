"""Single home for every magic number of the DNSMOS P.835 pipeline.

All values mirror the official Microsoft `sig_bak_ovr.onnx` graph and
`dnsmos_local.py` runner from microsoft/DNS-Challenge.
"""

import math

SR = 16000
"""Sample rate expected by the model."""

INPUT_LEN = 144160
"""Waveform samples per segment: 9.01 s @ 16 kHz."""

WIN = 320
"""Analysis window length in samples (20 ms)."""

HOP = 160
"""Hop length in samples (10 ms)."""

N_FRAMES = 900
"""Frames per segment: (INPUT_LEN - WIN) // HOP + 1."""

N_ROWS = 901
"""Rows of the STM32N6 device I/O layout: INPUT_LEN == N_ROWS * HOP."""

N_BINS = 161
"""Output bins of the trained stft-real/stft-imag projections."""

EPS = 1e-12
"""Floor applied to the power spectrogram before log10."""

LN10 = math.log(10.0)

# np.poly1d coefficient order (highest degree first), from dnsmos_local.py.
POLY_SIG = (-0.08397278, 1.22083953, 0.0052439)
POLY_BAK = (-0.13166888, 1.60915514, -0.39604546)
POLY_OVR = (-0.06766283, 1.11546468, 0.04602535)

OFFICIAL_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/"
    "master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
OFFICIAL_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
OFFICIAL_SIZE = 1157965

FC3_BIAS_FINGERPRINT = (0.33885, 0.40790, 0.32571)
"""Final Dense bias of the official model; hard-fail transplant if it differs."""

RUNNER_HOP_SECONDS = 1.0
"""Sliding-window hop of the official file-level runner."""
