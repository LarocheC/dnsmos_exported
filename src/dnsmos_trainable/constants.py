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


class DnsmosConfig:
    """Geometry + widths of a DNSMOS-topology model.

    The official model is the default. A distilled *student* may shrink any of
    these while keeping the same op topology (7 convs, 3 pools, global max,
    3 dense) — which is what lets `DnsmosLossGraph`'s hand-written backward
    apply unchanged, no matter the size.

    The three levers that move the on-chip activation peak (= conv1's output,
    ``c1 * n_frames * n_bins``) are `input_len`, `n_bins` and `c1`; they
    multiply.
    """

    __slots__ = ("input_len", "n_bins", "channels", "fc", "win", "hop")

    def __init__(self, input_len=INPUT_LEN, n_bins=N_BINS,
                 channels=(128, 64, 64, 32, 32, 32, 64), fc=(128, 64),
                 win=WIN, hop=HOP):
        if (input_len - win) % hop:
            raise ValueError(f"input_len {input_len} is not win+k*hop for win={win}, hop={hop}")
        self.input_len = int(input_len)
        self.n_bins = int(n_bins)
        self.channels = tuple(int(c) for c in channels)
        self.fc = tuple(int(c) for c in fc)
        self.win = int(win)
        self.hop = int(hop)
        if len(self.channels) != 7:
            raise ValueError("channels must have 7 entries (conv1..conv7)")

    @property
    def n_frames(self) -> int:
        return (self.input_len - self.win) // self.hop + 1

    @property
    def n_rows(self) -> int:
        """Rows of the STM32N6 [1, n_rows, hop] I/O layout."""
        return self.input_len // self.hop

    @property
    def seconds(self) -> float:
        return self.input_len / SR

    @property
    def peak_activation_elems(self) -> int:
        """conv1's output — the graph's largest tensor."""
        return self.channels[0] * self.n_frames * self.n_bins

    def __repr__(self) -> str:
        return (f"DnsmosConfig({self.seconds:.2f}s, T={self.n_frames}, bins={self.n_bins}, "
                f"ch={self.channels}, fc={self.fc})")


OFFICIAL_CONFIG = DnsmosConfig()
