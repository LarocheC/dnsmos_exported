#!/usr/bin/env python3
"""Measured room impulse responses for the shift testbeds.

The synthetic RIRs (exponentially-decaying noise, `reverb_adapt_eval.make_rir`)
were enough to create a distribution shift, but any reviewer will ask whether
the preset/router/adaptation results survive *real* rooms. This loads the MIT
IR Survey (Traer & McDermott, PNAS 2016): 270 measured single-channel IRs from
real spaces — living rooms, hallways, stairwells, streets.

    https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip

Files are mono 32 kHz; they are lowpassed and decimated to 16 kHz here (no
scipy dependency), leading silence before the direct path is trimmed, tails
are truncated to 1 s, and each IR is normalized to unit energy — the same
convention as the synthetic generator, so gains are comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(
    "/tmp/claude-1000/-home-claroche-dnsmos-exported/"
    "dec8099b-3421-445d-829f-32a9edb3a984/scratchpad/rir/mit/Audio")
MAX_LEN_S = 1.0


def _resample_2to1(x: np.ndarray) -> np.ndarray:
    """32 kHz -> 16 kHz: windowed-sinc lowpass at the new Nyquist, then take
    every second sample."""
    taps = 101
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(0.5 * n) * np.hamming(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same")[::2]


def load_rirs(rir_dir: Path | str = DEFAULT_DIR, sr: int = 16000,
              max_len_s: float = MAX_LEN_S) -> list[np.ndarray]:
    import soundfile as sf

    rir_dir = Path(rir_dir)
    out = []
    for f in sorted(rir_dir.glob("*.wav")):
        x, fsr = sf.read(str(f), dtype="float64")
        if x.ndim > 1:
            x = x[:, 0]
        if fsr == 2 * sr:
            x = _resample_2to1(x)
        elif fsr != sr:
            continue                        # only 32k and 16k sources expected
        # trim to the direct path: first sample within 20 dB of the peak
        peak = np.abs(x).max()
        if peak <= 0:
            continue
        start = int(np.argmax(np.abs(x) >= 0.1 * peak))
        x = x[start: start + int(max_len_s * sr)]
        e = np.sqrt(np.sum(x ** 2))
        if e < 1e-9 or len(x) < sr // 100:
            continue
        out.append(x / e)
    if not out:
        raise FileNotFoundError(
            f"no usable RIRs under {rir_dir} — download "
            "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip and unzip there")
    return out


if __name__ == "__main__":
    bank = load_rirs()
    lens = np.array([len(r) / 16000 for r in bank])
    # crude rt60 proxy: time for the Schroeder integral to fall 60 dB
    rts = []
    for r in bank:
        edc = np.cumsum(r[::-1] ** 2)[::-1]
        edc = 10 * np.log10(edc / edc[0] + 1e-12)
        idx = np.nonzero(edc <= -60)[0]
        rts.append(idx[0] / 16000 if len(idx) else len(r) / 16000)
    rts = np.array(rts)
    print(f"{len(bank)} RIRs; length {lens.min():.2f}..{lens.max():.2f} s; "
          f"rt60* median {np.median(rts):.2f} s, 10-90% "
          f"{np.percentile(rts,10):.2f}-{np.percentile(rts,90):.2f} s")
