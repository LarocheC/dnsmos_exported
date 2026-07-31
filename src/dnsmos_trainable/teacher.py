"""Official-runner semantics and an ORT-backed teacher for distillation."""

from pathlib import Path

import numpy as np

from dnsmos_trainable.constants import (
    INPUT_LEN,
    POLY_BAK,
    POLY_OVR,
    POLY_SIG,
    RUNNER_HOP_SECONDS,
    SR,
)


def poly_map(raw: np.ndarray) -> np.ndarray:
    """Official polynomial mapping, raw [.., 3] -> mos [.., 3]."""
    out = np.empty_like(raw)
    for i, (c2, c1, c0) in enumerate((POLY_SIG, POLY_BAK, POLY_OVR)):
        out[..., i] = c2 * raw[..., i] ** 2 + c1 * raw[..., i] + c0
    return out


def tile_to_min_length(audio: np.ndarray, min_len: int = INPUT_LEN) -> np.ndarray:
    """Repeat-tile like the official runner until the clip covers one segment."""
    while len(audio) < min_len:
        audio = np.concatenate([audio, audio])
    return audio


class OnnxTeacher:
    """Raw-score oracle over an ONNX DNSMOS model (official or exported)."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.n_outputs = len(self.sess.get_outputs())

    def raw_scores(self, wav_batch: np.ndarray) -> np.ndarray:
        out = self.sess.run(None, {self.input_name: wav_batch.astype(np.float32)})
        return out[0]

    def official_score(self, audio: np.ndarray, sr: int) -> dict:
        """Full official-runner aggregate for one clip.

        Resample to 16 kHz, tile to >= 9.01 s, slide 9.01 s windows at 1 s hop,
        average raw scores' polynomial mappings over windows.
        """
        if sr != SR:
            import soxr

            audio = soxr.resample(audio, sr, SR)
        audio = tile_to_min_length(np.asarray(audio, dtype=np.float32))
        hop = int(RUNNER_HOP_SECONDS * SR)
        num_hops = int(np.floor(len(audio) / SR) - INPUT_LEN / SR) + 1
        segs = []
        for h in range(max(num_hops, 1)):
            seg = audio[h * hop : h * hop + INPUT_LEN]
            if len(seg) < INPUT_LEN:
                break
            segs.append(seg)
        raw = self.raw_scores(np.stack(segs))
        mos = poly_map(raw)
        return {
            "raw": raw.mean(axis=0).tolist(),
            "mos": mos.mean(axis=0).tolist(),
            "num_hops": len(segs),
        }
