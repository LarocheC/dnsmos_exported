"""Pluggable audio datasets for distillation and calibration."""

from pathlib import Path

import numpy as np
import torch

from dnsmos_trainable.constants import INPUT_LEN, SR
from dnsmos_trainable.teacher import tile_to_min_length


def load_wav_16k(path: str | Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        import soxr

        audio = soxr.resample(audio, sr, SR)
    return audio.astype(np.float32)


class FileListAudioDataset(torch.utils.data.Dataset):
    """Fixed-length 9.01 s segments from a text file of wav paths.

    The list file has one path per line (absolute or relative to the list's
    directory). Users point it at LibriSpeech, DNS-Challenge noisy clips, or
    their own corpus.
    """

    def __init__(self, list_file: str | Path, train: bool = True, seed: int = 0) -> None:
        list_file = Path(list_file)
        base = list_file.parent
        self.paths = []
        for line in list_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            self.paths.append(p if p.is_absolute() else base / p)
        if not self.paths:
            raise ValueError(f"no audio paths in {list_file}")
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        audio = tile_to_min_length(load_wav_16k(self.paths[idx]))
        if self.train and len(audio) > INPUT_LEN:
            start = int(self.rng.integers(0, len(audio) - INPUT_LEN + 1))
        else:
            start = (len(audio) - INPUT_LEN) // 2
        return torch.from_numpy(audio[start : start + INPUT_LEN].copy())
