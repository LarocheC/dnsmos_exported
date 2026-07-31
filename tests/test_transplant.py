import numpy as np
import torch

from dnsmos_trainable.constants import FC3_BIAS_FINGERPRINT, N_BINS, WIN
from dnsmos_trainable.transplant import extract_params, load_official


def test_state_dict_complete(transplanted):
    sd = transplanted.state_dict()
    assert sd["frontend.stft.w_re"].shape == (N_BINS, WIN)
    assert sd["frontend.stft.w_im"].shape == (N_BINS, WIN)
    assert sd["body.conv1.weight"].shape == (128, 1, 3, 3)
    assert sd["body.conv7.weight"].shape == (64, 32, 3, 3)
    assert sd["body.fc1.weight"].shape == (128, 64)
    assert sd["body.fc3.weight"].shape == (3, 64)


def test_fc3_bias_fingerprint(official_path):
    params = extract_params(load_official(official_path))
    assert np.allclose(params["fc3.bias"], FC3_BIAS_FINGERPRINT, atol=1e-4)


def test_stft_kernels_are_not_a_pure_dft(transplanted):
    """Guard against 'helpfully' replacing the trained matrices with a clean
    (windowed) DFT.

    The trained kernels are CLOSE to a hann-windowed DFT (0.93-0.98 per-bin
    cosine on mid-band bins; 0.995 at the Nyquist bin) — they look
    DFT-initialized and mildly trained — but they are not exactly one, and
    exact parity with the official model requires the verbatim weights. A
    substituted pure windowed DFT scores ~1.000 on every bin, so mid-band
    bins with a 0.98 threshold separate the two cleanly (real max ~0.956).
    """
    n = np.arange(WIN)
    windows = {
        "rect": np.ones(WIN),
        "hann": np.hanning(WIN),
        "hamming": np.hamming(WIN),
    }
    mats = {
        "w_re": transplanted.frontend.stft.w_re.detach().numpy(),
        "w_im": transplanted.frontend.stft.w_im.detach().numpy(),
    }
    worst, worst_desc = 0.0, ""
    for mat_name, mat in mats.items():
        for k in (40, 80, 120, 150, 159):
            row = mat[k]
            for win_name, win in windows.items():
                for basis in (np.cos(2 * np.pi * k * n / WIN) * win,
                              np.sin(2 * np.pi * k * n / WIN) * win):
                    denom = np.linalg.norm(row) * np.linalg.norm(basis)
                    if denom < 1e-9:
                        continue
                    cos_sim = abs(np.dot(row, basis)) / denom
                    if cos_sim > worst:
                        worst, worst_desc = cos_sim, f"{mat_name}[{k}] ~ {win_name}"
    assert worst < 0.98, (
        f"stft kernels look like a pure {worst_desc} DFT ({worst:.3f}) — transplant is wrong"
    )
