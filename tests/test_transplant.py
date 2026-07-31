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


def test_stft_kernels_are_not_a_dft(transplanted):
    """Guard: the trained stft kernels must never be 'helpfully' replaced by a DFT."""
    w_re = transplanted.frontend.stft.w_re.numpy()
    n = np.arange(WIN)
    worst = 0.0
    for k in (1, 40, 80, 120, 160):
        ideal = np.cos(2 * np.pi * k * n / WIN)
        row = w_re[k]
        cos_sim = abs(np.dot(row, ideal)) / (np.linalg.norm(row) * np.linalg.norm(ideal))
        worst = max(worst, cos_sim)
    assert worst < 0.95, "stft kernels look like an ideal DFT — transplant is wrong"
