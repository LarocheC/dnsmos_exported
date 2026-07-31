import os

import numpy as np
import pytest
import torch

from dnsmos_trainable.model import DnsmosModel
from dnsmos_trainable.verify import compare_intermediates, compare_raw_scores, ort_session


def test_raw_score_parity_fp32(transplanted, official_path, synthetic_batch):
    diff = compare_raw_scores(transplanted, ort_session(official_path), synthetic_batch)
    # Cross-engine fp32 comparison carries both engines' accumulation noise.
    assert diff < 5e-5, f"fp32 parity {diff:.2e}"


def test_raw_score_parity_fp64(transplanted, official_path, synthetic_batch):
    """Exactness: in fp64 the only residual is ORT's own fp32 rounding."""
    m64 = DnsmosModel().double()
    m64.load_state_dict(
        {k: v.double() for k, v in transplanted.state_dict().items()}, strict=False
    )
    m64.eval()
    with torch.no_grad():
        raw64, _ = m64(torch.from_numpy(synthetic_batch).double())
    ref = ort_session(official_path).run(None, {"input_1": synthetic_batch})[0]
    diff = float(np.abs(raw64.numpy() - ref).max())
    assert diff < 1e-5, f"fp64 parity {diff:.2e}"


def test_stagewise_parity(transplanted, official_path, synthetic_batch):
    stages = compare_intermediates(official_path, transplanted, synthetic_batch)
    assert stages["frames"] == 0.0
    for name, diff in stages.items():
        assert diff < 1e-3, f"stage {name}: {diff:.2e}"


def test_spatial_trace(transplanted, synthetic_batch):
    """Keras 'valid' pooling floors odd dims: (900,161)->(450,80)->(225,40)->(112,20)."""
    body = transplanted.body
    x = transplanted.frontend(torch.from_numpy(synthetic_batch[:1]))
    x = torch.relu(body.conv4(torch.relu(body.conv3(torch.relu(body.conv2(torch.relu(body.conv1(x))))))))
    x = body.pool(x)
    assert x.shape[2:] == (450, 80)
    x = body.pool(torch.relu(body.conv5(x)))
    assert x.shape[2:] == (225, 40)
    x = body.pool(torch.relu(body.conv6(x)))
    assert x.shape[2:] == (112, 20)


def test_real_audio_parity_optional(transplanted, official_path):
    audio_dir = os.environ.get("DNSMOS_TEST_AUDIO")
    if not audio_dir:
        pytest.skip("set DNSMOS_TEST_AUDIO to a dir of wavs for real-audio parity")
    import soundfile as sf

    from dnsmos_trainable.constants import INPUT_LEN, SR

    from pathlib import Path

    wavs = sorted(Path(audio_dir).glob("*.wav"))[:5]
    assert wavs, f"no wavs in {audio_dir}"
    sess = ort_session(official_path)
    for path in wavs:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        assert sr == SR, "resample test fixtures to 16 kHz first"
        while len(audio) < INPUT_LEN:
            audio = np.concatenate([audio, audio])
        seg = audio[:INPUT_LEN][None].astype(np.float32)
        diff = compare_raw_scores(transplanted, sess, seg)
        assert diff < 5e-5, f"{path.name}: {diff:.2e}"
