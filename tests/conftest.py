import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dnsmos_trainable.constants import OFFICIAL_SHA256  # noqa: E402


@pytest.fixture(scope="session")
def official_path() -> Path:
    path = ROOT / "models" / "sig_bak_ovr.onnx"
    if not path.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            from download_official import download

            download(path)
        except Exception as exc:  # pragma: no cover - offline environments
            pytest.skip(f"official model unavailable and download failed: {exc}")
    import hashlib

    if hashlib.sha256(path.read_bytes()).hexdigest() != OFFICIAL_SHA256:
        pytest.fail("official model sha256 mismatch — refusing to test against it")
    return path


@pytest.fixture(scope="session")
def transplanted(official_path):
    from dnsmos_trainable.model import DnsmosModel
    from dnsmos_trainable.transplant import transplant

    model = DnsmosModel()
    missing, unexpected = model.load_state_dict(transplant(official_path), strict=False)
    assert not unexpected
    assert all(k.startswith("poly.") for k in missing)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture(scope="session")
def synthetic_batch():
    from dnsmos_trainable.verify import make_synthetic_batch

    return make_synthetic_batch(8, seed=0)
