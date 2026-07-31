#!/usr/bin/env python3
"""Download the official DNSMOS P.835 sig_bak_ovr.onnx (sha256-pinned)."""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dnsmos_trainable.constants import OFFICIAL_SHA256, OFFICIAL_URL


def download(dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"{dest} already exists, verifying checksum")
    else:
        print(f"downloading {OFFICIAL_URL}")
        with urllib.request.urlopen(OFFICIAL_URL) as resp:
            data = resp.read()
        dest.write_bytes(data)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != OFFICIAL_SHA256:
        raise RuntimeError(
            f"sha256 mismatch for {dest}: got {digest}, expected {OFFICIAL_SHA256}. "
            "Upstream file may have changed; do not proceed."
        )
    print(f"ok: {dest} ({dest.stat().st_size} bytes, sha256={digest[:12]}...)")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "sig_bak_ovr.onnx",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.dest, args.force)
