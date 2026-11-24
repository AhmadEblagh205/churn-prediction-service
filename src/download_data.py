"""Download and verify the public IBM Telco customer-churn sample.

The raw file is intentionally git-ignored. CI and new clones reproduce the
exact input by downloading it from IBM's archived public repository and
checking the known SHA-256 before writing it to ``data/raw``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


def main():
    with urlopen(URL, timeout=30) as fh:
        content = fh.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "Downloaded dataset checksum mismatch: "
            f"expected {EXPECTED_SHA256}, got {digest}"
        )
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(content)
    print(f"Saved verified IBM sample -> {DEST} ({digest})")


if __name__ == "__main__":
    main()
