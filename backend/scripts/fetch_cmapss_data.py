"""Fetches the NASA C-MAPSS FD001 files the Dockerfile needs at build time,
pinned to a specific commit SHA (not a mutable branch ref) with a SHA-256
checksum verified before any file is used.

Found in external code review: pulling from someone else's unpinned
"master" has no integrity guarantee -- the maintainer force-pushing, the
repo being deleted/renamed, or a compromised upstream would all silently
change what gets baked into the image. The commit SHA and hashes below
were captured directly from
https://github.com/hankroark/Turbofan-Engine-Degradation (an unofficial but
widely-used mirror of NASA's public C-MAPSS data) and cross-checked against
the branch HEAD at the time this was written; see docs/day1_status.md for
how they were captured.

Usage: python3 fetch_cmapss_data.py /data/cmapss
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT_SHA = "ffde9492639af9c02ef6ab6607f8029dc8302591"
BASE_URL = f"https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/{PINNED_COMMIT_SHA}/CMAPSSData"

EXPECTED_SHA256 = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}


def main(dest_dir: str) -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for filename, expected_hash in EXPECTED_SHA256.items():
        url = f"{BASE_URL}/{filename}"
        dest_path = dest / filename
        print(f"fetching {url} -> {dest_path}")
        urllib.request.urlretrieve(url, dest_path)

        actual_hash = hashlib.sha256(dest_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"{filename}: SHA-256 mismatch -- expected {expected_hash}, got {actual_hash}. "
                "The pinned commit's content changed unexpectedly; do not proceed without "
                "re-verifying this file against NASA's own C-MAPSS distribution."
            )
        print(f"  verified sha256={actual_hash}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/data/cmapss")
