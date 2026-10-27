"""Verify the archived reviewer-experiment files against their SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reviewer_experiments" / "reviewer_test_data_20261025"
CHECKSUMS = PACKAGE / "checksums.sha256"
MANIFEST = PACKAGE / "MANIFEST.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    rows = []
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, relative = line.split(maxsplit=1)
            rows.append((expected, Path(relative)))

    missing = []
    mismatched = []
    total_bytes = 0
    for expected, relative in rows:
        path = PACKAGE / relative
        if not path.is_file():
            missing.append(str(relative))
            continue
        total_bytes += path.stat().st_size
        actual = file_sha256(path)
        if actual != expected:
            mismatched.append({"path": str(relative), "expected": expected, "actual": actual})

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_count = int(manifest["file_count"])
    result = {
        "status": "PASS" if not missing and not mismatched and len(rows) == expected_count else "FAIL",
        "files": len(rows),
        "bytes": total_bytes,
        "missing": missing,
        "mismatched": mismatched,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
