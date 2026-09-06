"""Production preflight checks (no S3). Fail-closed."""

from __future__ import annotations

import os
import sys


def main() -> int:
    errors: list[str] = []
    if os.getenv("LIVE_TRADING", "false").strip().lower() in {"1", "true", "yes", "on"}:
        errors.append("LIVE_TRADING must be false")
    if os.getenv("PAPER_TRADING", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        errors.append("PAPER_TRADING must be true")
    if os.getenv("IDXBOT_USE_FIXTURE", "false").strip().lower() in {"1", "true", "yes", "on"}:
        errors.append("IDXBOT_USE_FIXTURE must be false in production")
    s3_keys = [
        "IDXBOT_S3_ENDPOINT",
        "IDXBOT_S3_BUCKET",
        "IDXBOT_S3_ACCESS_KEY",
        "IDXBOT_S3_SECRET_KEY",
        "IDXBOT_S3_REGION",
        "IDXBOT_S3_PREFIX",
    ]
    for key in s3_keys:
        if os.getenv(key):
            errors.append(f"S3 is disabled; unset {key}")
    backend = os.getenv("IDXBOT_STATE_BACKEND", "local").strip().lower()
    if backend == "s3":
        errors.append("IDXBOT_STATE_BACKEND=s3 is not supported (local only)")
    if errors:
        for e in errors:
            print(f"PRODUCTION PREFLIGHT: FAIL: {e}", file=sys.stderr)
        return 1
    print("PRODUCTION PREFLIGHT: PASS (local atomic storage, no S3, paper-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
