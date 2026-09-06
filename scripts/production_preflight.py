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
    if os.getenv("IDXBOT_STATE_BACKEND", "local").strip().lower() != "s3":
        errors.append("IDXBOT_STATE_BACKEND must be s3 in production")
    required = ["IDXBOT_S3_ENDPOINT", "IDXBOT_S3_BUCKET", "IDXBOT_S3_ACCESS_KEY", "IDXBOT_S3_SECRET_KEY", "IDXBOT_S3_REGION"]
    for key in required:
        if not os.getenv(key):
            errors.append(f"missing required production secret/env: {key}")
    if errors:
        for e in errors:
            print(f"PRODUCTION PREFLIGHT: FAIL: {e}", file=sys.stderr)
        return 1
    print("PRODUCTION PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
