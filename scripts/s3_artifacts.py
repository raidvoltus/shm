"""Durable model/experience artifact bundle for ephemeral CI runners."""
from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import quote

import requests

from idxbot.storage.s3_backend import S3CompatibleStorageBackend, _sigv4_headers

ARTIFACT_KEY = "idxbot/artifacts/latest.tar.gz"
ROOTS = (".models", ".experience")


def _backend() -> S3CompatibleStorageBackend:
    return S3CompatibleStorageBackend()


def _url(backend: S3CompatibleStorageBackend, key: str) -> str:
    return f"{backend.endpoint}/{backend.bucket}/{quote(key, safe='/')}"


def pull() -> int:
    b = _backend()
    path = f"/{b.bucket}/{quote(ARTIFACT_KEY, safe='/')}"
    headers = _sigv4_headers(
        method="GET", url_path=path, query="", body=b"", access_key=b.access_key,
        secret_key=b.secret_key, region=b.region, host=b._host(), content_type="application/gzip",
    )
    try:
        r = requests.get(_url(b, ARTIFACT_KEY), headers=headers, timeout=b.timeout)
    except requests.RequestException as e:
        raise SystemExit(f"artifact pull network error: {type(e).__name__}") from e
    if r.status_code == 404:
        print("No prior artifact bundle; starting clean.")
        return 0
    if r.status_code != 200:
        raise SystemExit(f"artifact pull failed: HTTP {r.status_code}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        tmp = Path(f.name)
        f.write(r.content)
    try:
        with tarfile.open(tmp, "r:gz") as tf:
            members = tf.getmembers()
            for m in members:
                target = (Path.cwd() / m.name).resolve()
                if target != Path.cwd().resolve() and Path.cwd().resolve() not in target.parents:
                    raise SystemExit("artifact contains unsafe path")
            tf.extractall(Path.cwd())
    finally:
        tmp.unlink(missing_ok=True)
    print("Artifact bundle restored.")
    return 0


def push() -> int:
    b = _backend()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        tmp = Path(f.name)
    try:
        with tarfile.open(tmp, "w:gz") as tf:
            for root in ROOTS:
                p = Path(root)
                if p.exists():
                    tf.add(p, arcname=root, recursive=True)
        raw = tmp.read_bytes()
        key = ARTIFACT_KEY
        path = f"/{b.bucket}/{quote(key, safe='/')}"
        headers = _sigv4_headers(
            method="PUT", url_path=path, query="", body=raw, access_key=b.access_key,
            secret_key=b.secret_key, region=b.region, host=b._host(), content_type="application/gzip",
        )
        try:
            r = requests.put(_url(b, key), headers=headers, data=raw, timeout=b.timeout)
        except requests.RequestException as e:
            raise SystemExit(f"artifact push network error: {type(e).__name__}") from e
        if r.status_code not in (200, 201):
            raise SystemExit(f"artifact push failed: HTTP {r.status_code}")
        print(f"Artifact bundle persisted ({len(raw)} bytes).")
        return 0
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("pull", "push"))
    args = ap.parse_args()
    return pull() if args.action == "pull" else push()


if __name__ == "__main__":
    raise SystemExit(main())
