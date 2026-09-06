"""
S3-compatible durable StateStore (pure requests + SigV4).

Env:
  IDXBOT_STATE_BACKEND=s3
  IDXBOT_S3_ENDPOINT=https://s3.amazonaws.com   # or MinIO / R2 / etc.
  IDXBOT_S3_BUCKET=my-bucket
  IDXBOT_S3_ACCESS_KEY=...
  IDXBOT_S3_SECRET_KEY=...
  IDXBOT_S3_REGION=us-east-1
  IDXBOT_S3_PREFIX=idxbot/state/

Never logs secrets. Optimistic concurrency via state_version in object body
and If-Match-style metadata when supported; otherwise version field checked
after load.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

import requests

from idxbot.storage.backend import (
    StateCorruptionError,
    StorageBackend,
    StorageError,
    VersionConflictError,
)

logger = logging.getLogger(__name__)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _sigv4_headers(
    *,
    method: str,
    url_path: str,
    query: str,
    body: bytes,
    access_key: str,
    secret_key: str,
    region: str,
    service: str = "s3",
    host: str,
    content_type: str = "application/json",
) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    payload_hash = _sha256(body)
    canonical_headers = f"content-type:{content_type}\nhost:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(
        [method, url_path, query, canonical_headers, signed_headers, payload_hash]
    )
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, credential_scope, _sha256(canonical_request.encode())]
    )
    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), datestamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Authorization": authorization,
        "Content-Type": content_type,
        "Host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }


class S3CompatibleStorageBackend(StorageBackend):
    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        bucket: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: float = 20.0,
    ) -> None:
        self.endpoint = (endpoint or os.environ.get("IDXBOT_S3_ENDPOINT") or "").rstrip("/")
        self.bucket = bucket or os.environ.get("IDXBOT_S3_BUCKET") or ""
        self.access_key = access_key or os.environ.get("IDXBOT_S3_ACCESS_KEY") or ""
        self.secret_key = secret_key or os.environ.get("IDXBOT_S3_SECRET_KEY") or ""
        self.region = region or os.environ.get("IDXBOT_S3_REGION") or "us-east-1"
        self.prefix = (prefix or os.environ.get("IDXBOT_S3_PREFIX") or "idxbot/state/").lstrip("/")
        self.timeout = timeout
        if not all([self.endpoint, self.bucket, self.access_key, self.secret_key]):
            raise StorageError(
                "S3 backend requires IDXBOT_S3_ENDPOINT, IDXBOT_S3_BUCKET, "
                "IDXBOT_S3_ACCESS_KEY, IDXBOT_S3_SECRET_KEY"
            )

    def _object_key(self, key: str) -> str:
        safe = key.replace("..", "").replace("/", "_")
        return f"{self.prefix}{safe}.json"

    def _host(self) -> str:
        # path-style: endpoint host
        from urllib.parse import urlparse
        return urlparse(self.endpoint).netloc

    def _url(self, object_key: str) -> str:
        return f"{self.endpoint}/{self.bucket}/{object_key}"

    def load_state(self, key: str) -> Optional[dict[str, Any]]:
        object_key = self._object_key(key)
        path = f"/{self.bucket}/{object_key}"
        body = b""
        headers = _sigv4_headers(
            method="GET",
            url_path=path,
            query="",
            body=body,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region,
            host=self._host(),
        )
        try:
            resp = requests.get(self._url(object_key), headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise StorageError(f"S3 load network error: {type(e).__name__}") from e
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise StorageError(f"S3 load HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as e:
            raise StateCorruptionError(f"S3 corrupt JSON for {key}") from e
        if not isinstance(data, dict):
            raise StateCorruptionError(f"S3 state not dict for {key}")
        stored_checksum = data.get("checksum")
        if stored_checksum:
            unsigned = {k: v for k, v in data.items() if k != "checksum"}
            canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if stored_checksum != _sha256(canonical):
                raise StateCorruptionError(f"S3 checksum mismatch for {key}")
        return data

    def save_state(
        self,
        key: str,
        data: dict[str, Any],
        expected_version: Optional[int] = None,
    ) -> None:
        # optimistic concurrency on state_version field
        if expected_version is not None:
            current = self.load_state(key)
            if current is None:
                if expected_version != 0:
                    raise VersionConflictError(
                        f"version conflict key={key}: expected={expected_version} current=missing"
                    )
            else:
                cur_v = int(current.get("state_version", 0))
                if cur_v != expected_version:
                    raise VersionConflictError(
                        f"version conflict key={key}: expected={expected_version} current={cur_v}"
                    )
        payload = dict(data)
        payload.setdefault("state_version", int(payload.get("state_version", 0)))
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["checksum"] = _sha256(raw)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

        object_key = self._object_key(key)
        path = f"/{self.bucket}/{object_key}"
        headers = _sigv4_headers(
            method="PUT",
            url_path=path,
            query="",
            body=raw,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region,
            host=self._host(),
        )
        try:
            resp = requests.put(self._url(object_key), data=raw, headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise StorageError(f"S3 save network error: {type(e).__name__}") from e
        if resp.status_code not in (200, 201):
            raise StorageError(f"S3 save HTTP {resp.status_code}: {resp.text[:200]}")

    def delete_state(self, key: str) -> None:
        object_key = self._object_key(key)
        path = f"/{self.bucket}/{object_key}"
        headers = _sigv4_headers(
            method="DELETE",
            url_path=path,
            query="",
            body=b"",
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region,
            host=self._host(),
        )
        try:
            resp = requests.delete(self._url(object_key), headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise StorageError(f"S3 delete network error: {type(e).__name__}") from e
        if resp.status_code not in (200, 204, 404):
            raise StorageError(f"S3 delete HTTP {resp.status_code}")

    def exists(self, key: str) -> bool:
        try:
            return self.load_state(key) is not None
        except StorageError:
            return False


def build_storage_backend() -> StorageBackend:
    """Factory from IDXBOT_STATE_BACKEND env."""
    from idxbot.storage.backend import LocalStorageBackend

    mode = (os.environ.get("IDXBOT_STATE_BACKEND") or "local").strip().lower()
    if mode == "s3":
        return S3CompatibleStorageBackend()
    if mode in ("ephemeral", "none"):
        # still local path but caller should treat as ephemeral
        return LocalStorageBackend(os.environ.get("IDXBOT_STATE_DIR", ".state"))
    return LocalStorageBackend(os.environ.get("IDXBOT_STATE_DIR", ".state"))
