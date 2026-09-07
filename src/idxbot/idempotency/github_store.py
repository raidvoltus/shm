"""
GitHub Contents API ledger store on branch idxbot-state.

Uses GITHUB_TOKEN only (Actions built-in). No PAT required if permissions.contents: write.
Never logs token.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from idxbot.idempotency.ledger import (
    SCHEMA_VERSION,
    FileLedgerStore,
    LedgerUnavailable,
    MemoryLedgerStore,
)

logger = logging.getLogger(__name__)

DEFAULT_BRANCH = "idxbot-state"
DEFAULT_PATH = "idempotency/ledger.json"
API = "https://api.github.com"


class GitHubLedgerStore:
    """
    Read/write ledger.json via GitHub Contents API.

    Atomicity: single-file SHA optimistic concurrency.
    On 409 conflict → LedgerUnavailable (caller fail-closed).
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        branch: str = DEFAULT_BRANCH,
        path: str = DEFAULT_PATH,
    ) -> None:
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN", "")
        repo_full = os.environ.get("GITHUB_REPOSITORY", "")
        if owner and repo:
            self.owner, self.repo = owner, repo
        elif "/" in repo_full:
            self.owner, self.repo = repo_full.split("/", 1)
        else:
            self.owner = owner or os.environ.get("GITHUB_REPOSITORY_OWNER", "")
            self.repo = repo or ""
        self.branch = branch or os.environ.get("IDXBOT_LEDGER_BRANCH", DEFAULT_BRANCH)
        self.path = path or os.environ.get("IDXBOT_LEDGER_PATH", DEFAULT_PATH)
        self._sha: Optional[str] = None

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise LedgerUnavailable("GITHUB_TOKEN missing")
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "idxbot-ledger",
        }

    def _url(self) -> str:
        if not self.owner or not self.repo:
            raise LedgerUnavailable("GITHUB_REPOSITORY not set")
        return f"{API}/repos/{self.owner}/{self.repo}/contents/{self.path}?ref={self.branch}"

    def _put_url(self) -> str:
        if not self.owner or not self.repo:
            raise LedgerUnavailable("GITHUB_REPOSITORY not set")
        return f"{API}/repos/{self.owner}/{self.repo}/contents/{self.path}"

    def load(self) -> dict[str, Any]:
        req = urllib.request.Request(self._url(), headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self._sha = None
                return {"schema_version": SCHEMA_VERSION, "entries": []}
            raise LedgerUnavailable(f"github ledger load HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise LedgerUnavailable(f"github ledger load failed: {type(e).__name__}") from e

        self._sha = body.get("sha")
        try:
            raw = base64.b64decode(body.get("content", "")).decode("utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except (ValueError, json.JSONDecodeError) as e:
            raise LedgerUnavailable(f"github ledger decode failed: {type(e).__name__}") from e
        if not isinstance(data, dict):
            raise LedgerUnavailable("github ledger corrupt")
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("entries", [])
        return data

    def save(self, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        content_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        body: dict[str, Any] = {
            "message": "chore(ledger): update signal delivery idempotency",
            "content": content_b64,
            "branch": self.branch,
        }
        if self._sha:
            body["sha"] = self._sha
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._put_url(),
            data=data,
            headers={**self._headers(), "Content-Type": "application/json"},
            method="PUT",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (409, 422):
                raise LedgerUnavailable(f"github ledger conflict HTTP {e.code}") from e
            raise LedgerUnavailable(f"github ledger save HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise LedgerUnavailable(f"github ledger save failed: {type(e).__name__}") from e
        content = result.get("content") or {}
        self._sha = content.get("sha") or self._sha


def build_ledger_store():
    """
    Select store from environment.

    IDXBOT_LEDGER_BACKEND=github|file|memory
    Default: github if GITHUB_TOKEN+GITHUB_REPOSITORY set, else memory.
    """
    backend = os.environ.get("IDXBOT_LEDGER_BACKEND", "").strip().lower()
    if not backend:
        if os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPOSITORY"):
            backend = "github"
        else:
            backend = "memory"
    if backend == "github":
        return GitHubLedgerStore()
    if backend == "file":
        path = os.environ.get("IDXBOT_LEDGER_FILE", ".state/idempotency_ledger.json")
        return FileLedgerStore(path)
    return MemoryLedgerStore()
