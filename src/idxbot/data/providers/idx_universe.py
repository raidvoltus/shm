"""
Dynamic IDX active-universe discovery.

Terminology: ALL ACTIVE IDX SYMBOLS AVAILABLE FROM PROVIDER
— not a claim of exhaustive official BEI membership if upstream is incomplete.

Sources (deterministic order):
1. Optional env IDXBOT_UNIVERSE_URL
2. Yahoo Finance screener for exchange JKT
3. Expanded liquid seed fallback (never 5-only)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

import requests

from idxbot.data.providers.base import SymbolInfo
from idxbot.data.providers.yahoo import to_canonical_symbol

logger = logging.getLogger(__name__)

USER_AGENT = "idxbot-universe/1.0 (+https://github.com/raidvoltus/shm)"

_SEED_BARE: tuple[str, ...] = (
    "BBCA", "BBRI", "BMRI", "TLKM", "ASII", "BBNI", "BRIS", "UNVR", "ICBP",
    "INDF", "KLBF", "ADRO", "PTBA", "ITMG", "ANTM", "INCO", "MDKA", "TPIA", "CPIN",
    "JPFA", "SMGR", "INTP", "WIKA", "WSKT", "PGAS", "EXCL", "ISAT", "TOWR", "TBIG",
    "GGRM", "HMSP", "MNCN", "SCMA", "EMTK", "GOTO", "BUKA", "ACES", "MAPI", "ERAA",
    "MIKA", "HEAL", "SILO", "SIDO", "KAEF", "DVLA", "PYFA", "BRPT", "AKRA", "MEDC",
    "ELSA", "DOID", "HRUM", "BYAN", "KKGI", "BUMI", "INDY", "TOBA", "PTRO",
    "SMRA", "CTRA", "BSDE", "PWON", "DMAS", "JRPT", "ASRI", "APLN", "SMDM", "LPCK",
    "BJTM", "BJBR", "BNGA", "MEGA", "NISP", "BNII", "BBTN", "PNBN", "ARTO", "BBSI",
    "AMRT", "MIDI", "LPPF", "RALS", "CSAP", "MAPA", "HRTA", "ULTJ", "MYOR", "ROTI",
    "CLEO", "DLTA", "TBLA", "SSMS", "LSIP", "AALI", "SIMP", "PALM", "ANJT", "SGRO",
    "JSMR", "CMNP", "META", "BALI", "BIRD", "TAXI", "ASSA", "TMAS", "BULL",
    "INKP", "TKIM", "FASW", "SPMA", "WOOD", "TIRT", "IPOL", "BRNA", "ESIP",
)


def _normalize_many(raw: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in raw:
        s = (s or "").strip().upper()
        if not s:
            continue
        bare = s.replace(".JK", "").replace(".IDX", "").replace("IJ", "")
        bare = re.sub(r"[^A-Z0-9]", "", bare)
        if not bare or len(bare) > 8:
            continue
        can = to_canonical_symbol(bare)
        if can in seen:
            continue
        seen.add(can)
        out.append(can)
    out.sort()
    return out


def _from_env_url(session: requests.Session, timeout: float) -> list[str]:
    url = os.environ.get("IDXBOT_UNIVERSE_URL", "").strip()
    if not url:
        return []
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "json" in ct or r.text.strip().startswith(("[", "{")):
            data = r.json()
            if isinstance(data, list):
                items = [str(x.get("symbol", x) if isinstance(x, dict) else x) for x in data]
            elif isinstance(data, dict) and "symbols" in data:
                items = [str(x) for x in data["symbols"]]
            else:
                items = []
        else:
            items = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
        return _normalize_many(items)
    except Exception as e:
        logger.warning("universe_env_url_failed", extra={"error": type(e).__name__})
        return []


def _from_yahoo_screener(session: requests.Session, timeout: float) -> list[str]:
    url = "https://query1.finance.yahoo.com/v1/finance/screener"
    body = {
        "size": 250,
        "offset": 0,
        "sortField": "intradaymarketcap",
        "sortType": "DESC",
        "quoteType": "EQUITY",
        "query": {
            "operator": "and",
            "operands": [
                {"operator": "eq", "operands": ["region", "id"]},
                {"operator": "eq", "operands": ["exchange", "JKT"]},
            ],
        },
    }
    try:
        r = session.post(url, json=body, timeout=timeout)
        if r.status_code != 200:
            return _from_yahoo_search(session, timeout)
        data = r.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        symbols = []
        for q in quotes:
            sym = q.get("symbol") or q.get("ticker")
            if sym:
                symbols.append(str(sym))
        return _normalize_many(symbols)
    except Exception as e:
        logger.warning("universe_yahoo_screener_failed", extra={"error": type(e).__name__})
        return _from_yahoo_search(session, timeout)


def _from_yahoo_search(session: requests.Session, timeout: float) -> list[str]:
    out: list[str] = []
    try:
        for q in ("BBCA.JK", "bank IDX", "IDX stock"):
            url = "https://query1.finance.yahoo.com/v1/finance/search"
            r = session.get(url, params={"q": q, "quotesCount": 50, "newsCount": 0}, timeout=timeout)
            if r.status_code != 200:
                continue
            data = r.json()
            for item in data.get("quotes", []):
                sym = item.get("symbol", "")
                if sym.endswith(".JK") or item.get("exchange") in ("JKT", "JK", "IDX"):
                    out.append(sym)
    except Exception as e:
        logger.warning("universe_yahoo_search_failed", extra={"error": type(e).__name__})
    return _normalize_many(out)


def discover_idx_symbols(*, timeout: float = 20.0, session: Optional[requests.Session] = None) -> dict[str, Any]:
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    candidates: list[str] = []
    source = "seed_fallback"
    env_syms = _from_env_url(sess, timeout)
    if env_syms:
        candidates = env_syms
        source = "env_url"
    else:
        y = _from_yahoo_screener(sess, timeout)
        if len(y) >= 20:
            candidates = y
            source = "yahoo_screener"
        elif y:
            candidates = _normalize_many(y + list(_SEED_BARE))
            source = "yahoo_partial+seed"
        else:
            candidates = _normalize_many(list(_SEED_BARE))
            source = "seed_fallback"
    valid: list[str] = []
    excluded: list[str] = []
    for s in candidates:
        bare = s.replace(".JK", "")
        if not bare or not re.match(r"^[A-Z0-9]{3,8}$", bare):
            excluded.append(s)
            continue
        valid.append(s)
    logger.info(
        "universe_discovered",
        extra={"discovered": len(candidates), "valid": len(valid), "excluded": len(excluded), "source": source},
    )
    return {"symbols": valid, "source": source, "discovered": len(candidates), "valid": len(valid), "excluded": len(excluded)}


def symbols_as_info(symbols: list[str]) -> list[SymbolInfo]:
    return [SymbolInfo(symbol=s, exchange="IDX", status="ACTIVE", listing_status="LISTED") for s in symbols]
