"""
Persistent paper portfolio store (repository source-of-truth).

Layout (idxbot-state branch or local dir):
  portfolio/account.json
  portfolio/positions.json
  portfolio/transactions.jsonl
  portfolio/signals.jsonl
  portfolio/state.json

Corrupt state → fail-closed (never silent reset to 10_000_000).
No secrets stored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_INITIAL = 10_000_000.0
CURRENCY = "IDR"


class PortfolioStoreError(RuntimeError):
    pass


class PortfolioCorruptError(PortfolioStoreError):
    pass


class PortfolioConflictError(PortfolioStoreError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime] = None) -> str:
    d = dt or _utcnow()
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class AccountSnapshot:
    initial_balance: float = DEFAULT_INITIAL
    cash_available: float = DEFAULT_INITIAL
    equity: float = DEFAULT_INITIAL
    currency: str = CURRENCY
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AccountSnapshot":
        return cls(
            initial_balance=float(d.get("initial_balance", DEFAULT_INITIAL)),
            cash_available=float(d["cash_available"]),
            equity=float(d.get("equity", d["cash_available"])),
            currency=str(d.get("currency", CURRENCY)),
            realized_pnl=float(d.get("realized_pnl", 0.0)),
            unrealized_pnl=float(d.get("unrealized_pnl", 0.0)),
        )


@dataclass
class PositionSnapshot:
    symbol: str
    quantity: int
    average_entry: float
    market_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    status: str = "OPEN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PositionSnapshot":
        return cls(
            symbol=str(d["symbol"]),
            quantity=int(d["quantity"]),
            average_entry=float(d["average_entry"]),
            market_price=(float(d["market_price"]) if d.get("market_price") is not None else None),
            market_value=(float(d["market_value"]) if d.get("market_value") is not None else None),
            unrealized_pnl=(float(d["unrealized_pnl"]) if d.get("unrealized_pnl") is not None else None),
            status=str(d.get("status", "OPEN")),
        )


@dataclass
class PortfolioBundle:
    account: AccountSnapshot
    positions: list[PositionSnapshot] = field(default_factory=list)
    transactions: list[dict[str, Any]] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)
    state_version: int = 1
    last_updated: str = field(default_factory=_iso)
    schema_version: int = SCHEMA_VERSION
    content_sha: str = ""

    def recompute_equity(self) -> None:
        holdings = 0.0
        unreal = 0.0
        for p in self.positions:
            if p.status != "OPEN" or p.quantity <= 0:
                continue
            px = p.market_price if p.market_price is not None else p.average_entry
            mv = p.quantity * px
            p.market_value = mv
            p.unrealized_pnl = mv - (p.quantity * p.average_entry)
            holdings += mv
            unreal += p.unrealized_pnl or 0.0
        self.account.unrealized_pnl = unreal
        self.account.equity = self.account.cash_available + holdings

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "account": self.account.to_dict(),
                "positions": [p.to_dict() for p in self.positions],
                "n_tx": len(self.transactions),
                "n_sig": len(self.signals),
                "state_version": self.state_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def validate_bundle(b: PortfolioBundle) -> None:
    if b.schema_version != SCHEMA_VERSION:
        raise PortfolioCorruptError(f"unsupported schema_version={b.schema_version}")
    if b.account.cash_available < -1e-9:
        raise PortfolioCorruptError("cash_available negative")
    if b.account.initial_balance <= 0:
        raise PortfolioCorruptError("invalid initial_balance")
    for p in b.positions:
        if p.quantity < 0:
            raise PortfolioCorruptError(f"negative qty {p.symbol}")
        if p.average_entry < 0:
            raise PortfolioCorruptError(f"negative avg {p.symbol}")


def create_initial(balance: float = DEFAULT_INITIAL) -> PortfolioBundle:
    return PortfolioBundle(
        account=AccountSnapshot(
            initial_balance=balance,
            cash_available=balance,
            equity=balance,
        ),
        positions=[],
        transactions=[],
        signals=[],
        state_version=1,
        last_updated=_iso(),
    )


class LocalPortfolioStore:
    def __init__(self, root: str | Path = ".state") -> None:
        self.root = Path(root) / "portfolio"
        self.root.mkdir(parents=True, exist_ok=True)
        self._version = 0

    def _read_json(self, name: str) -> Optional[dict[str, Any]]:
        path = self.root / name
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            if not raw.strip():
                raise PortfolioCorruptError(f"{name} empty")
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise PortfolioCorruptError(f"{name} invalid JSON") from e
        except OSError as e:
            raise PortfolioStoreError(f"read {name}: {e}") from e

    def _write_json(self, name: str, data: Any) -> None:
        path = self.root / name
        text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), prefix=".pf-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _append_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        path = self.root / name
        fd, tmp = tempfile.mkstemp(dir=str(self.root), prefix=".pf-", suffix=".tmp")
        try:
            existing = ""
            if path.exists():
                existing = path.read_text(encoding="utf-8")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write(existing + "\n")
                elif existing:
                    f.write(existing)
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.root / name
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as e:
            raise PortfolioCorruptError(f"{name} corrupt") from e
        return out

    def load(self) -> PortfolioBundle:
        state = self._read_json("state.json")
        account_d = self._read_json("account.json")
        positions_d = self._read_json("positions.json")
        if state is None and account_d is None:
            return create_initial()
        if account_d is None or positions_d is None or state is None:
            raise PortfolioCorruptError("incomplete portfolio files")
        if "schema_version" not in state:
            raise PortfolioCorruptError("missing schema_version")
        txs = self._read_jsonl("transactions.jsonl")
        sigs = self._read_jsonl("signals.jsonl")
        positions = [PositionSnapshot.from_dict(p) for p in (positions_d.get("positions") or [])]
        bundle = PortfolioBundle(
            account=AccountSnapshot.from_dict(account_d),
            positions=positions,
            transactions=txs,
            signals=sigs,
            state_version=int(state.get("state_version", 1)),
            last_updated=str(state.get("last_updated", _iso())),
            schema_version=int(state.get("schema_version", SCHEMA_VERSION)),
            content_sha=str(state.get("content_sha", "")),
        )
        validate_bundle(bundle)
        self._version = bundle.state_version
        return bundle

    def save(self, bundle: PortfolioBundle, *, expected_version: Optional[int] = None) -> None:
        validate_bundle(bundle)
        bundle.recompute_equity()
        bundle.state_version = self._version + 1
        bundle.last_updated = _iso()
        bundle.content_sha = bundle.fingerprint()
        self._write_json("account.json", bundle.account.to_dict())
        self._write_json(
            "positions.json",
            {"positions": [p.to_dict() for p in bundle.positions]},
        )
        tx_file = self.root / "transactions.jsonl"
        if tx_file.exists():
            tx_file.unlink()
        if bundle.transactions:
            self._append_jsonl("transactions.jsonl", bundle.transactions)
        else:
            tx_file.write_text("", encoding="utf-8")
        sig_file = self.root / "signals.jsonl"
        if sig_file.exists():
            sig_file.unlink()
        if bundle.signals:
            self._append_jsonl("signals.jsonl", bundle.signals)
        else:
            sig_file.write_text("", encoding="utf-8")
        self._write_json(
            "state.json",
            {
                "schema_version": SCHEMA_VERSION,
                "state_version": bundle.state_version,
                "last_updated": bundle.last_updated,
                "content_sha": bundle.content_sha,
            },
        )
        self._version = bundle.state_version


class PaperPortfolioEngine:
    def __init__(
        self, store: LocalPortfolioStore, *, cost_bps: float = 15.0, lot_size: int = 100
    ) -> None:
        self.store = store
        self.cost_bps = cost_bps
        self.lot_size = lot_size
        self._seen_signal_ids: set[str] = set()
        self._seen_cycle_ids: set[str] = set()

    def load(self) -> PortfolioBundle:
        return self.store.load()

    def _cost(self, notional: float) -> float:
        return abs(notional) * (self.cost_bps / 10_000.0)

    def apply_buy(
        self,
        bundle: PortfolioBundle,
        *,
        symbol: str,
        price: float,
        signal_id: str,
        cycle_id: str = "",
        quantity: Optional[int] = None,
        max_position_pct: float = 0.15,
        confidence: float = 0.0,
        score: float = 0.0,
        reasons: tuple[str, ...] = (),
    ) -> tuple[PortfolioBundle, dict[str, Any]]:
        if signal_id and signal_id in self._seen_signal_ids:
            raise PortfolioStoreError(f"duplicate signal_id {signal_id[:12]}")
        if cycle_id and cycle_id in self._seen_cycle_ids:
            raise PortfolioStoreError(f"duplicate cycle_id {cycle_id}")
        for t in bundle.transactions:
            if t.get("signal_id") == signal_id:
                raise PortfolioStoreError("duplicate signal in ledger")
            if cycle_id and t.get("cycle_id") == cycle_id and t.get("side") == "BUY":
                raise PortfolioStoreError("duplicate cycle BUY")

        if price <= 0:
            raise PortfolioStoreError("invalid price")
        equity = max(bundle.account.equity, bundle.account.cash_available)
        max_notional = equity * max_position_pct
        if quantity is None:
            unit = price * self.lot_size * (1 + self.cost_bps / 10_000)
            affordable = int(bundle.account.cash_available / unit) if unit > 0 else 0
            max_lots = int(max_notional / (price * self.lot_size)) if price > 0 else 0
            lots = max(0, min(affordable, max(1, max_lots) if max_lots > 0 else affordable))
            quantity = lots * self.lot_size
        if quantity <= 0 or quantity % self.lot_size != 0:
            raise PortfolioStoreError("quantity must be positive multiple of lot")

        cash_before = bundle.account.cash_available
        gross = quantity * price
        fee = self._cost(gross)
        net = gross + fee
        if net > cash_before + 1e-9:
            raise PortfolioStoreError("insufficient cash")
        cash_after = cash_before - net
        if cash_after < 0:
            raise PortfolioStoreError("cash would go negative")

        pos = next((p for p in bundle.positions if p.symbol == symbol and p.status == "OPEN"), None)
        if pos is None:
            pos = PositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                average_entry=price,
                market_price=price,
                status="OPEN",
            )
            bundle.positions.append(pos)
        else:
            total_qty = pos.quantity + quantity
            pos.average_entry = ((pos.average_entry * pos.quantity) + (price * quantity)) / total_qty
            pos.quantity = total_qty
            pos.market_price = price

        tx_id = str(uuid.uuid4())
        tx = {
            "transaction_id": tx_id,
            "signal_id": signal_id,
            "cycle_id": cycle_id,
            "timestamp": _iso(),
            "symbol": symbol,
            "side": "BUY",
            "quantity": quantity,
            "price": price,
            "gross_value": gross,
            "transaction_cost": fee,
            "net_value": net,
            "cash_before": cash_before,
            "cash_after": cash_after,
        }
        bundle.transactions.append(tx)
        sig = {
            "signal_id": signal_id,
            "cycle_id": cycle_id,
            "timestamp": _iso(),
            "symbol": symbol,
            "action": "BUY",
            "score": score,
            "confidence": confidence,
            "rank": 1,
            "reasons": list(reasons),
            "data_used": ["OHLCV", "Volume", "Momentum", "Model"],
            "portfolio_transaction_id": tx_id,
            "notification_status": "PENDING",
        }
        bundle.signals.append(sig)
        bundle.account.cash_available = cash_after
        bundle.recompute_equity()
        if signal_id:
            self._seen_signal_ids.add(signal_id)
        if cycle_id:
            self._seen_cycle_ids.add(cycle_id)
        return bundle, tx

    def mark_notification(self, bundle: PortfolioBundle, signal_id: str, status: str) -> None:
        for s in bundle.signals:
            if s.get("signal_id") == signal_id:
                s["notification_status"] = status


def format_portfolio_telegram(
    bundle: PortfolioBundle, *, top: Optional[dict[str, Any]] = None
) -> str:
    lines: list[str] = []
    if top:
        action = top.get("action", "BUY")
        emoji = "🟢" if action == "BUY" else ("🔴" if action == "SELL" else "⚪")
        lines.extend(
            [
                f"{emoji} IDX TOP 1 — {action}",
                str(top.get("symbol", "")),
                f"Harga: Rp{top.get('price', 0):,.0f}".replace(",", "."),
                f"Quantity: {int(top.get('quantity', 0)) // 100} lot"
                if int(top.get("quantity", 0)) >= 100
                else f"Quantity: {top.get('quantity', 0)}",
                f"Score: {top.get('score', 0):.1f}",
                f"Confidence: {float(top.get('confidence', 0)) * 100:.0f}%",
                "",
                "Kenapa dipilih?",
                str(top.get("why", "Skor tertinggi setelah filter & risk gate.")),
                "",
            ]
        )
    lines.append("💼 PORTFOLIO")
    if top and top.get("action") == "BUY":
        lines.append(f"BUY {top.get('symbol')} tercatat ✓")
        lines.append("Status: OPEN")
    elif top and top.get("action") == "NO_SIGNAL":
        lines.append("Tidak ada BUY — portfolio tidak diubah")
    lines.append("")
    cash = bundle.account.cash_available
    lines.append("💰 Saldo tersedia:")
    lines.append(f"Rp{cash:,.0f}".replace(",", "."))
    lines.append("")
    lines.append("📦 Aset yang dimiliki:")
    open_pos = [p for p in bundle.positions if p.status == "OPEN" and p.quantity > 0]
    if not open_pos:
        lines.append("Tidak ada")
    else:
        for p in open_pos:
            px = p.market_price if p.market_price is not None else p.average_entry
            pnl = p.unrealized_pnl if p.unrealized_pnl is not None else 0.0
            sign = "+" if pnl >= 0 else ""
            lots = p.quantity // 100 if p.quantity >= 100 else p.quantity
            lines.append(p.symbol)
            lines.append(f"{lots} lot")
            lines.append(f"Avg: Rp{p.average_entry:,.0f}".replace(",", "."))
            lines.append(f"Market: Rp{px:,.0f}".replace(",", "."))
            lines.append(f"P/L: {sign}Rp{pnl:,.0f}".replace(",", "."))
    lines.append("")
    lines.append("⚠️ PAPER TRADING — bukan order broker nyata.")
    return "\n".join(lines)
