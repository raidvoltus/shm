# Persistent paper portfolio

Source of truth: branch `idxbot-state` path `portfolio/`.

```
portfolio/
  account.json
  positions.json
  transactions.jsonl
  signals.jsonl
  state.json
```

- Initial capital: `INITIAL_BALANCE` (default 10_000_000 IDR)
- Existing state is **never** silently reset
- Corrupt state → fail-closed
- Order: analysis → TOP1 → paper BUY → persist → verify → Telegram
- Message composer is narrator only
