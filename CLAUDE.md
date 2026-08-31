# CLAUDE.md — SitRep

Operating manual for Claude Code working in this repo. Read fully at the start of every
session. It overrides your defaults.

**Also read `NOTES.md` if it exists** — it carries working preferences that aren't committed.

Repo and folder name: `sitrep` (lowercase). Display name in prose and UI: **SitRep**.

---

## 1. What this is

SitRep receives a trading alert webhook (TradingView format), pulls two live market numbers,
has the Claude API write a short structured situation report, and delivers it to Telegram —
end to end in a few seconds.

It reports on a signal you already defined. It does not generate signals, does not backtest,
and never touches a broker.

---

## 2. Working style

- **Act as a senior reviewer.** Disagree when something is wrong. Agreeable answers that let
  fragile code ship are a failure.
- **Plan before editing.** For anything beyond a one-file change, state the plan and the exact
  files you'll touch, then wait for approval.
- **Never invent data.** If Binance is unreachable, surface the failure. Never fabricate OHLCV
  values — not in code, not in tests, not in examples. Test fixtures use real recorded
  responses.
- **Prefer deleting to adding.** The default answer to "should we add dependency X?" is no.
- Answer "why" with the failure it prevents, not with a principle.
- Give rules that can be *checked*, not rules that require judgment. "Nothing that opens a
  socket runs before the return" beats "avoid slow things in the handler."

---

## 3. The six invariants

Violating any of these is a bug regardless of test status.

1. **The webhook responds in under 500 ms.** Authenticate, validate, dedupe, enqueue, return
   `202 Accepted`. **Nothing that opens a socket runs before the return.** TradingView drops
   slow endpoints, and the LLM call alone takes seconds — a synchronous call in the handler is
   the classic failure of this design.
2. **No financial advice, ever.** The model outputs *context*, never a recommendation. No buy,
   sell, enter, target, should. Enforced structurally: `SignalBrief` has no field that could
   hold one, plus a guard that rejects advice-shaped text and falls back to the unenriched
   alert. A product decision, not a disclaimer.
3. **Structured output only.** Every LLM response parses into a Pydantic model before it
   reaches the formatter. Free text is a failure path, not a fallback.
4. **Degrade, never drop.** Enrichment fails → brief without context. Claude fails after
   retries → deliver the raw alert tagged `[unenriched]`. Every degradation path logs a reason
   code and has a test.
5. **Idempotency.** `alert_id = sha256(symbol|timeframe|condition|bar_time)`, deduped for
   5 minutes. TradingView retries; duplicate messages are a visible defect.
6. **Cost ceiling.** One LLM call per alert. Hard per-day spend cap from config. Demo endpoint
   rate-limited by IP. On cap, serve a cached example and say so plainly.

---

## 4. Stack — six runtime dependencies, do not expand

`fastapi`, `uvicorn`, `pydantic` (v2), `pydantic-settings`, `httpx`, `anthropic`

Dev: `pytest`, `pytest-asyncio`, `ruff`.

### Authentication

A shared secret token in a request header, compared with `hmac.compare_digest`. **Not HMAC
body signing** — TradingView alert bodies are static text and cannot compute a signature, so a
bearer token is the correct design here, not merely the simpler one. Constant-time comparison
still matters.

### Market data

One function. `GET https://api.binance.com/api/v3/klines` via httpx, returning two numbers:

- **volume vs the 20-day average** — conviction
- **% distance from the 20MA** — extension

No ATR, no provider interface, no abstraction layer. Crypto rather than equities because stock
markets are closed roughly 80% of the time, and the demo must show live data at any hour. On
failure: return `None`, log a reason code, continue.

### Claude API

- Model: **`claude-haiku-4-5-20251001`**, pinned in `config.py`, never at call sites. Verify
  the current ID at https://docs.claude.com/en/docs/about-claude/models/overview.
- The `anthropic` SDK directly, not a gateway. The call lives in one function, so switching
  providers is a single-file change.
- **Send computed features, not raw data.** Two numbers, not a hundred OHLCV rows. Fewer tokens
  and better output, because the model isn't doing arithmetic.

### Explicitly not in this project

`structlog` (stdlib `logging` plus a ~15-line JSON formatter), `mypy`, `respx` (use
`httpx.MockTransport`), SQLite/`aiosqlite` (use `deque(maxlen=50)` — free hosts have ephemeral
disk, so a database silently loses data on restart), `yfinance`, `ccxt`, Celery, Redis, Docker,
a React frontend.

---

## 5. Hosting

| Piece | Where |
|---|---|
| Backend (FastAPI) | Render free web service, native Python build, no Dockerfile |
| Demo page | GitHub Pages, served separately from the backend |
| CI | GitHub Actions — ruff + pytest on push |
| Market data | Binance public REST |
| Delivery | Telegram Bot API |

Secrets live in `.env`, gitignored, with `.env.example` committed as placeholders.

**Do not use HuggingFace Spaces** — as of roughly July 2026, only Static Spaces are free.

**Cold starts.** Render free services sleep after 15 minutes. The demo page is static on GitHub
Pages so it loads instantly, and it fires `GET /healthz` on load so the backend wakes while the
visitor reads the intro. About four lines of JS, and it removes the need for a keep-warm cron.

---

## 6. Repo layout

```
sitrep/
├── app/
│   ├── main.py          # FastAPI app, router wiring
│   ├── config.py        # pydantic-settings; ALL env vars, typed
│   ├── schemas.py       # AlertPayload, MarketContext, SignalBrief
│   ├── security.py      # token check (constant-time), demo IP rate limit
│   ├── dedupe.py        # TTL cache keyed on alert_id
│   ├── enrich.py        # Binance klines → two numbers
│   ├── llm.py           # Anthropic client, retries, cost accounting
│   ├── guards.py        # advice detector
│   ├── telegram.py      # formatter + send
│   ├── prompts/         # versioned .md prompt files
│   └── routes/          # webhook.py, demo.py, health.py
├── eval/                # dataset.jsonl, run_eval.py, results.md
├── demo/                # static page for GitHub Pages
├── tests/
├── docs/architecture.png
├── .github/workflows/ci.yml
├── .env.example
├── CLAUDE.md
└── README.md
```

### Conventions

- Type hints everywhere. Public functions document what fails and how.
- No bare `except`. Catch specific exceptions, log a reason code, degrade.
- All I/O async.
- Config only via `config.py`. `os.getenv` anywhere else is a review rejection.
- **Prompts are code** — versioned `.md` files in `app/prompts/`. Changing one requires
  re-running `make eval` and updating `eval/results.md`.
- One log line = one JSON object with `alert_id`, `stage`, `latency_ms`, `outcome`.
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`. Imperative,
  lowercase, no trailing period. One idea per commit.

Make targets: `setup`, `run`, `test`, `lint`, `eval`, `demo-alert`.

### Testing

- Every degradation path in invariant #4 gets a test.
- **The Anthropic API is never called in tests** — mock at the client boundary.
- Required cases: malformed JSON, wrong token, missing token, duplicate within TTL, duplicate
  after TTL expiry, LLM returns invalid JSON, LLM returns advice-shaped text, Binance timeout,
  Telegram 5xx, and a **latency assertion** on the webhook handler.
- Tests run offline in under 30 seconds. Target ≥80% coverage on `app/`.

### Eval harness

`eval/dataset.jsonl` holds labelled payloads across three categories: normal, edge-case
(missing fields, absurd values, unknown tickers), and adversarial (text attempting to elicit
advice). `make eval` reports schema-valid rate, severity agreement against labels, advice-guard
catch rate, fallback rate, latency p50/p95, and median cost per alert in USD. Results are
committed to `eval/results.md`.

---

## 7. Out of scope — push back if raised

Backtesting, signal generation, price prediction; user accounts; paid market-data feeds; broker
integration; embeddings or a vector store (there is no retrieval problem here); Kubernetes; an
agent loop.

**On agents:** the test is *does the system need to decide something at runtime that cannot be
decided at design time?* Here it doesn't — fixed input shape, fixed output shape, one
deterministic path. An agent would add latency variance, cost variance, nondeterminism and more
failure modes for no benefit.

**Backtesting is the most likely way this project dies.** It is a different project and it
triples the timeline. Push back hard if it comes up.

---

## 8. Deferred

Docker, Discord as a second delivery channel, HMAC body signing, richer features such as ATR, a
provider abstraction, and a keep-warm cron.

Also deferred: **adaptive enrichment**, where the model selects *which* context to fetch based
on alert type — volume spike pulls order-book depth, MA cross pulls higher-timeframe trend, gap
pulls recent news. That is genuine tool selection, and the one place an agent would earn its
keep. Not in v1.
