---
title: CISSOU
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# CISSOU — ESI Student Assistant

CISSOU is the AI assistant of **ESI** (École Supérieure d'Informatique, Algiers), built by the
**CSE** (Scientific Club of ESI). It answers questions from new and prospective students — about
admission, the five-year curriculum, student life and CSE — in **French or English**, grounded in
the official ESI documentation.

---

## How it works

```
question ──► BM25 retrieval ──► system prompt ──► provider router ──► answer
             (68 chunks)         (~10 sections)    Gemini → Groq → offline
                                                          ▲
             session history ────────────────────────────┘
```

1. **Retrieval.** The knowledge document is split along its Markdown headings into 68 chunks,
   each keeping its heading breadcrumb (`Programme > PHASE 2 > 2CS > SIQ`). Every FAQ entry
   becomes its own chunk. A BM25 index picks the ~10 sections relevant to the question.
2. **Grounding.** Only those sections go into the prompt, with instructions to answer strictly
   from them and to reply in the student's language.
3. **Generation.** A router calls Gemini; if a key is revoked or rate-limited it moves to the
   next key, then to Groq, and finally degrades to a knowledge-base excerpt rather than failing.
4. **Memory.** History is scoped per session, so follow-ups like *"et en 2CP ?"* work.

**Retrieval is embedding-free on purpose.** The corpus is 66k characters — a BM25 index built at
start-up answers in microseconds with no embedding API calls, no quota cost and no vector store to
deploy. What makes it accurate is the domain layer: accent-folded FR/EN normalisation plus an
ESI/CSE alias table, so *prépa*, *1CP* and *first year* all reach the same passages.

Measured on 32 bilingual questions (`scripts/eval_retrieval.py`):

| | |
|---|---|
| Recall (answer present in retrieved context) | **32/32 (100%)** |
| Average context per question | 9,629 chars (~2.4k tokens) |
| Previous approach (whole document, every turn) | 66,281 chars (~16.6k tokens) |
| **Context reduction** | **85.6%** |

`RETRIEVAL_TOP_K=10` is not arbitrary — recall degrades below it:

| `top_k` | 4 | 5 | 6 | 8 | **10** | 12 |
|---|---|---|---|---|---|---|
| recall | 28/32 | 30/32 | 30/32 | 30/32 | **32/32** | 32/32 |
| ~tokens | 893 | 1,210 | 1,488 | 1,977 | **2,386** | 2,718 |

Lower it if you are tightly rate-limited; free-tier throughput is bound by tokens per minute.

**Free-tier rate limits are the real throughput ceiling.** Groq's free tier allows 7,000 input
and 1,000 output tokens per minute per key, so a ~4k-token prompt permits roughly one request per
minute per key. Add more keys (they are pooled and round-robined), lower `RETRIEVAL_TOP_K`, or
move to a paid tier. When every key is rate-limited CISSOU answers from the knowledge base
directly rather than erroring.

---

## Quick start

```bash
git clone https://github.com/pedros18/chatbot-esi-cse.git
cd chatbot-esi-cse

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then add your key(s)
python app.py               # http://localhost:5000
```

Get a free Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), and
optionally a Groq key at [console.groq.com/keys](https://console.groq.com/keys) for failover.

Without any key CISSOU still starts in **offline mode** and answers with raw knowledge-base
excerpts — useful for working on retrieval without spending quota.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET`  | `/` | Chat UI |
| `GET`  | `/health` | Status, knowledge stats, provider availability |
| `POST` | `/api/chat` | Ask a question |
| `POST` | `/api/chat/stream` | Same, streamed as server-sent events |
| `POST` | `/api/session/reset` | Clear a conversation |
| `GET`  | `/api/search?q=` | Inspect retrieval without an LLM call |
| `GET`  | `/api/diagnostics` | Per-key health, models, active sessions |

```bash
curl -X POST localhost:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Quelles sont les spécialités en 2CS ?"}'
```

```json
{
  "reply": "En 2CS, les étudiants choisissent l'une des quatre spécialités…",
  "session_id": "3f2a…",
  "provider": "gemini",
  "sources": ["Complete 5-Year Program Structure > PHASE 2 > 2CS"],
  "latency_ms": 1840
}
```

Pass the returned `session_id` back on the next request to keep conversation context.

---

## Configuration

Everything is environment-driven — see [`.env.example`](.env.example) for the annotated list.
The most useful knobs:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_GEMINI_API_KEY`, `…_2` … `…_20` | — | Gemini key pool; rotated on quota errors |
| `GROQ_API_KEY` | — | Failover provider |
| `PROVIDER_ORDER` | `gemini,groq` | Order providers are tried in |
| `GEMINI_MODEL` | `gemini-3.5-flash` | |
| `GROQ_MAX_OUTPUT_TOKENS` | `900` | Must stay under Groq's free-tier OTPM limit (1000) |
| `RETRIEVAL_TOP_K` | `10` | Sections injected per question (recall drops below 10 — see below) |
| `MAX_ATTEMPTS_PER_REQUEST` | `4` | Keys one request may try before falling back |
| `ANSWER_DEADLINE` | `45` | Seconds before a request stops trying new keys |
| `MAX_CONTEXT_CHARS` | `24000` | Hard cap on retrieved context |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-IP limit |
| `CORS_ORIGINS` | `*` | **Restrict this in production** |

---

## Development

```bash
pip install -r requirements-dev.txt

pytest                                   # 80 tests, no network required
python scripts/eval_retrieval.py         # retrieval recall + token cost
python scripts/ask.py "What is 1CP?"     # one-shot CLI
python scripts/ask.py --retrieval-only "admission"   # inspect ranking
python scripts/ask.py                    # interactive REPL
```

Tests run entirely against the offline provider, so the suite needs no API key and costs nothing.

### Layout

```
app.py                  entrypoint (gunicorn app:app)
cissou/
  config.py             settings from the environment
  knowledge.py          document loading + heading-aware chunking
  retrieval.py          BM25 index, FR/EN normalisation, alias table
  prompts.py            system prompt
  providers/            gemini.py · groq.py · echo.py (offline) · base.py
  router.py             key pool, health tracking, cross-provider failover
  sessions.py           per-session history with TTL + LRU
  service.py            retrieve → prompt → generate → remember
  api.py                HTTP routes, validation, rate limiting
  app_factory.py        create_app()
data/esi_knowledge.md   the knowledge base  ← edit this to teach CISSOU
static/index.html       chat UI
scripts/                CLI + evaluation harness
tests/                  80 tests
```

### Updating the knowledge base

Edit `data/esi_knowledge.md` and restart — chunking and indexing happen at start-up. Keep the
Markdown heading structure: headings become the breadcrumbs CISSOU cites as sources, and
`**Q: …**` lines are indexed as individual FAQ entries.

After editing, re-run `python scripts/eval_retrieval.py` to confirm nothing regressed, and add a
case there for any question the new content should answer.

---

## Deployment

Render / Railway / Heroku, using the included `Procfile`:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
```

Set `GOOGLE_GEMINI_API_KEY` (and optionally `GROQ_API_KEY`) as environment variables in the
dashboard, and set `CORS_ORIGINS` to your frontend's origin. Sessions are in-process, so keep
`--workers 1` (threads scale fine) unless you move session storage to Redis.

---

## Security

**Never commit API keys.** `.env` is git-ignored; `.env.example` holds placeholders only.
Keys pushed to a public repository are detected and suspended automatically by the provider —
this project has already lost a full set of Gemini keys that way. If a key is ever exposed,
revoke it in the provider console; rotating the file is not enough, because the key stays in
git history.

---

## Credits

Built by the **CSE (Scientific Club of ESI)**. CISSOU was developed by **Badreddine Sayah**,**ZOUAK Syrine Lyna** , **ABID Ali**
CSE AI Co-Managers, inspired by alumni development managers **Yasmine Zaidi**,
**Hamza Arab** and **Youcef Missoum**.

- [ESI](http://www.esi.dz/) · [CSE](https://cse.club/)

Made with ❤️ for the ESI community.
