# HumanLens

**AI Writing Forensics + Natural Writing Assistant** — fully local, zero-cost, no paid API keys.

HumanLens estimates the likelihood that a piece of text shares characteristics with AI-generated
writing, explains *why* it reached that estimate, and offers a genuine writing-improvement engine
that makes text clearer and more natural without inventing facts or randomly swapping in synonyms.

> **This tool does not — and cannot — prove who wrote something.** Text-only AI detection is
> probabilistic. HumanLens reports an AI-likelihood estimate, a confidence level, and the specific
> signals behind that estimate. Treat every result as a hint, not a verdict.

**Using this for SEO / content-marketing work:** the humanizer is built to genuinely improve
clarity and reduce formulaic phrasing — not to specifically evade any commercial AI detector
(Originality.ai, GPTZero, Winston, etc.). It may incidentally lower AI-likelihood scores as a side
effect of removing stock phrases and varying sentence structure, but that's not its design goal.
Also worth knowing: search engines generally don't penalize content for being AI-assisted — they
penalize low-quality, unhelpful content regardless of how it was produced — so treat the detector
score as a writing-quality signal, not an SEO ranking signal.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Installation](#installation)
  - [Windows](#windows)
  - [Linux / macOS](#linux--macos)
- [Running the app](#running-the-app)
- [Production deployment](#production-deployment)
- [Optional: local LLM via Ollama](#optional-local-llm-via-ollama)
- [Dataset format](#dataset-format)
- [Training the model](#training-the-model)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [API reference](#api-reference)
- [Settings](#settings)
- [Privacy](#privacy)
- [Security](#security)
- [License](#license)
- [Limitations (read this)](#limitations-read-this)
- [Design decisions](#design-decisions)
- [Future extensibility](#future-extensibility)

---

## Features

**AI-likelihood detection**
- Multi-signal analysis: statistical features, stylometry, formulaic-phrase detection, and an
  ML classifier — never a single heuristic.
- Paragraph-level breakdown, not just a whole-document score.
- Explainability: every result comes with supporting signals *and* counter-signals in plain
  language, plus a confidence level that's capped for short text.
- Configurable sensitivity, with a documented, non-magic threshold-shift mechanism.

**Writing improvement ("humanizer")**
- Rule-based rewriting that removes empty hedging phrases, simplifies corporate-speak, varies
  repeated sentence openers, and flags (rather than blindly rewrites) overly long sentences.
- Every change is logged as a real diff (`rule`, `before`, `after`, `reason`) — no black box.
- Meaning-preservation validation: checks that numbers, dates, URLs, quoted text, and named
  entities from the original survive the rewrite, plus a TF-IDF cosine-similarity score.
- Optional local LLM rewriting via [Ollama](https://ollama.com) — fully local, no API key — with
  automatic, silent fallback to the rule-based engine if Ollama isn't running.

**Everything else**
- FastAPI backend with 9 documented endpoints, Pydantic validation, structured logging.
- Production middleware: optional API key auth, per-client rate limiting, request-ID tracing,
  docs disabled outside development, CORS locked to configured origins (see [Security](#security)).
- Streamlit dashboard: Analyze / Writing Improvement / History / Evaluation / Settings pages.
- SQLite history with configurable full-text storage (privacy-conscious).
- Docker + docker-compose setup for one-command deployment of both services with persistent storage.
- Real evaluation framework: stratified k-fold cross-validation (not train-on-test-and-brag),
  plus a hand-written adversarial test suite that reports failures honestly.
- 79 automated tests (pytest) covering features, detector, humanizer, database, API, security,
  the dataset import/merge scripts, and the Streamlit UI's navigation flows.
- Light theme with a clean home page (hero input + action pills), similar in spirit to familiar
  writing-tool dashboards — configured in `.streamlit/config.toml` plus custom CSS.
- MIT licensed, with a third-party license summary for every dependency (safe for commercial use).

---

## Architecture

```
                     ┌─────────────────────┐
                     │   Streamlit frontend │
                     └──────────┬───────────┘
                                │ HTTP (localhost)
                     ┌──────────▼───────────┐
                     │     FastAPI backend   │
                     └──────────┬───────────┘
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                   ▼
     ┌───────────────┐  ┌───────────────┐   ┌───────────────┐
     │    Detector    │  │   Humanizer    │   │   Database     │
     │ Features→Model │  │ Rules/Ollama → │   │   (SQLite)     │
     │ →Calibration→  │  │  Validation    │   │                │
     │  Explanation   │  │                │   │                │
     └───────────────┘  └───────────────┘   └───────────────┘
```

The detector is explicitly layered (`FeatureExtractor → Model → Calibration → Explanation`) and
the model implements a small, swappable interface (`predict_proba`, `is_trained`, `save`, `load`)
so a transformer-based model can replace `AIDetectorModel` later without touching the API,
frontend, or database.

**Stack:** Python · FastAPI · Streamlit · scikit-learn · NLTK · textstat · SQLite · pytest.
No PyTorch/Transformers dependency is required for the base install; the architecture supports
adding one later (see [Future extensibility](#future-extensibility)).

---

## Project structure

```
humanlens/
├── .streamlit/
│   └── config.toml               # light theme + accent color for the frontend
├── app/
│   ├── config.py                 # central settings (env-var driven)
│   ├── frontend/
│   │   └── streamlit_app.py      # dashboard: Analyze/Improve/History/Evaluation/Settings
│   ├── api/
│   │   └── main.py               # FastAPI app + endpoints
│   ├── detector/
│   │   ├── features.py           # statistical + stylometric feature extraction
│   │   ├── formulaic.py          # configurable formulaic-phrase detector
│   │   ├── classifier.py         # TF-IDF + stylometry -> Logistic Regression
│   │   ├── calibration.py        # sensitivity threshold-shift, short-text shrinkage
│   │   ├── explain.py            # human-readable supporting/counter signals
│   │   └── detector.py           # orchestrates the pipeline end to end
│   ├── humanizer/
│   │   ├── rules.py              # deterministic rewriting rules
│   │   ├── engine.py             # orchestrates rewriting + validation
│   │   ├── validation.py         # meaning-preservation checks
│   │   └── ollama_client.py      # optional local LLM integration
│   ├── database/
│   │   ├── db.py                 # SQLite access layer
│   │   └── models.py             # record dataclasses
│   └── models/                   # trained model artifacts (generated, gitignored)
├── datasets/
│   └── combined_sample.csv       # small original seed dataset (see Limitations)
├── evaluation/
│   ├── evaluate.py               # stratified k-fold cross-validation
│   ├── metrics.py                # accuracy/precision/recall/F1/ROC-AUC/confusion matrix
│   ├── adversarial.py            # hand-written robustness stress test
│   └── reports/                  # generated JSON reports
├── scripts/
│   ├── build_seed_dataset.py     # generates datasets/combined_sample.csv
│   ├── download_raid.py          # imports a larger sample from the RAID benchmark
│   ├── merge_datasets.py         # combines multiple text,label CSVs, dedupes, reports balance
│   └── train_model.py            # trains and saves the classifier
├── tests/                        # 58 pytest tests (unit + integration)
├── requirements.txt
├── .env.example
├── Dockerfile                    # API service image
├── Dockerfile.frontend           # Streamlit service image
├── docker-compose.yml            # runs both with persistent storage
├── .dockerignore
├── LICENSE                       # MIT
├── THIRD_PARTY_LICENSES.md       # dependency license summary
├── run.py                        # launches API + frontend together
└── README.md
```

---

## Installation

Requires **Python 3.10+**.

### Windows

```powershell
git clone <your-repo-url> humanlens
cd humanlens
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
python scripts\build_seed_dataset.py
python scripts\train_model.py
```

### Linux / macOS

```bash
git clone <your-repo-url> humanlens
cd humanlens
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt_tab')"
python3 scripts/build_seed_dataset.py
python3 scripts/train_model.py
```

This installs everything (FastAPI, Streamlit, scikit-learn, NLTK, textstat — all free/open-source),
generates the seed dataset, and trains the initial classifier. No API keys, no paid services.

---

## Running the app

**Easiest — one command, starts both services:**
```bash
python run.py
```
Then open the Streamlit URL it prints (usually `http://localhost:8501`).

**Or run each separately (two terminals — useful for seeing separate logs):**
```bash
# Terminal 1
uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2
streamlit run app/frontend/streamlit_app.py
```

The frontend calls the API at `http://127.0.0.1:8000` by default (configurable via
`HUMANLENS_API_HOST` / `HUMANLENS_API_PORT`). API docs are auto-generated at
`http://127.0.0.1:8000/docs` (development only — see [Security](#security)).

---

## Production deployment

**Docker (recommended for production):**

```bash
cp .env.example .env
# edit .env: set HUMANLENS_API_KEY to a long random value,
# set HUMANLENS_ALLOWED_ORIGINS to your real frontend origin(s),
# set HUMANLENS_ENV=production

docker compose up --build -d
```

This builds and runs the API (port 8000) and Streamlit frontend (port 8501) as separate
containers, with the trained model and SQLite database persisted in a named Docker volume
(`humanlens_data`) so they survive container restarts/rebuilds. The Dockerfile trains the base
model from the seed dataset at build time; mount your own retrained model over the same volume
path to ship a stronger one (see [Training the model](#training-the-model)).

> Docker builds were not executed in this environment (no Docker daemon available here), but
> every individual step the Dockerfile runs — `pip install -r requirements.txt`, the NLTK
> download, `build_seed_dataset.py`, and `train_model.py` — was verified directly. Run
> `docker compose up --build` yourself and open an issue/check logs if anything doesn't match.

**Without Docker**, running behind a reverse proxy (nginx, Caddy, etc.) also works — just make
sure to set `HUMANLENS_ENV=production`, `HUMANLENS_API_KEY`, and `HUMANLENS_ALLOWED_ORIGINS`
before starting `uvicorn`, and terminate TLS at the proxy since uvicorn itself serves plain HTTP.

**Before going live, at minimum:**
1. Set `HUMANLENS_API_KEY` to a long random value and keep it out of version control.
2. Set `HUMANLENS_ALLOWED_ORIGINS` to your actual frontend origin(s) — the default only allows
   `localhost`.
3. Set `HUMANLENS_ENV=production` to disable the `/docs` and `/redoc` pages.
4. Put a reverse proxy or load balancer in front of uvicorn for TLS; uvicorn alone serves HTTP.
5. Retrain on a larger dataset than the 95-row seed set (see [Limitations](#limitations-read-this)).
6. If you'll run more than one API process/instance, replace the in-memory rate limiter in
   `app/api/middleware.py` with a shared store (Redis, etc.) — see that file's docstring.

---

## Optional: local LLM via Ollama

The base app works fully without this. To enable LLM-based rewriting:

1. Install [Ollama](https://ollama.com) (free, runs locally).
2. Pull a model: `ollama pull qwen2.5-coder:3b` (or any model you prefer).
3. In the Streamlit **Writing Improvement** page, check "Use local LLM (Ollama) if available".
   The page shows live status (reachable / not reachable, and which models are pulled) and lets
   you type the exact model name to use for that request — it does **not** need to match
   `HUMANLENS_OLLAMA_MODEL` in your `.env`; the per-request field always wins.

If Ollama isn't running, or the requested model isn't pulled, or the call fails for any other
reason, HumanLens automatically falls back to the rule-based rewriter and shows you the *specific*
reason in a warning (e.g. `Ollama rejected the request for model 'llama3.1': model not found... 
Models currently pulled: qwen2.5-coder:3b.`) — not a generic "it didn't work". The app never
becomes unusable because of a missing or misconfigured LLM.

**Troubleshooting:** hit `GET /ollama/status` on the API (or just check the Writing Improvement
page) to see whether Ollama is reachable and which models are actually pulled. The most common
issue is a model-name mismatch — the default configured model is `HUMANLENS_OLLAMA_MODEL`
(`llama3.1` unless you change it), but you likely pulled something else (e.g. `qwen2.5-coder:3b`).
Either set `HUMANLENS_OLLAMA_MODEL` to match what you actually pulled, or just type the correct
model name into the "Ollama model to use" field on the Writing Improvement page.

---

## Dataset format

CSV with exactly two columns:

```csv
text,label
"Some human-written sentence.",human
"Some AI-generated sentence.",ai
```

`label` must be `human` or `ai`. See `datasets/combined_sample.csv` for the seed dataset, and
`scripts/build_seed_dataset.py` for how it was generated.

### Getting more data (recommended before relying on the detector)

The 95-row seed dataset is enough to prove the pipeline works, not enough for trustworthy
real-world accuracy (see [Limitations](#limitations-read-this)). `scripts/download_raid.py`
imports a much larger, stratified sample from the **RAID benchmark**
([liamdugan/raid](https://github.com/liamdugan/raid), ACL 2024, MIT licensed) — real human-written
text plus generations from 11 different AI models (ChatGPT, GPT-4, GPT-3, GPT-2, Llama-2-Chat,
Mistral, Mistral-Chat, MPT, MPT-Chat, Cohere, Cohere-Chat) across genres including news, reviews,
Reddit posts, wiki articles, book excerpts, and abstracts.

```bash
pip install datasets   # optional extra, only needed for this import script

# Pull a balanced sample (streams only what it needs — not the full ~11GB dataset)
python scripts/download_raid.py --per-class 300 --domains news reviews reddit wiki

# Merge with (or instead of) the seed dataset
python scripts/merge_datasets.py datasets/combined_sample.csv datasets/raid_sample.csv \
    --out datasets/combined_sample.csv

# Retrain and re-evaluate on the larger dataset
python scripts/train_model.py
python evaluation/evaluate.py
```

By default it keeps only "clean" (non-adversarial, greedy-decoded) generations — the right
starting point for a basic classifier. Pass `--include-adversarial` once that's working well, to
train against RAID's harder, adversarially-perturbed generations too (paraphrased, homoglyph
attacks, etc.) for a more robust detector.

**Honesty note:** this script's filtering/sampling logic was tested against a mock dataset that
exactly matches RAID's real column schema (`id`, `source_id`, `model`, `decoding`,
`repetition_penalty`, `attack`, `domain`, `generation`, etc.) — including a caught-and-fixed bug
where AI-model sampling could under-fill if fewer than 11 distinct models appeared in a stream
window. The actual network download to Hugging Face was **not** exercised in the sandbox this
project was built in (it blocks huggingface.co); run it yourself with a normal internet
connection, and if RAID's schema has changed since this was written, `_is_human_row()` near the
top of `scripts/download_raid.py` is the first place to check.

For your specific use case (SEO/content-marketing publishing), you'll likely get better real-world
accuracy by *also* adding your own labeled examples from that exact genre — your own published
human-written articles, plus outputs from whatever AI tools your writers actually use — since
that's the distribution the detector will actually see in production. `merge_datasets.py` accepts
any number of input CSVs, so you can combine the seed set, a RAID sample, and your own data in one
command.

---

## Training the model

```bash
python scripts/train_model.py --dataset datasets/combined_sample.csv
```

Trains a Logistic Regression classifier over combined word TF-IDF (1-2 grams), character TF-IDF
(3-5 grams), and the full stylometric feature set from `app/detector/features.py`, then saves it
to `app/models/`. The detector automatically picks up the trained model on next load; if no model
file exists, it falls back to a transparent, clearly-labeled heuristic (see `detector.py:
_heuristic_probability`) rather than crashing or pretending to have ML it doesn't.

**To meaningfully improve accuracy:** replace/extend `datasets/combined_sample.csv` with a much
larger, more diverse dataset (thousands of rows, multiple AI models/prompting styles, multiple
genres) using the same schema, then re-run this script.

---

## Evaluation

```bash
python evaluation/evaluate.py --dataset datasets/combined_sample.csv --folds 5
python evaluation/adversarial.py
```

`evaluate.py` runs **stratified k-fold cross-validation** and reports accuracy, precision, recall,
F1, ROC-AUC, false-positive rate, false-negative rate, and a confusion matrix — computed on
out-of-fold predictions only, so the model is never scored on data it trained on. Results are
written to `evaluation/reports/latest_report.json` and shown on the Streamlit **Evaluation** page.

`adversarial.py` stress-tests the detector against short text, long formal/informal text, mixed
human/AI text, lightly-edited AI text, and heavily paraphrased AI text — and **reports failures
honestly**. On the seed dataset, it currently fails on lightly-edited and heavily-paraphrased AI
text, which is expected and documented below.

---

## Testing

```bash
pytest tests/ -v
```

79 tests across `test_features.py`, `test_detector.py`, `test_humanizer.py`, `test_database.py`,
`test_api.py`, `test_security.py`, `test_dataset_scripts.py`, and `test_frontend.py` — covering
empty input, very short/long input, Unicode, URLs, numbers, contractions, repeated words/phrases,
paragraph detection, SQL-injection-shaped input, invalid API requests, full request/response round
trips, API key auth, rate limiting, the RAID import/merge logic, and simulated UI interactions
(including a regression test for a real Streamlit widget-state bug found and fixed during
development). The frontend's live-API end-to-end test auto-skips if no server is running.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Full document analysis; saves to history by default |
| `POST` | `/analyze/paragraphs` | Paragraph-level breakdown only |
| `POST` | `/humanize` | Rewrite text; optionally attach result to a history entry |
| `GET` | `/ollama/status` | Diagnostic: is Ollama reachable, and which models are pulled |
| `GET` | `/history?limit=&offset=` | List past analyses |
| `GET` | `/history/{id}` | Full detail for one analysis |
| `DELETE` | `/history/{id}` | Delete one analysis |
| `GET` | `/health` | Liveness + whether a trained model is loaded (no API key required) |
| `GET` | `/version` | App version, environment, and whether an API key is required |
| `GET` | `/stats` | Aggregate stats over stored history |

Interactive docs: `http://127.0.0.1:8000/docs` (Swagger UI, generated from the Pydantic schemas —
disabled automatically when `HUMANLENS_ENV=production`).

---

## Settings

Configurable via `.env` (see `.env.example`) or the Streamlit Settings page:

- **Detector sensitivity** — shifts the decision threshold; doesn't rewrite the raw score.
- **Minimum analysis length / maximum text length.**
- **Local LLM enabled/disabled, Ollama host and model.**
- **Database location.**
- **Full-text storage on/off** (privacy).

---

## Privacy

- Everything runs locally by default. No text is sent anywhere unless you explicitly enable
  Ollama — and Ollama itself runs entirely on your machine, not in the cloud.
- History is stored in a local SQLite file (`app/humanlens.db` by default).
- Set `HUMANLENS_STORE_FULL_TEXT=false` to store only a short preview and the numeric results,
  discarding the full submitted text after analysis.

---

## Security

The base app defaults to local single-user development (no API key required, CORS restricted to
`localhost`). For anything beyond that:

- **API key auth** — set `HUMANLENS_API_KEY`; every endpoint except `/health` then requires an
  `X-API-Key` header matching it. The bundled Streamlit frontend sends this automatically if the
  same env var is set for both services (docker-compose does this for you).
- **Rate limiting** — `HUMANLENS_RATE_LIMIT_PER_MINUTE` (default 60) caps requests per client
  (per API key, or per IP if no key is set) in a rolling 60-second window. Returns `429` when
  exceeded. Implemented as an in-memory sliding window — correct for a single process, but not
  shared across multiple instances; see `app/api/middleware.py` for the note on swapping in Redis
  for a multi-instance deployment.
- **CORS** — `HUMANLENS_ALLOWED_ORIGINS` (comma-separated) restricts which browser origins may
  call the API. Defaults to the local Streamlit app only.
- **No arbitrary code execution** — no `eval`/`exec`/shell execution of any user input anywhere
  in the codebase. All database queries are parameterized (see `test_database.py::test_sql_injection_safe_text`).
- **Docs disabled in production** — `/docs` and `/redoc` are only served when
  `HUMANLENS_ENV` is unset or `development`.
- **Non-root containers** — both Dockerfiles create and run as a dedicated `humanlens` user.
- **Request tracing** — every response includes an `X-Request-ID` header, logged alongside
  method/path/status/duration, for tracing a single request through logs.

This is a solid baseline for a small-to-medium deployment. It is not a substitute for a security
review if you're handling sensitive data at scale — in particular, add HTTPS termination (a
reverse proxy, not uvicorn itself), consider per-user accounts instead of a single shared API key
if you have multiple customers, and swap the rate limiter for a shared store if you run more than
one API process.

---

## License

HumanLens is released under the **MIT License** (see `LICENSE`) — free to use, modify, and sell
commercially, with no royalties and no obligation to open-source your own changes. All
dependencies use permissive licenses (MIT/BSD/Apache-2.0) that also permit commercial use; see
`THIRD_PARTY_LICENSES.md` for the full list. Not legal advice — if you're shipping this as a paid
product at meaningful scale, have a lawyer confirm license compliance for your specific use case.

---

## Limitations (read this)

Being direct about this, per the project's own design principle:

1. **The seed dataset is still small.** `datasets/combined_sample.csv` has 95 hand-written
   examples (45 human, 50 AI-style), including batches skewed toward content-marketing/SEO genres
   and lightly-paraphrased AI text (added specifically to address a gap found during adversarial
   testing — see below). Cross-validated accuracy is currently **91.6%** (precision 0.96, recall
   0.88, ROC-AUC 0.98) — down from an earlier, less honest 98.7% on an easier version of this
   dataset. That drop is a *good* sign, not a regression: it means the dataset now contains
   genuinely harder examples the model has to actually work to separate, rather than only
   trivially-distinct ones. Still a small dataset — treat this as a pipeline sanity check, not a
   real-world benchmark. See [Dataset format](#dataset-format) for how to pull in a much larger,
   more diverse dataset (the RAID benchmark import script).

2. **A concrete before/after example of the precision/recall trade-off on small data.** Adding
   paraphrased-AI training examples to specifically target a known weak spot fixed that weak spot
   (`heavily_paraphrased_ai` in the adversarial suite went from correctly-failing at 0.38 to
   correctly-passing at 0.64) — but also pulled the decision boundary enough to introduce a *new*
   false positive on formal human writing (`long_text_formal_human` moved from 0.47, correct, to
   0.57, a false positive) that wasn't there before. Net adversarial pass rate stayed at 8/10, just
   with different failures. This is the real, honest behavior of hand-patching a ~95-row dataset:
   fixing one gap can open another, because there isn't enough data for the model to learn the
   general pattern instead of the specific examples you gave it. The durable fix is volume and
   diversity (hundreds–thousands of examples via the RAID import script), not one-off patches to
   the seed set — run `python evaluation/adversarial.py` yourself after any dataset change to see
   the current trade-offs rather than assuming a fix in one place didn't cost something elsewhere.

3. **Short text is unreliable.** Below ~150 words, stylometric signal is noisy; the app shrinks
   extreme probabilities toward 0.5 and caps the reported confidence accordingly, but it still
   can't give you a reliable read on a two-sentence snippet.

4. **This cannot detect authorship, only patterns.** Heavily edited AI text, or AI text specifically
   prompted to write casually, can look statistically similar to human writing. A human writing
   formally (legal, academic, technical) can look statistically similar to some AI output. Neither
   direction is proof.

5. **The ML model is a lightweight Logistic Regression**, not a transformer. This is a deliberate
   zero-cost, fast, fully-local choice. The architecture (`AIDetectorModel`'s
   `predict_proba`/`is_trained`/`save`/`load` interface) is designed so a transformer encoder can
   be swapped in later without touching the API, frontend, or database — but that swap is not done
   in this base version.

6. **The named-entity extraction in meaning-preservation validation is a regex heuristic**
   (capitalized-phrase matching), not a real NER model — it works reasonably for catching obvious
   entity loss but will have false positives/negatives. Number/date/URL/quote checks are exact and
   more reliable.

7. **Formulaic-phrase matching is intentionally a weak signal.** Words like "furthermore" or "in
   conclusion" alone never drive the AI-likelihood score by more than a small amount — this is
   enforced in `explain.py` (the transition-word signal strength is explicitly down-weighted) and
   in the ML feature set (formulaic rate is one of ~20+ features, not a rule).

---

## Design decisions

- **Why Logistic Regression + TF-IDF instead of a transformer?** Zero cost, no download, trains
  in under a second on a laptop, and is fully interpretable (feature weights are inspectable) —
  appropriate for a base version whose main job is to prove the *pipeline* is sound. See
  `app/detector/classifier.py` docstring for the swap-in interface for stronger models later.

- **Why cross-validation instead of a single train/test split?** With only 50 seed examples, a
  single split would be noisy and easy to accidentally overfit-and-brag about. K-fold
  cross-validation on out-of-fold predictions is the honest way to report a small-dataset metric.

- **Why not auto-rewrite long, comma-heavy sentences?** Splitting a sentence correctly requires
  understanding its grammar well enough not to break meaning. Rather than guess, `rules.py` flags
  these sentences with a reason and lets the human writer decide how to split them.

- **Why regex-based meaning-preservation checks instead of an embedding API?** Zero-cost
  requirement — no paid embedding API. TF-IDF cosine similarity (scikit-learn, local) plus exact
  regex checks for numbers/dates/URLs/quotes catches the highest-value failure mode (a rewrite that
  silently drops a fact) without needing network access.

---

## Future extensibility

The codebase is structured so these can be added without a rewrite:

- Swap `AIDetectorModel` for a transformer-based encoder (same `predict_proba`/`is_trained` interface).
- Multilingual detection (feature extraction and phrase lists are currently English-only).
- Document upload (PDF/DOCX) — would feed into the same `Detector.analyze(text)` entry point.
- User writing-style profiles, browser extension, team accounts, cloud deployment.
- Plagiarism analysis, cross-document style comparison.

These are intentionally **not implemented** in this base version — the spec asked for a serious,
working zero-cost core, not padding.
