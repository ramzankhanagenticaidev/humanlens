"""
HumanLens Streamlit frontend.

A multi-page dashboard: Home, Analyze, Writing Improvement, History,
Evaluation, Settings. Talks to the FastAPI backend over local HTTP (no
paid services).

Run with:
    streamlit run app/frontend/streamlit_app.py
(the FastAPI backend must also be running — see run.py / README)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests
import streamlit as st

from app.config import settings

API_BASE = f"http://{settings.api_host}:{settings.api_port}"

st.set_page_config(
    page_title="HumanLens",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme: light background, green accent, rounded cards/pills.
# Native widget colors come from .streamlit/config.toml; this CSS handles
# custom components (cards, badges, hero box, pill nav) that Streamlit's
# theme system doesn't reach.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #fbfdfb;
        border-right: 1px solid #e6ebe7;
    }
    .hl-brand {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 1.3rem; font-weight: 800; color: #0f172a;
        padding: 0.2rem 0 0.1rem 0;
    }
    .hl-brand-tag { color: #16a34a; }
    .hl-brand-sub { color: #6b7280; font-size: 0.82rem; margin-bottom: 1rem; }

    /* Turn the sidebar radio into a QuillBot-style icon nav list */
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.55rem 0.7rem;
        border-radius: 10px;
        width: 100%;
        transition: background 0.15s ease;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #eefcf3;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }

    /* ---- Cards ---- */
    .hl-card {
        background: #ffffff;
        border: 1px solid #e6ebe7;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .hl-metric-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .hl-metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
    }
    .hl-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .hl-badge-high { background: #fee2e2; color: #b91c1c; }
    .hl-badge-moderate { background: #fef3c7; color: #92400e; }
    .hl-badge-low { background: #dcfce7; color: #15803d; }
    .hl-signal-row {
        padding: 0.5rem 0.7rem;
        border-left: 3px solid #f59e0b;
        background: #fffbeb;
        border-radius: 8px;
        margin-bottom: 0.45rem;
        font-size: 0.9rem;
        color: #1f2937;
    }
    .hl-counter-row {
        border-left: 3px solid #16a34a;
        background: #f0fdf4;
    }
    .hl-disclaimer {
        background: #f0fdf4;
        border: 1px solid #bbf1cf;
        border-radius: 10px;
        padding: 0.75rem 1.05rem;
        font-size: 0.85rem;
        color: #14532d;
        margin-bottom: 1rem;
    }

    /* ---- Home hero ---- */
    .hl-hero-wrap { padding-top: 2.5rem; text-align: center; }
    .hl-hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }
    .hl-hero-subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .hl-hero-box {
        background: #ffffff;
        border: 1px solid #e6ebe7;
        border-radius: 20px;
        padding: 1.4rem 1.4rem 1rem 1.4rem;
        box-shadow: 0 4px 18px rgba(16, 24, 40, 0.06);
        max-width: 900px;
        margin: 0 auto;
    }

    /* Pill-style buttons (st.button rendered inside these containers get the pill look via type) */
    div[data-testid="stButton"] > button {
        border-radius: 999px !important;
        border: 1px solid #e6ebe7 !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.1rem !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: #16a34a !important;
        border-color: #16a34a !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        border-color: #16a34a !important;
        color: #16a34a !important;
    }

    textarea, .stTextArea textarea {
        border-radius: 14px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def confidence_badge(level: str) -> str:
    cls = {"High": "hl-badge-high", "Moderate": "hl-badge-moderate", "Low": "hl-badge-low"}.get(level, "hl-badge-moderate")
    return f'<span class="hl-badge {cls}">{level} confidence</span>'


def call_api(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {}) or {}
    if settings.api_key:
        headers["X-API-Key"] = settings.api_key
    try:
        resp = requests.request(method, f"{API_BASE}{path}", timeout=60, headers=headers, **kwargs)
        if resp.status_code == 401:
            st.error("API rejected the request: missing or invalid API key. Check HUMANLENS_API_KEY.")
            return None
        if resp.status_code == 429:
            st.warning("Rate limit reached — please wait a moment before trying again.")
            return None
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            st.error(f"API error ({resp.status_code}): {detail}")
            return None
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not reach the HumanLens API at {API_BASE}. "
            "Make sure the backend is running: `python run.py` or `uvicorn app.api.main:app`."
        )
        return None
    except Exception as exc:
        st.error(f"Unexpected error calling API: {exc}")
        return None


NAV_OPTIONS = ["🏠 Home", "🔍 Analyze", "✨ Writing Improvement", "🕘 History", "📊 Evaluation", "⚙️ Settings"]
NAV_KEY_TO_LABEL = {opt.split(" ", 1)[1]: opt for opt in NAV_OPTIONS}


def switch_page(page_label: str, **state_updates) -> None:
    """Programmatic page switch, used as an on_click callback on the home
    page's pill buttons. Must be a callback (not called directly from the
    main script body) because the sidebar nav radio — which shares the
    session_state key this writes to — has already been instantiated
    earlier in the same run; Streamlit only allows mutating an
    already-instantiated widget's state from within a callback, which
    runs before the next script rerun rather than mid-run."""
    st.session_state["nav_page"] = NAV_KEY_TO_LABEL[page_label]
    for k, v in state_updates.items():
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="hl-brand">🪶 Human<span class="hl-brand-tag">Lens</span></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="hl-brand-sub">AI writing forensics + writing assistant</div>', unsafe_allow_html=True)

page_full = st.sidebar.radio("Navigate", NAV_OPTIONS, key="nav_page", label_visibility="collapsed")
page = page_full.split(" ", 1)[1]

if "sensitivity" not in st.session_state:
    st.session_state.sensitivity = settings.detector_sensitivity
if "use_ollama" not in st.session_state:
    st.session_state.use_ollama = settings.ollama_enabled
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

health = call_api("GET", "/health")
if health is not None:
    model_state = "✅ ML model trained" if health.get("model_trained") else "⚠️ Heuristic fallback (no trained model yet)"
    st.sidebar.caption(model_state)

# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------
if page == "Home":
    st.markdown(
        '<div class="hl-hero-wrap">'
        '<div class="hl-hero-title">Write something clearer today</div>'
        '<div class="hl-hero-subtitle">Check writing for AI-sounding patterns, or clean it up — free, local, no paid API.</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        st.markdown('<div class="hl-hero-box">', unsafe_allow_html=True)
        hero_text = st.text_area(
            "Paste your text",
            height=160,
            key="hero_input",
            placeholder="Paste a paragraph or two, then choose what you'd like to do with it below…",
            label_visibility="collapsed",
        )
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.button("🔍 AI Detector", use_container_width=True,
                      on_click=switch_page, args=("Analyze",), kwargs={"analyze_input": hero_text})
        with b2:
            st.button("✨ Humanizer", use_container_width=True,
                      on_click=switch_page, args=("Writing Improvement",), kwargs={"humanize_input": hero_text})
        with b3:
            st.button("🕘 History", use_container_width=True, on_click=switch_page, args=("History",))
        with b4:
            st.button("📊 Evaluation", use_container_width=True, on_click=switch_page, args=("Evaluation",))
        with b5:
            st.button("⚙️ Settings", use_container_width=True, on_click=switch_page, args=("Settings",))
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="hl-card"><b>🔍 AI Detector</b><br>'
            '<span style="color:#6b7280; font-size:0.9rem;">Paragraph-level AI-likelihood with plain-language '
            "supporting and counter signals — never a bare percentage.</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="hl-card"><b>✨ Writing Improvement</b><br>'
            '<span style="color:#6b7280; font-size:0.9rem;">Removes stiff, formulaic phrasing and shows every '
            "change as a real diff, with a meaning-preservation check.</span></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="hl-card"><b>🕘 History &amp; Evaluation</b><br>'
            '<span style="color:#6b7280; font-size:0.9rem;">Every analysis is saved locally, and the detector\'s '
            "real accuracy is always one click away — no hidden numbers.</span></div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# ANALYZE PAGE
# ---------------------------------------------------------------------------
elif page == "Analyze":
    st.title("Analyze Writing")
    st.markdown(
        '<div class="hl-disclaimer">'
        "This tool estimates the <b>likelihood</b> that text shares characteristics with AI-generated "
        "writing. It is a probabilistic assessment based on writing patterns — not proof of authorship. "
        "Short texts and heavily edited AI text are especially hard to assess reliably."
        "</div>",
        unsafe_allow_html=True,
    )

    text = st.text_area("Paste text to analyze", height=220, key="analyze_input",
                         placeholder="Paste at least a few sentences of text here…")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        analyze_clicked = st.button("Analyze Writing", type="primary", use_container_width=True)

    if analyze_clicked:
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing…"):
                result = call_api(
                    "POST", "/analyze",
                    json={"text": text, "sensitivity": st.session_state.sensitivity, "save_to_history": True},
                )
            if result:
                st.session_state.last_analysis = result
                st.session_state.last_text = text

    result = st.session_state.last_analysis
    if result:
        for w in result.get("warnings", []):
            st.warning(w)

        st.markdown("### Results")
        c1, c2, c3, c4 = st.columns(4)
        prob_pct = result["ai_probability"] * 100
        with c1:
            st.markdown(
                f'<div class="hl-card"><div class="hl-metric-label">AI-Likelihood</div>'
                f'<div class="hl-metric-value">{prob_pct:.0f}%</div>'
                f'<div style="margin-top:0.4rem;">{result["verdict"]}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="hl-card"><div class="hl-metric-label">Confidence</div>'
                f'<div style="margin-top:0.3rem;">{confidence_badge(result["confidence"])}</div>'
                f'<div style="margin-top:0.6rem; font-size:0.8rem; color:#6b7280;">'
                f'Model: {result["model_used"].replace("_", " ")}</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="hl-card"><div class="hl-metric-label">Words</div>'
                f'<div class="hl-metric-value">{result["num_words"]}</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="hl-card"><div class="hl-metric-label">Vocabulary Diversity</div>'
                f'<div class="hl-metric-value">{result["vocabulary_diversity"]:.2f}</div></div>',
                unsafe_allow_html=True,
            )

        st.progress(min(max(result["ai_probability"], 0.0), 1.0))

        sig_col1, sig_col2 = st.columns(2)
        with sig_col1:
            st.markdown("#### Supporting signals (AI-leaning)")
            if result["supporting_signals"]:
                for s in result["supporting_signals"]:
                    st.markdown(
                        f'<div class="hl-signal-row"><b>{s["label"]}</b><br>{s["detail"]}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No notable AI-leaning signals detected.")
        with sig_col2:
            st.markdown("#### Counter-signals (human-leaning)")
            if result["counter_signals"]:
                for s in result["counter_signals"]:
                    st.markdown(
                        f'<div class="hl-signal-row hl-counter-row"><b>{s["label"]}</b><br>{s["detail"]}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No notable human-leaning signals detected.")

        if result.get("formulaic_matches"):
            with st.expander(f"Formulaic phrase matches ({len(result['formulaic_matches'])})"):
                for m in result["formulaic_matches"]:
                    st.markdown(f"- **\"{m['phrase']}\"** in: _{m['sentence']}_")

        if result.get("paragraphs"):
            st.markdown("#### Paragraph-level analysis")
            for p in result["paragraphs"]:
                prob = p["ai_probability"]
                bar_color = "#dc2626" if prob >= 0.65 else ("#d97706" if prob >= 0.45 else "#16a34a")
                st.markdown(
                    f'<div class="hl-card"><div style="display:flex; justify-content:space-between;">'
                    f'<b>Paragraph {p["index"]+1}</b>'
                    f'<span style="color:{bar_color}; font-weight:700;">{p["verdict"]} ({prob*100:.0f}%)</span>'
                    f'</div><div style="margin-top:0.4rem; color:#374151; font-size:0.9rem;">{p["text"]}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.button("Improve this writing →", type="secondary",
                  on_click=switch_page, args=("Writing Improvement",),
                  kwargs={"humanize_input": st.session_state.last_text})

# ---------------------------------------------------------------------------
# WRITING IMPROVEMENT PAGE
# ---------------------------------------------------------------------------
elif page == "Writing Improvement":
    st.title("Writing Improvement")
    st.markdown(
        '<div class="hl-disclaimer">'
        "This rewrites text to sound clearer and more natural while preserving meaning. "
        "It does not attempt to evade any specific AI detector — the goal is genuinely better writing."
        "</div>",
        unsafe_allow_html=True,
    )

    default_text = st.session_state.get("humanize_input", "")
    text = st.text_area("Text to improve", value=default_text, height=200, key="humanize_input_box")

    ollama_status = call_api("GET", "/ollama/status")

    use_ollama = st.checkbox(
        "Use local LLM (Ollama) if available",
        value=st.session_state.use_ollama,
        help="Falls back automatically to rule-based rewriting if Ollama isn't running or the model isn't found.",
    )

    ollama_model_override = ""
    if use_ollama:
        if ollama_status:
            if ollama_status["reachable"]:
                pulled = ollama_status["models_pulled"]
                if pulled:
                    st.caption(f"✅ Ollama reachable at `{ollama_status['host']}`. Models pulled: {', '.join(pulled)}")
                else:
                    st.warning(
                        f"Ollama is reachable at `{ollama_status['host']}` but no models are pulled. "
                        f"Run `ollama pull <model>` first."
                    )
            else:
                st.warning(f"Ollama is not reachable at `{ollama_status['host']}`. Start it with `ollama serve`.")

        default_model = (ollama_status or {}).get("configured_default_model", settings.ollama_model)
        ollama_model_override = st.text_input(
            "Ollama model to use",
            value=st.session_state.get("ollama_model_override", default_model),
            help="Must match a model name from 'models pulled' above exactly, e.g. qwen2.5-coder:3b",
        )
        st.session_state.ollama_model_override = ollama_model_override

    if st.button("Improve Writing", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Rewriting…"):
                result = call_api(
                    "POST", "/humanize",
                    json={
                        "text": text,
                        "use_ollama": use_ollama,
                        "ollama_model": ollama_model_override or None,
                    },
                )
            if result:
                st.session_state.last_humanize = result

    result = st.session_state.get("last_humanize")
    if result:
        for w in result.get("warnings", []):
            st.warning(w)

        st.markdown(f"**Engine used:** `{result['engine_used']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original")
            st.markdown(f'<div class="hl-card">{result["original_text"]}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("#### Improved")
            st.markdown(f'<div class="hl-card">{result["revised_text"]}</div>', unsafe_allow_html=True)

        if result.get("changes"):
            st.markdown("#### Changes made")
            for c in result["changes"]:
                st.markdown(f"- **{c['rule'].replace('_', ' ')}**: `{c['before']}` → `{c['after']}` — {c['reason']}")
        else:
            st.caption("No rule-based changes were needed, or an LLM rewrite was used (see diff above).")

        val = result.get("validation", {})
        st.markdown("#### Meaning-preservation check")
        sim = val.get("semantic_similarity", 0.0)
        safe = val.get("is_safe", True)
        badge = confidence_badge("Low" if not safe else ("High" if sim > 0.6 else "Moderate"))
        st.markdown(
            f'<div class="hl-card">Semantic similarity (TF-IDF cosine): <b>{sim:.2f}</b> &nbsp; {badge}</div>',
            unsafe_allow_html=True,
        )
        if val.get("issues"):
            st.warning("Possible content differences detected:")
            for issue in val["issues"]:
                st.markdown(f"- **{issue['category']}**: `{issue['value']}` — {issue['detail']}")
        else:
            st.success("No missing numbers, dates, URLs, or quoted material detected.")

# ---------------------------------------------------------------------------
# HISTORY PAGE
# ---------------------------------------------------------------------------
elif page == "History":
    st.title("History")
    data = call_api("GET", "/history", params={"limit": 50})
    if data and data.get("items"):
        for item in data["items"]:
            with st.expander(
                f"{item['created_at'][:10]} — {item['word_count']} words — "
                f"AI-likelihood {item['ai_probability']*100:.0f}% ({item['verdict']})"
            ):
                st.write(item["text_preview"] + ("…" if len(item["text_preview"]) >= 200 else ""))
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("View full analysis", key=f"view_{item['id']}"):
                        detail = call_api("GET", f"/history/{item['id']}")
                        if detail:
                            st.json(detail["result"])
                with col2:
                    if st.button("Delete", key=f"del_{item['id']}"):
                        call_api("DELETE", f"/history/{item['id']}")
                        st.rerun()
    else:
        st.caption("No analyses yet. Run one from the Analyze page.")

# ---------------------------------------------------------------------------
# EVALUATION PAGE
# ---------------------------------------------------------------------------
elif page == "Evaluation":
    st.title("Model Evaluation")
    st.markdown(
        "Run `python evaluation/evaluate.py` from the project root to generate a fresh evaluation "
        "report against `datasets/combined_sample.csv` (or your own dataset with the same `text,label` schema)."
    )
    report_path = Path(__file__).resolve().parent.parent.parent / "evaluation" / "reports" / "latest_report.json"
    if report_path.exists():
        import json
        report = json.loads(report_path.read_text())
        st.markdown("### Latest evaluation report")
        m = report.get("metrics", {})
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m.get('accuracy', 0):.2f}")
        cols[1].metric("Precision", f"{m.get('precision', 0):.2f}")
        cols[2].metric("Recall", f"{m.get('recall', 0):.2f}")
        cols[3].metric("F1", f"{m.get('f1', 0):.2f}")
        cols2 = st.columns(3)
        cols2[0].metric("ROC-AUC", f"{m.get('roc_auc', 0):.2f}" if m.get("roc_auc") is not None else "N/A")
        cols2[1].metric("False Positive Rate", f"{m.get('false_positive_rate', 0):.2f}")
        cols2[2].metric("False Negative Rate", f"{m.get('false_negative_rate', 0):.2f}")

        st.markdown("#### Confusion matrix")
        st.write(report.get("confusion_matrix"))

        st.markdown(
            f"*Evaluated on {report.get('n_samples')} samples from `{report.get('dataset_path')}` "
            f"at {report.get('generated_at')}.*"
        )
        st.caption(
            "⚠️ This is a small bootstrap dataset (see README → Limitations). Treat these numbers as a "
            "pipeline sanity check, not a benchmark of real-world detector accuracy."
        )
    else:
        st.info("No evaluation report found yet. Run `python evaluation/evaluate.py` first.")

# ---------------------------------------------------------------------------
# SETTINGS PAGE
# ---------------------------------------------------------------------------
elif page == "Settings":
    st.title("Settings")

    st.markdown("#### Detector")
    st.session_state.sensitivity = st.slider(
        "Detector sensitivity",
        min_value=0.05, max_value=0.95, value=float(st.session_state.sensitivity), step=0.05,
        help="Lower = detector is more sensitive and flags more text as AI-leaning. Higher = more conservative.",
    )

    st.markdown("#### Local LLM (Ollama)")
    st.session_state.use_ollama = st.checkbox(
        "Default 'use local LLM' checkbox to on in Writing Improvement",
        value=st.session_state.use_ollama,
    )
    st.caption(f"Default Ollama host: `{settings.ollama_host}` · default model: `{settings.ollama_model}`")
    st.caption(
        "You can override the model per-request on the Writing Improvement page — useful if you've "
        "pulled a different model (e.g. qwen2.5-coder:3b) than the default configured here."
    )
    st.caption("Configure the host/default model via environment variables HUMANLENS_OLLAMA_HOST / HUMANLENS_OLLAMA_MODEL.")

    st.markdown("#### Storage & privacy")
    st.write(f"Database location: `{settings.db_path}`")
    st.write(f"Full text stored in history: **{'Yes' if settings.store_full_text else 'No'}**")
    st.caption(
        "Set HUMANLENS_STORE_FULL_TEXT=false to store only short previews and numeric results, "
        "not full submitted text. No text ever leaves your machine unless you enable Ollama, which "
        "itself runs entirely locally."
    )

    st.markdown("#### Analysis limits")
    st.write(f"Minimum words recommended for reliable analysis: **{settings.min_words_for_analysis}**")
    st.write(f"Maximum characters per request: **{settings.max_chars_for_analysis:,}**")
