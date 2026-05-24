"""
app.py — AgriScreen
Pivot & Co · May 2026

Launch: streamlit run app.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

import streamlit as st
import pandas as pd
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv

from utils.company_meta import COMPANY_INFO, enrich_scores_df

load_dotenv()

# ── Tooltips / Descriptions ───────────────────────────────────
ALERT_DESCRIPTIONS = {
    'RUNWAY_CRITICAL' : "Less than 12 months of cash remaining. Immediate financial risk.",
    'REVENUE_DECLINE' : "Revenue dropped year-over-year. Investigate cause urgently.",
    'ESG_ALERT'       : "ESG composite below the 60/100 minimum from the framework.",
    'GOVERNANCE_FLAG' : "Senior leadership change or governance risk detected in news.",
    'FUNDRAISE_ACTIVE': "Active fundraising process detected — engage before round closes.",
    'STRATEGIC_EXIT'  : "M&A or strategic sale detected — value could change rapidly.",
    'SCORE_PRIORITY'  : "Composite score crossed the 70/100 threshold — eligible for DD.",
}

SUGGESTED_ACTIONS = {
    'PRIORITY': [
        "Schedule a first call with management within 2 weeks",
        "Request data room access and detailed financials",
        "Brief the investment committee on the opportunity",
    ],
    'WATCH': [
        "Set up news monitoring alerts for catalyst events",
        "Schedule a 90-day check-in call",
        "Track key milestones (regulatory, fundraising, partnerships)",
    ],
    'LOW PRIORITY': [
        "Add to passive watchlist",
        "Re-evaluate in 6 months unless major catalyst emerges",
        "Skip outreach unless analyst flag changes",
    ],
}

SCORE_TOOLTIPS = {
    'financial' : "Revenue CAGR 3y + Gross margin + Runway + Burn efficiency. CSV-driven, deterministic.",
    'technology': "Patents (hard signal) + AI-extracted IP strength, data moat, integrations, benchmarks.",
    'market'    : "Hectare coverage + Investor conviction + AI-extracted TAM, geography, competition, partnerships.",
    'esg'       : "AI extracts E/S/G evidence. Python applies the ESG framework rules from the framework doc.",
}


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "AgriScreen · Pivot & Co",
    page_icon  = "🌱",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; letter-spacing: -0.01em; }

.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 2px solid #e5e7eb; background: transparent; }
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: 0.875rem;
    padding: 10px 20px; color: #6b7280; border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #166534 !important;
    border-bottom: 2px solid #166534 !important;
    background: #f0fdf4 !important;
}

.flag-priority { background:#dcfce7; color:#166534; border:1px solid #86efac; padding:3px 12px; border-radius:14px; font-weight:600; font-size:0.78rem; display:inline-block; }
.flag-watch    { background:#fef9c3; color:#854d0e; border:1px solid #fde047; padding:3px 12px; border-radius:14px; font-weight:600; font-size:0.78rem; display:inline-block; }
.flag-low      { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; padding:3px 12px; border-radius:14px; font-weight:600; font-size:0.78rem; display:inline-block; }

.score-bar-wrap { background:#f1f5f9; border-radius:6px; height:10px; margin:4px 0 12px; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:6px; transition: width 0.4s; }
.score-bar-red    { background: linear-gradient(90deg, #ef4444, #f87171); }
.score-bar-amber  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.score-bar-green  { background: linear-gradient(90deg, #16a34a, #4ade80); }

.compare-card {
    background:#fafafa; border:1px solid #e5e7eb; border-radius:10px;
    padding:18px 20px; margin-bottom:12px;
}
.compare-card h4 { margin:0 0 8px 0; font-size:1rem; }

.alert-high   { border-left: 4px solid #ef4444; background: #fff5f5; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 6px 0; }
.alert-medium { border-left: 4px solid #f59e0b; background: #fffbeb; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 6px 0; }
.alert-low    { border-left: 4px solid #3b82f6; background: #eff6ff; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 6px 0; }

.source-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 10px 14px; margin: 4px 0;
    font-family: 'DM Mono', monospace; font-size: 0.78rem;
    line-height: 1.6; color: #374151;
}

.note-box { background:#fafafa; border:1px solid #e5e7eb; border-radius:10px; padding:28px 32px; margin-top:8px; line-height:1.8; }

[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #6b7280 !important; }
[data-testid="stMetricValue"] { font-family: 'Playfair Display', serif !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────

def score_bar_html(score: float, max_score: float = 25) -> str:
    pct = min(100, (score / max_score) * 100)
    if pct >= 70:    cls = "score-bar-green"
    elif pct >= 50:  cls = "score-bar-amber"
    else:            cls = "score-bar-red"
    return f'<div class="score-bar-wrap"><div class="score-bar-fill {cls}" style="width:{pct}%"></div></div>'


def render_sources(sources, label_prefix=""):
    if not sources:
        return
    with st.expander(f"📚 {label_prefix}{len(sources)} source chunks retrieved"):
        for src in sources:
            meta = src['chunk'].metadata
            co  = f" · {meta.company}" if meta.company else ''
            sec = f" · {meta.section}" if meta.section else ''
            nid = f" · {meta.news_id}" if meta.news_id else ''
            dt  = f" · [{meta.doc_type}]"
            st.markdown(
                f'<div class="source-box">'
                f'<strong>[{src["rank"]}] {meta.source_file}</strong>'
                f'{dt}{co}{sec}{nid} · sim={src["score"]:.3f}<br><br>'
                f'{src["chunk"].text[:300]}…'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_suggested_steps(flag: str, header_text: str = "📌 Suggested next steps"):
    suggestions = SUGGESTED_ACTIONS.get(flag, [])
    if not suggestions:
        return
    bg = {'PRIORITY':'#dcfce7','WATCH':'#fef9c3','LOW PRIORITY':'#fee2e2'}.get(flag, '#f3f4f6')
    border = {'PRIORITY':'#16a34a','WATCH':'#eab308','LOW PRIORITY':'#ef4444'}.get(flag, '#9ca3af')
    steps_html = ''.join(f'<li>{s}</li>' for s in suggestions)
    st.markdown(
        f'<div style="background:{bg}; border-left:4px solid {border}; '
        f'border-radius:0 8px 8px 0; padding:14px 18px; margin-top:16px;">'
        f'<strong>{header_text}</strong>'
        f'<ul style="margin:8px 0 0 0; padding-left:20px;">{steps_html}</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Cached loaders ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising AI engine…")
def get_qa_engine():
    from rag.qa_engine import QAEngine
    return QAEngine(index_dir="outputs/index", api_key=os.getenv("GROQ_API_KEY"))


@st.cache_data(show_spinner="Running scoring pipeline…")
def get_scores():
    from scoring.composite import run_scoring
    df = run_scoring(index_dir="outputs/index", api_key=os.getenv("GROQ_API_KEY"))
    return enrich_scores_df(df)


@st.cache_data(show_spinner="Evaluating alerts…")
def get_alerts(_scores_hash):
    from monitoring.alerts import run_alerts
    from ingestion.loader import load_financials
    return run_alerts(get_scores(), load_financials(), index_dir="outputs/index")


def get_load_time():
    if 'load_time' not in st.session_state:
        st.session_state['load_time'] = datetime.now()
    return st.session_state['load_time']


# ── Bootstrap ──────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY", "")
scores_df, alerts, qa_engine = None, [], None

if api_key:
    try:
        scores_df = get_scores()
        alerts    = get_alerts(hash(str(scores_df['total'].tolist())))
        qa_engine = get_qa_engine()
    except Exception as e:
        st.error(f"Initialisation error: {e}")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 AgriScreen")
    st.caption("Pivot & Co · Agricultural Impact Fund · May 2026")
    st.divider()

    if not api_key:
        st.error("GROQ_API_KEY not set")
        entered = st.text_input("Enter Groq API key:", type="password")
        if entered:
            os.environ["GROQ_API_KEY"] = entered
            st.rerun()
    else:
        st.success("✓ Groq API connected")

    if scores_df is not None:
        n_priority = int((scores_df['flag'] == 'PRIORITY').sum())
        n_watch    = int((scores_df['flag'] == 'WATCH').sum())
        n_high     = sum(1 for a in alerts if a['severity'] == 'HIGH')

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Companies", len(scores_df))
        c2.metric("🟢 Priority", n_priority)
        c1.metric("🟡 Watch", n_watch)
        c2.metric("🔴 High Alerts", n_high)

        st.divider()
        if st.button("♻️  Refresh Scores", use_container_width=True):
            st.cache_data.clear()
            if 'load_time' in st.session_state:
                del st.session_state['load_time']
            st.rerun()

    st.divider()
    load_time = get_load_time()
    st.caption(f"📅 Data last updated\n\n{load_time.strftime('%d %b %Y · %H:%M')}")

    st.divider()
    st.caption(
        "LLMs used for grounded signal extraction only. "
        "All scoring, normalization, and thresholding is "
        "deterministic Python."
    )


# ── Guard ──────────────────────────────────────────────────────
if scores_df is None:
    st.title("🌱 AgriScreen")
    st.warning("Set your GROQ_API_KEY in the sidebar or `.env` file to start.")
    st.stop()


# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Dashboard",
    "💬  Analyst Chat",
    "🚨  Alerts",
    "📋  Company Notes",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Portfolio Scoring Dashboard")

    # Scoring ranges inline beneath the title
    st.markdown(
        '<div style="margin: 4px 0 20px 0; font-size: 0.9rem; color: #4b5563; line-height: 1.8;">'
        'Composite investment scores across Financial (F), Technology (T), Market (M), '
        'and ESG (E) dimensions.<br>'
        '<span class="flag-priority">PRIORITY</span> ≥ 70 — advance to Due Diligence'
        '&nbsp;&nbsp;·&nbsp;&nbsp;'
        '<span class="flag-watch">WATCH</span> 50–69 — monitor for 90 days'
        '&nbsp;&nbsp;·&nbsp;&nbsp;'
        '<span class="flag-low">LOW</span> &lt; 50 — deprioritise'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Filters: flag + sort ─────────────────────────────────
    col_filt1, col_filt2 = st.columns([1, 1])
    with col_filt1:
        flag_filter = st.multiselect(
            "Filter by flag",
            ["PRIORITY", "WATCH", "LOW PRIORITY"],
            default=["PRIORITY", "WATCH", "LOW PRIORITY"],
        )
    with col_filt2:
        sort_by = st.selectbox(
            "Sort by",
            [
                "Total ↓", "Total ↑",
                "Financial ↓", "Financial ↑",
                "Technology ↓", "Technology ↑",
                "Market ↓", "Market ↑",
                "ESG ↓", "ESG ↑",
            ],
        )

    filtered_df = scores_df[scores_df['flag'].isin(flag_filter)].copy()
    sort_map = {
        "Total ↓": ('total', False), "Total ↑": ('total', True),
        "Financial ↓": ('financial', False), "Financial ↑": ('financial', True),
        "Technology ↓": ('technology', False), "Technology ↑": ('technology', True),
        "Market ↓": ('market', False), "Market ↑": ('market', True),
        "ESG ↓": ('esg', False), "ESG ↑": ('esg', True),
    }
    sort_col, sort_asc = sort_map[sort_by]
    filtered_df = filtered_df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    st.caption(f"Showing {len(filtered_df)} of {len(scores_df)} companies")
    st.divider()

    if filtered_df.empty:
        st.info("No companies match the selected filter.")
    else:
        cols = st.columns(min(len(filtered_df), 6))
        for i, (_, row) in enumerate(filtered_df.iterrows()):
            if i >= 6: break
            cols[i].metric(
                label = row['company'].split(' ')[0],
                value = f"{row['total']:.1f}",
                delta = row['flag'],
            )

        st.divider()

        display_df = filtered_df[['company', 'financial', 'technology', 'market', 'esg', 'total', 'flag']].copy()
        display_df.columns = ['Company', 'F /25', 'T /25', 'M /25', 'E /25', 'Total /100', 'Flag']

        def _flag_color(val):
            return {
                'PRIORITY'    : 'background-color:#dcfce7; color:#166534; font-weight:600',
                'WATCH'       : 'background-color:#fef9c3; color:#854d0e; font-weight:600',
                'LOW PRIORITY': 'background-color:#fee2e2; color:#991b1b; font-weight:600',
            }.get(val, '')

        styled = (
            display_df.style
            .applymap(_flag_color, subset=['Flag'])
            .background_gradient(subset=['Total /100'], cmap='RdYlGn', vmin=20, vmax=90)
            .background_gradient(subset=['F /25', 'T /25', 'M /25', 'E /25'], cmap='Blues', vmin=0, vmax=25)
            .format({'F /25':'{:.1f}', 'T /25':'{:.1f}', 'M /25':'{:.1f}', 'E /25':'{:.1f}', 'Total /100':'{:.1f}'})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # SCORE BREAKDOWN
    # ══════════════════════════════════════════════════════════
    st.divider()
    st.markdown("## Score Breakdown")

    selected = st.selectbox(
        "Select company:",
        scores_df['company'].tolist(),
        key="db_company",
    )

    if selected:
        row = scores_df[scores_df['company'] == selected].iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Financial",  f"{row['financial']:.1f} / 25",  help=SCORE_TOOLTIPS['financial'])
        c2.metric("Technology", f"{row['technology']:.1f} / 25", help=SCORE_TOOLTIPS['technology'])
        c3.metric("Market",     f"{row['market']:.1f} / 25",     help=SCORE_TOOLTIPS['market'])
        c4.metric("ESG",        f"{row['esg']:.1f} / 25",        help=SCORE_TOOLTIPS['esg'])
        c5.metric("Total",      f"{row['total']:.1f} / 100",     delta=row['flag'])

        st.markdown("**Score bars**")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown("Financial")
            st.markdown(score_bar_html(row['financial']), unsafe_allow_html=True)
        with b2:
            st.markdown("Technology")
            st.markdown(score_bar_html(row['technology']), unsafe_allow_html=True)
        with b3:
            st.markdown("Market")
            st.markdown(score_bar_html(row['market']), unsafe_allow_html=True)
        with b4:
            st.markdown("ESG")
            st.markdown(score_bar_html(row['esg']), unsafe_allow_html=True)

        render_suggested_steps(row['flag'])

        st.markdown("**Sub-component breakdown**")
        col_f, col_t, col_m, col_e = st.columns(4)

        def _breakdown_table(col, title, breakdown):
            col.markdown(f"**{title}**")
            if isinstance(breakdown, dict):
                for k, v in breakdown.items():
                    col.write(f"`{k}` → **{v}**")

        _breakdown_table(col_f, "Financial",  row.get('f_breakdown', {}))
        _breakdown_table(col_t, "Technology", row.get('t_breakdown', {}))
        _breakdown_table(col_m, "Market",     row.get('m_breakdown', {}))
        _breakdown_table(col_e, "ESG",        row.get('e_breakdown', {}))

        esg_flags = row.get('esg_flags', [])
        if esg_flags:
            st.warning("**ESG Red Flags:**  " + "  |  ".join(esg_flags))

        with st.expander("📖 Scoring formula reference"):
            st.markdown("""
**Financial (0–25):** Revenue CAGR 3y (0–10) + Gross margin / 80 × 7 (0–7) + Runway tiered (0–5) + EBITDA margin tiered (0–3)

**Technology (0–25):** `has_patent` → 8 pts hard signal + LLM-extracted ip_strength (0–5) + data_moat (0–4) + integration_maturity (0–4) + benchmark_evidence (0–4)

**Market (0–25):** log₁₀(coverage_ha) scaled (0–7) + total raised tiered (0–8) + LLM-extracted TAM (0–3) + geography (0–3) + competition (0–2) + partnerships (0–2)

**ESG (0–25):** RAG extracts E/S/G evidence → Python applies ESG framework rules → (E+S+G)/3 → ÷4

**LLMs used for extraction only. All scoring is deterministic Python.**
""")

    # ══════════════════════════════════════════════════════════
    # COMPARE COMPANIES (read-only, no Q&A)
    # ══════════════════════════════════════════════════════════
    st.divider()
    st.markdown("## 🔄 Compare Companies")
    st.caption("Select 2 to 4 companies to compare side-by-side. For comparative questions, use the Analyst Chat tab.")

    selected_companies = st.multiselect(
        "Select companies (2–4):",
        scores_df['company'].tolist(),
        default=scores_df['company'].head(2).tolist(),
        max_selections=4,
        key="compare_select",
    )

    if len(selected_companies) < 2:
        st.info("Select at least 2 companies to compare.")
    else:
        n = len(selected_companies)

        # Scorecards
        st.markdown("### Scorecards")
        cols = st.columns(n)
        for col, company in zip(cols, selected_companies):
            row = scores_df[scores_df['company'] == company].iloc[0]
            meta = COMPANY_INFO.get(company, {})
            flag = row['flag']
            flag_cls = {'PRIORITY':'flag-priority','WATCH':'flag-watch','LOW PRIORITY':'flag-low'}[flag]

            with col:
                st.markdown(
                    f'<div class="compare-card">'
                    f'<h4>{company}</h4>'
                    f'<div style="color:#6b7280;font-size:0.85rem;margin-bottom:8px;">'
                    f'{meta.get("country_name","?")} · {meta.get("sub_sector","?")} · Founded {meta.get("founded","?")}'
                    f'</div>'
                    f'<div style="font-size:2rem;font-weight:600;font-family:Playfair Display,serif;">{row["total"]:.1f}<span style="font-size:1rem;color:#6b7280;">/100</span></div>'
                    f'<div><span class="{flag_cls}">{flag}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for dim, label in [('financial','Financial'),('technology','Technology'),('market','Market'),('esg','ESG')]:
                    val = row[dim]
                    st.markdown(f"<small>{label}: <strong>{val:.1f}/25</strong></small>", unsafe_allow_html=True)
                    st.markdown(score_bar_html(val), unsafe_allow_html=True)

        # Alerts
        st.markdown("### Active Alerts")
        cols = st.columns(n)
        for col, company in zip(cols, selected_companies):
            co_alerts = [a for a in alerts if a['company'] == company]
            with col:
                if not co_alerts:
                    st.success("No active alerts")
                else:
                    nh = sum(1 for a in co_alerts if a['severity'] == 'HIGH')
                    nm = sum(1 for a in co_alerts if a['severity'] == 'MEDIUM')
                    nl = sum(1 for a in co_alerts if a['severity'] == 'LOW')
                    st.markdown(f"**{len(co_alerts)} alerts:**  🔴 {nh}  🟡 {nm}  🔵 {nl}")
                    for a in co_alerts[:5]:
                        icon = {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🔵'}[a['severity']]
                        st.markdown(f"- {icon} **{a['alert_type']}**: {a['trigger']}")


# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYST CHAT (with comparison mode)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Analyst Chat")
    st.caption(
        "Ask any question about the portfolio. Answers stream in token-by-token, "
        "grounded in source documents. If the corpus can't answer, it says so."
    )

    # ── Comparison toggle ────────────────────────────────────
    use_comparison = st.toggle(
        "🔄 Comparison mode — get parallel answers for 2-4 selected companies",
        value=False,
        key="chat_compare_toggle",
        help="When on, the same question is asked separately for each selected company "
             "with company-filtered retrieval. Best for 'how does A compare to B' questions.",
    )

    selected_compare = []
    if use_comparison:
        selected_compare = st.multiselect(
            "Companies to compare (2–4):",
            scores_df['company'].tolist(),
            default=scores_df['company'].head(2).tolist(),
            max_selections=4,
            key="chat_compare_select",
        )
        if len(selected_compare) < 2:
            st.info("Select at least 2 companies for comparison mode, or toggle off for normal chat.")

    st.markdown("**Quick-start queries:**")
    ex_cols = st.columns(3)
    examples = [
        "What is AquaGrow's gross margin and when did they become profitable?",
        "Which companies have active fundraising processes? What amounts are they targeting?",
        "Compare the ESG impact claims of Verdant Farms and SoilSense AI.",
    ]
    for i, (col, ex) in enumerate(zip(ex_cols, examples)):
        if col.button(f"💬 Example {i+1}", key=f"ex_{i}", use_container_width=True, help=ex):
            st.session_state['pending_query'] = ex

    st.divider()

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Render history
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            if msg.get('is_comparison'):
                # Render comparison columns
                n = len(msg['comparison_results'])
                cols = st.columns(n)
                for col, (company, data) in zip(cols, msg['comparison_results'].items()):
                    with col:
                        st.markdown(f"**{company}**")
                        st.markdown(data['answer'])
                        if data.get('sources'):
                            render_sources(data['sources'])
            else:
                st.markdown(msg['content'])
                if msg['role'] == 'assistant' and msg.get('sources'):
                    render_sources(msg['sources'])

    # Chat input
    pending = st.session_state.pop('pending_query', None)
    query   = st.chat_input("Ask a question about the portfolio…") or pending

    if query:
        # ── Comparison mode ──────────────────────────────────
        if use_comparison and len(selected_compare) >= 2:
            st.session_state['chat_history'].append({
                'role': 'user',
                'content': f"[Comparison: {', '.join(selected_compare)}]  {query}",
            })

            with st.chat_message('user'):
                st.markdown(f"**Comparison ({len(selected_compare)} companies):**  {query}")

            with st.chat_message('assistant'):
                n = len(selected_compare)
                cols = st.columns(n)
                comparison_results = {}

                # Retrieve all in parallel
                with st.spinner(f"Retrieving for {n} companies in parallel…"):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
                        futures = {
                            executor.submit(qa_engine.ask_stream, query, 5, company): company
                            for company in selected_compare
                        }
                        prep = {}
                        for future in concurrent.futures.as_completed(futures):
                            company = futures[future]
                            generator, sources = future.result()
                            prep[company] = (generator, sources)

                # Stream into columns in original order
                for col, company in zip(cols, selected_compare):
                    generator, sources = prep[company]
                    with col:
                        st.markdown(f"**{company}**")
                        answer = st.write_stream(generator)
                        render_sources(sources)
                        comparison_results[company] = {
                            'answer'  : answer,
                            'sources' : sources,
                        }

            st.session_state['chat_history'].append({
                'role'              : 'assistant',
                'is_comparison'     : True,
                'comparison_results': comparison_results,
            })

        # ── Normal mode ──────────────────────────────────────
        else:
            st.session_state['chat_history'].append({'role': 'user', 'content': query})

            with st.chat_message('user'):
                st.markdown(query)

            with st.chat_message('assistant'):
                with st.spinner("Retrieving documents…"):
                    generator, sources = qa_engine.ask_stream(query, k=8)
                full_answer = st.write_stream(generator)
                render_sources(sources)

            st.session_state['chat_history'].append({
                'role'    : 'assistant',
                'content' : full_answer,
                'sources' : sources,
            })

    if st.session_state.get('chat_history'):
        if st.button("🗑  Clear chat", key="clear_chat"):
            st.session_state['chat_history'] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 3 — ALERTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Monitoring Alerts")
    st.caption("Hover over any alert to see its description.")

    n_high   = sum(1 for a in alerts if a['severity'] == 'HIGH')
    n_medium = sum(1 for a in alerts if a['severity'] == 'MEDIUM')
    n_low    = sum(1 for a in alerts if a['severity'] == 'LOW')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts", len(alerts))
    c2.metric("🔴 High",   n_high,   help="Immediate action required")
    c3.metric("🟡 Medium", n_medium, help="Needs investigation but not urgent")
    c4.metric("🔵 Low",    n_low,    help="Low priority — informational signals")

    st.divider()

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sev_filter = st.multiselect("Severity", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM","LOW"])
    with col_f2:
        co_filter = st.selectbox("Company", ["All"] + sorted({a['company'] for a in alerts}))
    with col_f3:
        type_filter = st.selectbox("Alert type", ["All"] + sorted({a['alert_type'] for a in alerts}))

    filtered = [
        a for a in alerts
        if a['severity'] in sev_filter
        and (co_filter == "All" or a['company'] == co_filter)
        and (type_filter == "All" or a['alert_type'] == type_filter)
    ]

    st.caption(f"Showing {len(filtered)} of {len(alerts)} alerts")

    with st.expander("📖 What each alert type means"):
        for alert_type, desc in ALERT_DESCRIPTIONS.items():
            st.markdown(f"**{alert_type}** — {desc}")

    st.divider()

    icons = {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🔵'}
    cls   = {'HIGH':'alert-high','MEDIUM':'alert-medium','LOW':'alert-low'}

    if not filtered:
        st.info("No alerts match the selected filters.")

    for a in filtered:
        tooltip = ALERT_DESCRIPTIONS.get(a['alert_type'], '')
        st.markdown(
            f'<div class="{cls[a["severity"]]}" title="{tooltip}">'
            f'<strong>{icons[a["severity"]]} {a["alert_type"]}</strong>'
            f' &nbsp;·&nbsp; {a["company"]}'
            f'<div style="font-size:0.8rem; color:#6b7280; margin-top:4px; font-style:italic;">{tooltip}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"**Trigger:** `{a['trigger']}`")
            st.markdown(f"**Evidence:** {a['evidence']}")
            st.caption(f"Source: {a['source']}")
        with col_b:
            st.info(f"**Action**\n\n{a['action']}")
        st.divider()


# ══════════════════════════════════════════════════════════════
# TAB 4 — COMPANY NOTES
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Company Investment Notes")
    st.caption(
        "AI-generated structured investment notes grounded in source documents. "
        "Download in your preferred format."
    )

    selected_co = st.selectbox(
        "Select company:",
        scores_df['company'].tolist(),
        key="notes_co",
    )

    if selected_co:
        row        = scores_df[scores_df['company'] == selected_co].iloc[0]
        co_alerts  = [a for a in alerts if a['company'] == selected_co]
        flag       = row['flag']

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Score", f"{row['total']:.1f} / 100")
        c2.metric("Flag", flag)
        c3.metric("Financial", f"{row['financial']:.1f}",  help=SCORE_TOOLTIPS['financial'])
        c4.metric("Technology", f"{row['technology']:.1f}", help=SCORE_TOOLTIPS['technology'])
        c5.metric("Alerts", len(co_alerts))

        st.divider()

        note_key = f"note__{selected_co}"
        bc1, bc2, _ = st.columns([2, 2, 6])

        if bc1.button(
            "✨ Generate Note" if note_key not in st.session_state else "✨ Regenerate",
            type="primary", key=f"gen_{selected_co}",
        ):
            with st.spinner(f"Generating note for {selected_co}…"):
                from monitoring.company_notes import generate_note
                note = generate_note(
                    company_name=selected_co, score_row=row,
                    alerts=alerts, qa_engine=qa_engine,
                )
                st.session_state[note_key] = note
                os.makedirs("outputs/notes", exist_ok=True)
                safe = selected_co.replace(' ', '_')
                with open(f"outputs/notes/{safe}.md", 'w') as f:
                    f.write(note)

        if note_key in st.session_state:
            if bc2.button("🗑  Clear", key=f"clear_{selected_co}"):
                del st.session_state[note_key]
                st.rerun()

        if note_key in st.session_state:
            note_md = st.session_state[note_key]

            # Download buttons
            from utils.exporters import export_to_txt, export_to_docx, export_to_pdf
            safe_name = selected_co.replace(' ', '_')
            dl1, dl2, dl3, _ = st.columns([1, 1, 1, 5])

            dl1.download_button(
                label="📄 .txt",
                data=export_to_txt(note_md),
                file_name=f"{safe_name}_note.txt",
                mime="text/plain",
                use_container_width=True,
            )
            dl2.download_button(
                label="📘 .docx",
                data=export_to_docx(note_md, selected_co),
                file_name=f"{safe_name}_note.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
            dl3.download_button(
                label="📕 .pdf",
                data=export_to_pdf(note_md, selected_co),
                file_name=f"{safe_name}_note.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.markdown(
                f'<div class="note-box">{note_md}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(f"Click **Generate Note** to create an AI-written investment note for **{selected_co}**.")

            render_suggested_steps(flag, header_text=f"📌 Suggested next steps for {flag}")

            if co_alerts:
                st.markdown("**Active alerts for this company:**")
                for a in co_alerts:
                    icon = {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🔵'}.get(a['severity'], '⚪')
                    desc = ALERT_DESCRIPTIONS.get(a['alert_type'], '')
                    st.markdown(
                        f"- {icon} **{a['alert_type']}** — {a['trigger']}  \n"
                        f"  <small style='color:#6b7280; font-style:italic;'>{desc}</small>",
                        unsafe_allow_html=True,
                    )