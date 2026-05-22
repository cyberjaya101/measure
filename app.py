"""LLM Evaluation Dashboard — Streamlit entry point."""

import json
import streamlit as st
import pandas as pd

from evaluator import evaluate_response, batch_evaluate
from test_cases import TEST_CASES, get_test_case_labels
from visualizations import (
    create_radar_chart,
    create_comparison_bar_chart,
    create_heatmap,
    create_overall_scores_chart,
)
from utils import (
    score_color_class,
    score_emoji,
    score_label,
    format_score,
    results_to_json,
    dataframe_to_json,
    dataframe_to_csv,
    metric_card_html,
    METRIC_DESCRIPTIONS,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetric"] label { color: #475569 !important; font-weight: 600; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0f172a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 LLM Evaluation Dashboard")
st.markdown(
    "_Production-ready quality assurance for LLM applications — "
    "faithfulness · relevancy · context precision · hallucination detection_"
)
st.divider()

# ── Pre-compute batch results for header KPIs ──────────────────────────────────
@st.cache_data
def _get_batch_df() -> pd.DataFrame:
    return batch_evaluate(TEST_CASES)


batch_df = _get_batch_df()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Test Cases", len(batch_df))
kpi2.metric("Avg Overall Score", f"{batch_df['overall_score'].mean():.3f}")
best_row = batch_df.loc[batch_df["overall_score"].idxmax()]
worst_row = batch_df.loc[batch_df["overall_score"].idxmin()]
kpi3.metric("Best Case", best_row["label"].split(" ", 1)[-1], f"{best_row['overall_score']:.3f}")
kpi4.metric("Worst Case", worst_row["label"].split(" ", 1)[-1], f"{worst_row['overall_score']:.3f}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["📊 Single Eval", "📈 Batch Eval"])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — Single Evaluation                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab_single:
    st.subheader("Evaluate a Single Response")

    options = ["Custom Input"] + get_test_case_labels()
    selection = st.selectbox("Select test case", options, index=0)

    # Resolve inputs
    if selection == "Custom Input":
        col_q, col_c = st.columns(2)
        with col_q:
            question = st.text_area("Question", placeholder="e.g. What is the speed of light?", height=100)
        with col_c:
            context = st.text_area(
                "Retrieved Context",
                placeholder="Paste the context given to the LLM…",
                height=100,
            )
        answer = st.text_area("LLM Answer", placeholder="Paste the LLM response…", height=100)
        case_id = "custom"
        failure_note = None
    else:
        # Find the matching test case by label
        case = next(c for c in TEST_CASES if c["label"] == selection)
        question = case["question"]
        context = case["context"]
        answer = case["llm_answer"]
        case_id = case["id"]
        failure_note = case.get("failure_mode")

        with st.expander("📋 Inputs", expanded=True):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Question:**\n\n{question}")
            c1.markdown(f"**Expected Answer:**\n\n{case['ground_truth']}")
            c2.markdown(f"**Context:**\n\n{context}")
            st.markdown(f"**LLM Answer:**\n\n{answer}")
            if failure_note:
                st.info(f"**Failure mode:** {failure_note}")

    run_btn = st.button("▶ Run Evaluation", type="primary", disabled=not (question and context and answer))

    if run_btn or (selection != "Custom Input"):
        if question and context and answer:
            with st.spinner("Evaluating…"):
                results = evaluate_response(question, context, answer)

            st.divider()
            st.subheader("📈 Evaluation Results")

            # Overall score banner
            overall = results["overall_score"]
            color_cls = score_color_class(overall)
            banner_fn = getattr(st, color_cls)
            banner_fn(
                f"**Overall Score: {format_score(overall)}** — {score_emoji(overall)} {score_label(overall)}"
            )

            # 4 metric cards (color-coded)
            m1, m2, m3, m4 = st.columns(4)
            for col, metric, label in zip(
                [m1, m2, m3, m4],
                ["faithfulness", "relevancy", "precision", "hallucination"],
                ["Faithfulness", "Answer Relevancy", "Context Precision", "Hallucination"],
            ):
                with col:
                    s = results[metric]["score"]
                    st.markdown(
                        metric_card_html(metric, label, s),
                        unsafe_allow_html=True,
                    )

            st.divider()
            # Radar chart
            chart_col, explain_col = st.columns([1, 1])
            with chart_col:
                scores_dict = {
                    "faithfulness": results["faithfulness"]["score"],
                    "relevancy": results["relevancy"]["score"],
                    "precision": results["precision"]["score"],
                    "hallucination": results["hallucination"]["score"],
                }
                st.plotly_chart(
                    create_radar_chart(scores_dict, title="Score Radar"),
                    use_container_width=True,
                )

            with explain_col:
                st.markdown("### Metric Explanations")
                for metric, label in [
                    ("faithfulness", "Faithfulness"),
                    ("relevancy", "Answer Relevancy"),
                    ("precision", "Context Precision"),
                    ("hallucination", "Hallucination"),
                ]:
                    s = results[metric]["score"]
                    st.markdown(
                        f"**{score_emoji(s)} {label}** — `{s:.3f}`\n\n"
                        f"{results[metric]['explanation']}"
                    )

            # Detailed breakdown
            with st.expander("🔬 Detailed Breakdown", expanded=False):
                for metric in ["faithfulness", "relevancy", "precision", "hallucination"]:
                    st.markdown(f"#### {metric.title()}")
                    st.json(results[metric]["details"])

            # Export
            st.markdown("#### 📥 Export Results")
            exp1, exp2 = st.columns(2)
            with exp1:
                st.download_button(
                    "⬇️ Download JSON",
                    data=results_to_json(results, case_id),
                    file_name=f"eval_{case_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with exp2:
                single_row = pd.DataFrame([{
                    "case_id": case_id,
                    "overall_score": results["overall_score"],
                    "faithfulness": results["faithfulness"]["score"],
                    "relevancy": results["relevancy"]["score"],
                    "precision": results["precision"]["score"],
                    "hallucination": results["hallucination"]["score"],
                }])
                st.download_button(
                    "⬇️ Download CSV",
                    data=dataframe_to_csv(single_row),
                    file_name=f"eval_{case_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — Batch Evaluation                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab_batch:
    st.subheader("Batch Evaluation — All Test Cases")

    if st.button("▶ Run All Test Cases", type="primary"):
        with st.spinner("Running batch evaluation…"):
            df = batch_evaluate(TEST_CASES)
        st.session_state["batch_df"] = df

    # Use cached batch results if available
    if "batch_df" not in st.session_state:
        st.session_state["batch_df"] = batch_df

    df: pd.DataFrame = st.session_state["batch_df"]

    # Results table
    st.markdown("### Results Table")
    display_df = df[["label", "overall_score", "faithfulness", "relevancy", "precision", "hallucination", "failure_mode"]].copy()
    display_df.columns = ["Test Case", "Overall", "Faithfulness", "Relevancy", "Precision", "Hallucination", "Failure Mode"]
    score_cols = ["Overall", "Faithfulness", "Relevancy", "Precision", "Hallucination"]

    def _color_score(val):
        if not isinstance(val, float):
            return ""
        if val >= 0.7:
            return "background-color: #d4edda; color: #155724"
        if val >= 0.4:
            return "background-color: #fff3cd; color: #856404"
        return "background-color: #f8d7da; color: #721c24"

    st.dataframe(
        display_df.style
            .map(_color_score, subset=score_cols)
            .format("{:.3f}", subset=score_cols),
        use_container_width=True,
        hide_index=True,
    )

    # Summary stats
    st.markdown("### Summary Statistics")
    metric_cols = ["overall_score", "faithfulness", "relevancy", "precision", "hallucination"]
    stats = df[metric_cols].agg(["mean", "min", "max"]).round(3)
    stats.index = ["Mean", "Min", "Max"]
    stats.columns = ["Overall", "Faithfulness", "Relevancy", "Precision", "Hallucination"]
    st.dataframe(stats, use_container_width=True)

    st.divider()

    # Charts
    chart_a, chart_b = st.columns([1, 1])
    with chart_a:
        st.plotly_chart(create_overall_scores_chart(df), use_container_width=True)
    with chart_b:
        st.plotly_chart(create_heatmap(df), use_container_width=True)

    st.plotly_chart(create_comparison_bar_chart(df), use_container_width=True)

    # Export
    st.markdown("#### 📥 Export Results")
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "⬇️ Download JSON",
            data=dataframe_to_json(df),
            file_name="batch_eval_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            "⬇️ Download CSV",
            data=dataframe_to_csv(df),
            file_name="batch_eval_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 About")
    st.markdown(
        "This dashboard evaluates LLM response quality using 4 key metrics "
        "critical for production deployments."
    )
    st.divider()

    st.markdown("## 🧪 Example Test Cases")
    st.caption("Pick any preset in **Single Eval** or run all in **Batch Eval**.")
    for case in TEST_CASES:
        with st.expander(case["label"], expanded=False):
            st.markdown(f"**Q:** {case['question']}")
            st.markdown(f"**Failure:** {case.get('failure_mode', '—')}")
            st.caption(f"ID: `{case['id']}`")

    st.divider()

    st.markdown("## 📐 Metrics Reference")
    for metric, desc in METRIC_DESCRIPTIONS.items():
        with st.expander(metric.title()):
            st.markdown(desc)

    st.divider()
    st.markdown("## ⚙️ Score Thresholds")
    st.markdown(
        "| Label | Range |\n"
        "|-------|-------|\n"
        "| 🟢 Excellent | ≥ 0.85 |\n"
        "| 🟡 Good      | 0.70–0.84 |\n"
        "| 🟠 Fair      | 0.40–0.69 |\n"
        "| 🔴 Poor      | < 0.40 |"
    )

    st.divider()
    st.markdown("## 🛠 Stack")
    st.markdown(
        "- **Streamlit** — UI\n"
        "- **Plotly** — charts\n"
        "- **pandas** — data layer\n"
        "- **Python 3.12** — runtime"
    )
    st.divider()
    st.caption("Project 4/5 · AI Portfolio · LLM Eval Dashboard")
