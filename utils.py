"""Utility helpers for the evaluation dashboard."""

import json
from datetime import datetime


def score_color_class(score: float) -> str:
    """Return a Streamlit status category string based on score.

    Returns:
        'success' | 'warning' | 'error'
    """
    if score >= 0.7:
        return "success"
    if score >= 0.4:
        return "warning"
    return "error"


def score_emoji(score: float) -> str:
    """Return a descriptive emoji for a given score."""
    if score >= 0.85:
        return "🟢"
    if score >= 0.7:
        return "🟡"
    if score >= 0.4:
        return "🟠"
    return "🔴"


def score_label(score: float) -> str:
    """Return a human-readable quality label for a score."""
    if score >= 0.85:
        return "Excellent"
    if score >= 0.7:
        return "Good"
    if score >= 0.4:
        return "Fair"
    return "Poor"


def format_score(score: float) -> str:
    """Format a score as a percentage string, e.g. '87.5%'."""
    return f"{score * 100:.1f}%"


def results_to_json(results: dict, case_id: str = "custom") -> str:
    """Serialise evaluation results to a JSON string for download.

    Args:
        results: Dict returned by evaluate_response().
        case_id: Identifier for the evaluated case.

    Returns:
        Pretty-printed JSON string.
    """
    export = {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "case_id": case_id,
        "overall_score": results["overall_score"],
        "metrics": {
            metric: {
                "score": results[metric]["score"],
                "explanation": results[metric]["explanation"],
            }
            for metric in ("faithfulness", "relevancy", "precision", "hallucination")
        },
    }
    return json.dumps(export, indent=2)


def dataframe_to_json(df) -> str:
    """Serialise a batch results DataFrame to JSON.

    Args:
        df: pandas DataFrame from batch_evaluate().

    Returns:
        JSON string.
    """
    return df.to_json(orient="records", indent=2)


def dataframe_to_csv(df) -> str:
    """Serialise a batch results DataFrame to CSV."""
    return df.to_csv(index=False)


METRIC_CARD_STYLES = {
    "faithfulness": {"color": "#2563eb", "bg": "#eff6ff", "icon": "📎"},
    "relevancy": {"color": "#059669", "bg": "#ecfdf5", "icon": "🎯"},
    "precision": {"color": "#7c3aed", "bg": "#f5f3ff", "icon": "🔍"},
    "hallucination": {"color": "#d97706", "bg": "#fffbeb", "icon": "🛡️"},
}


def metric_card_html(metric_key: str, label: str, score: float) -> str:
    """Return an HTML metric card with distinct colors per metric."""
    style = METRIC_CARD_STYLES[metric_key]
    border = "#22c55e" if score >= 0.7 else "#eab308" if score >= 0.4 else "#ef4444"
    return f"""
    <div style="
        background: {style['bg']};
        border-left: 4px solid {style['color']};
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    ">
        <div style="color: {style['color']}; font-size: 0.85rem; font-weight: 600;">
            {style['icon']} {label}
        </div>
        <div style="font-size: 1.75rem; font-weight: 700; color: #111827; margin: 0.25rem 0;">
            {format_score(score)}
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="
                background: {border};
                color: white;
                font-size: 0.75rem;
                font-weight: 600;
                padding: 0.15rem 0.5rem;
                border-radius: 999px;
            ">{score_emoji(score)} {score_label(score)}</span>
            <span style="color: #6b7280; font-size: 0.8rem;">{score:.3f}</span>
        </div>
    </div>
    """


METRIC_DESCRIPTIONS = {
    "faithfulness": (
        "Measures how well the LLM answer is grounded in the provided context. "
        "High score = claims traceable to context. Low score = fabricated content."
    ),
    "relevancy": (
        "Measures whether the answer actually addresses the question. "
        "High score = on-topic. Low score = off-topic or evasive."
    ),
    "precision": (
        "Measures how much of the retrieved context was actually used. "
        "High score = tight, relevant context. Low score = noisy retrieval."
    ),
    "hallucination": (
        "Detects claims in the answer not supported by the context. "
        "Score of 1.0 = fully grounded. Score near 0 = mostly hallucinated."
    ),
}
