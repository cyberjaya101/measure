"""Plotly chart builders for the evaluation dashboard."""

import plotly.graph_objects as go
import pandas as pd


_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "relevancy": "Relevancy",
    "precision": "Precision",
    "hallucination": "Hallucination",
}

_COLORS = {
    "perfect": "#2ecc71",
    "hallucinated": "#e74c3c",
    "off_topic": "#e67e22",
    "irrelevant_context": "#9b59b6",
    "mixed_quality": "#3498db",
    "custom": "#1abc9c",
    "default": "#3498db",
}

_METRIC_COLORS = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]


def create_radar_chart(scores: dict[str, float], title: str = "Evaluation Radar") -> go.Figure:
    """4-axis radar chart for a single evaluation result.

    Args:
        scores: Dict with keys faithfulness, relevancy, precision, hallucination.
        title: Chart title shown above the plot.

    Returns:
        Plotly Figure object.
    """
    metrics = list(_METRIC_LABELS.values())
    values = [
        scores.get("faithfulness", 0),
        scores.get("relevancy", 0),
        scores.get("precision", 0),
        scores.get("hallucination", 0),
    ]
    # Close the polygon
    values_closed = values + [values[0]]
    metrics_closed = metrics + [metrics[0]]

    fig = go.Figure()

    # Reference zone: "good" threshold at 0.7
    fig.add_trace(
        go.Scatterpolar(
            r=[0.7] * (len(metrics) + 1),
            theta=metrics_closed,
            fill="toself",
            fillcolor="rgba(46, 204, 113, 0.1)",
            line=dict(color="rgba(46, 204, 113, 0.4)", dash="dash"),
            name="Good threshold (0.7)",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=metrics_closed,
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.25)",
            line=dict(color="#3498db", width=2),
            name="Scores",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
        ),
        showlegend=True,
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_comparison_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing all 4 metrics across test cases.

    Args:
        df: DataFrame with columns id, label, faithfulness, relevancy,
            precision, hallucination.

    Returns:
        Plotly Figure object.
    """
    metric_keys = ["faithfulness", "relevancy", "precision", "hallucination"]
    fig = go.Figure()

    for metric, color in zip(metric_keys, _METRIC_COLORS):
        fig.add_trace(
            go.Bar(
                name=_METRIC_LABELS[metric],
                x=df["label"],
                y=df[metric],
                marker_color=color,
                text=[f"{v:.2f}" for v in df[metric]],
                textposition="outside",
            )
        )

    fig.update_layout(
        title=dict(text="Metric Comparison Across Test Cases", x=0.5, font=dict(size=16)),
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.15]),
        xaxis=dict(title="Test Case"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
        margin=dict(l=40, r=40, t=80, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap between evaluation metrics.

    Args:
        df: DataFrame containing metric score columns.

    Returns:
        Plotly Figure object.
    """
    metric_keys = ["faithfulness", "relevancy", "precision", "hallucination", "overall_score"]
    available = [m for m in metric_keys if m in df.columns]
    corr = df[available].corr().round(2)
    labels = [_METRIC_LABELS.get(m, m.replace("_", " ").title()) for m in corr.columns]

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale="RdYlGn",
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text}",
            textfont=dict(size=13),
            showscale=True,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=dict(text="Metric Correlation Heatmap", x=0.5, font=dict(size=16)),
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_overall_scores_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of overall scores, colour-coded by quality tier.

    Args:
        df: DataFrame with columns label and overall_score.

    Returns:
        Plotly Figure object.
    """
    colors = [
        "#2ecc71" if s >= 0.7 else ("#f39c12" if s >= 0.4 else "#e74c3c")
        for s in df["overall_score"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["overall_score"],
            y=df["label"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in df["overall_score"]],
            textposition="outside",
        )
    )

    # Threshold line at 0.7
    fig.add_vline(x=0.7, line_dash="dash", line_color="green", annotation_text="Good (0.7)")
    fig.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Poor (0.4)")

    fig.update_layout(
        title=dict(text="Overall Scores by Test Case", x=0.5, font=dict(size=16)),
        xaxis=dict(title="Overall Score", range=[0, 1.2]),
        yaxis=dict(title=""),
        height=360,
        margin=dict(l=40, r=60, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
