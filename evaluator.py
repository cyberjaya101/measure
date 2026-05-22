"""Core evaluation logic: single response and batch evaluation."""

import pandas as pd

from metrics import (
    calculate_faithfulness,
    calculate_hallucination,
    calculate_precision,
    calculate_relevancy,
)


def evaluate_response(question: str, context: str, answer: str) -> dict:
    """Run all 4 metrics against a single LLM response.

    Args:
        question: The user's original question.
        context: Retrieved context provided to the LLM.
        answer: The LLM-generated answer to evaluate.

    Returns:
        Dict with per-metric results plus an overall_score (mean of all four).
    """
    faithfulness = calculate_faithfulness(context, answer)
    relevancy = calculate_relevancy(question, answer)
    precision = calculate_precision(context, answer)
    hallucination = calculate_hallucination(context, answer)

    scores = [
        faithfulness["score"],
        relevancy["score"],
        precision["score"],
        hallucination["score"],
    ]
    overall = round(sum(scores) / len(scores), 3)

    return {
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "precision": precision,
        "hallucination": hallucination,
        "overall_score": overall,
    }


def batch_evaluate(test_cases: list[dict]) -> pd.DataFrame:
    """Evaluate a list of test cases and return a summary DataFrame.

    Args:
        test_cases: List of dicts with keys: id, question, context, llm_answer.

    Returns:
        DataFrame with columns: id, label, overall_score, faithfulness,
        relevancy, precision, hallucination, failure_mode.
    """
    rows = []
    for case in test_cases:
        result = evaluate_response(
            question=case["question"],
            context=case["context"],
            answer=case["llm_answer"],
        )
        rows.append(
            {
                "id": case["id"],
                "label": case.get("label", case["id"]),
                "overall_score": result["overall_score"],
                "faithfulness": result["faithfulness"]["score"],
                "relevancy": result["relevancy"]["score"],
                "precision": result["precision"]["score"],
                "hallucination": result["hallucination"]["score"],
                "failure_mode": case.get("failure_mode", "—"),
            }
        )
    return pd.DataFrame(rows)
