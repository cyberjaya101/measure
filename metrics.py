"""LLM evaluation metrics: faithfulness, relevancy, context precision, hallucination."""

import re
from typing import Any


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, stripping punctuation."""
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


def _stopwords() -> set[str]:
    return {
        "a", "an", "the", "is", "it", "in", "of", "and", "or", "to", "for",
        "that", "this", "with", "was", "are", "be", "by", "as", "at", "on",
        "has", "have", "had", "from", "not", "but", "what", "which", "who",
        "how", "when", "where", "why", "does", "do", "did", "been", "its",
        "they", "them", "their", "we", "you", "i", "he", "she", "also", "so",
    }


def _content_words(text: str) -> set[str]:
    return _tokenize(text) - _stopwords()


def calculate_faithfulness(context: str, answer: str) -> dict[str, Any]:
    """Measure how well the answer is grounded in the provided context.

    A high score means most answer claims can be traced back to the context.
    Formula: answer words found in context / total answer words (content only).
    """
    answer_words = _content_words(answer)
    context_words = _tokenize(context)

    if not answer_words:
        return {"score": 0.0, "explanation": "Answer is empty.", "details": {}}

    supported = answer_words & context_words
    score = round(len(supported) / len(answer_words), 3)

    return {
        "score": score,
        "explanation": (
            f"{len(supported)} of {len(answer_words)} content words in answer "
            f"are found in context."
        ),
        "details": {
            "answer_content_words": sorted(answer_words),
            "supported_words": sorted(supported),
            "unsupported_words": sorted(answer_words - context_words),
        },
    }


def calculate_relevancy(question: str, answer: str) -> dict[str, Any]:
    """Measure how well the answer addresses the question.

    Compares content-word overlap between question and answer.
    Off-topic answers are heavily penalised.
    """
    q_words = _content_words(question)
    a_words = _content_words(answer)

    if not q_words:
        return {"score": 0.0, "explanation": "Question is empty.", "details": {}}

    if not a_words:
        return {"score": 0.0, "explanation": "Answer is empty.", "details": {}}

    overlap = q_words & a_words
    # Jaccard-like but question-centric
    score = round(len(overlap) / len(q_words), 3)
    # Clamp to [0, 1] and boost slightly when answer is non-trivially long
    length_bonus = min(0.1, len(a_words) / 100)
    score = min(1.0, round(score + length_bonus, 3))

    return {
        "score": score,
        "explanation": (
            f"{len(overlap)} of {len(q_words)} question keywords addressed in answer."
        ),
        "details": {
            "question_keywords": sorted(q_words),
            "answer_keywords": sorted(a_words),
            "matched_keywords": sorted(overlap),
        },
    }


def calculate_precision(context: str, answer: str) -> dict[str, Any]:
    """Measure how much of the retrieved context is actually used in the answer.

    High precision = context is tightly relevant; low = context is padded / off-topic.
    Formula: context words that appear in answer / total context content words.
    """
    context_words = _content_words(context)
    answer_words = _tokenize(answer)

    if not context_words:
        return {"score": 0.0, "explanation": "Context is empty.", "details": {}}

    used = context_words & answer_words
    score = round(len(used) / len(context_words), 3)

    return {
        "score": score,
        "explanation": (
            f"{len(used)} of {len(context_words)} context content words "
            f"referenced in answer."
        ),
        "details": {
            "context_content_words": sorted(context_words),
            "used_in_answer": sorted(used),
            "unused_in_answer": sorted(context_words - answer_words),
        },
    }


def calculate_hallucination(context: str, answer: str) -> dict[str, Any]:
    """Detect answer claims not supported by the context.

    Score of 1.0 = no hallucination; 0.0 = fully hallucinated.
    Formula: 1 - (unique answer words absent from context / total unique answer words).
    """
    answer_words = _content_words(answer)
    context_words = _tokenize(context)

    if not answer_words:
        return {"score": 1.0, "explanation": "Answer is empty — nothing to hallucinate.", "details": {}}

    unsupported = answer_words - context_words
    hallucination_ratio = len(unsupported) / len(answer_words)
    score = round(1.0 - hallucination_ratio, 3)

    level = "low" if score >= 0.7 else ("moderate" if score >= 0.4 else "high")

    return {
        "score": score,
        "explanation": (
            f"{len(unsupported)} of {len(answer_words)} answer content words "
            f"not found in context — {level} hallucination risk."
        ),
        "details": {
            "hallucinated_words": sorted(unsupported),
            "grounded_words": sorted(answer_words & context_words),
            "hallucination_ratio": round(hallucination_ratio, 3),
        },
    }
