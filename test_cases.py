"""Pre-loaded test scenarios covering common LLM failure modes."""

TEST_CASES: list[dict] = [
    {
        "id": "perfect",
        "label": "✅ Perfect Answer",
        "question": "What is the capital of France?",
        "context": (
            "Paris is the capital and largest city of France. "
            "It has been a major European center since the 12th century."
        ),
        "llm_answer": "The capital of France is Paris.",
        "ground_truth": "Paris",
        "expected_scores": {
            "faithfulness": 0.9,
            "relevancy": 0.95,
            "precision": 0.8,
            "hallucination": 0.95,
        },
        "failure_mode": "None — ideal response grounded in context.",
    },
    {
        "id": "hallucinated",
        "label": "🌀 Hallucinated Facts",
        "question": "What is the population of Tokyo?",
        "context": "Tokyo is the capital of Japan.",
        "llm_answer": (
            "Tokyo has a population of 14 million people "
            "and is the largest city in the world by area."
        ),
        "ground_truth": "Approximately 14 million (city); 37 million (metro)",
        "expected_scores": {
            "faithfulness": 0.2,
            "relevancy": 0.7,
            "precision": 0.3,
            "hallucination": 0.2,
        },
        "failure_mode": "Specific numbers and claims not present in context.",
    },
    {
        "id": "off_topic",
        "label": "🎯 Off-Topic Answer",
        "question": "Explain photosynthesis",
        "context": (
            "Photosynthesis is the process by which plants convert light energy "
            "into chemical energy stored in glucose. Chlorophyll absorbs sunlight "
            "and uses it to transform water and carbon dioxide into glucose and oxygen."
        ),
        "llm_answer": "The weather today is sunny and warm. I enjoy outdoor activities.",
        "ground_truth": (
            "Photosynthesis converts light energy to chemical energy in plants "
            "using chlorophyll, water, and CO2."
        ),
        "expected_scores": {
            "faithfulness": 0.1,
            "relevancy": 0.05,
            "precision": 0.05,
            "hallucination": 0.1,
        },
        "failure_mode": "Answer completely ignores the question and context.",
    },
    {
        "id": "irrelevant_context",
        "label": "📚 Irrelevant Context",
        "question": "What causes rain?",
        "context": (
            "The Amazon rainforest is the largest tropical rainforest on Earth, "
            "covering over 5.5 million square kilometres. It is home to an "
            "estimated 10% of all species on Earth."
        ),
        "llm_answer": (
            "Rain is caused by water vapor in the atmosphere condensing "
            "into droplets that form clouds, which eventually fall as precipitation."
        ),
        "ground_truth": "Rain forms when water vapor condenses and falls as precipitation.",
        "expected_scores": {
            "faithfulness": 0.15,
            "relevancy": 0.85,
            "precision": 0.1,
            "hallucination": 0.2,
        },
        "failure_mode": "Context is unrelated; answer is correct but ungrounded.",
    },
    {
        "id": "mixed_quality",
        "label": "⚠️ Mixed Quality",
        "question": "Who invented the telephone?",
        "context": (
            "Alexander Graham Bell is credited with inventing the telephone in 1876. "
            "He received the first patent for the telephone on March 7, 1876."
        ),
        "llm_answer": (
            "The telephone was invented by Alexander Graham Bell in 1876. "
            "Bell was also a famous musician who performed at Carnegie Hall "
            "and wrote several bestselling books on acoustics."
        ),
        "ground_truth": "Alexander Graham Bell invented the telephone in 1876.",
        "expected_scores": {
            "faithfulness": 0.5,
            "relevancy": 0.8,
            "precision": 0.6,
            "hallucination": 0.45,
        },
        "failure_mode": "Correct core fact, but hallucinated additional biographical details.",
    },
]


def get_test_case(case_id: str) -> dict | None:
    """Return a single test case by its ID."""
    return next((c for c in TEST_CASES if c["id"] == case_id), None)


def get_test_case_labels() -> list[str]:
    """Return display labels for the UI dropdown."""
    return [c["label"] for c in TEST_CASES]
