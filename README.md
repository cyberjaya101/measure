# 📊 measure

> Production LLM evaluation with 4 core metrics. Measure faithfulness, relevancy, precision, hallucination.

## Why This Exists

"My LLM works great" isn't measurable. Production systems need metrics.

## Features

- **4 Production Metrics** - Faithfulness, relevancy, precision, hallucination
- **Batch Evaluation** - Test multiple scenarios
- **Visual Dashboard** - Plotly charts and score cards
- **Keyword-Based** - No API calls, runs locally

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Metrics

| Metric | Measures | Good Score |
|--------|----------|------------|
| Faithfulness | Grounding in context | > 0.7 |
| Relevancy | Matches question | > 0.8 |
| Precision | Focused answer | > 0.5 |
| Hallucination | No made-up facts | > 0.9 |

## Tech Stack

Streamlit • Plotly • Pandas • Python

## License

MIT
