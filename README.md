# NLP Graphic Tool — Research Assignment Option 2

A modern Python application implementing a **graphic tool** for document processing with full support for **Tigrinya (ትግርኛ)** and **English** text.

## Professor Requirements — All Implemented ✅

| # | Requirement | Implementation |
|---|-------------|----------------|
| **2.1** | Eliminate stopwords | `tigrinya-nlp` (Tigrinya) / NLTK (English) |
| **2.2** | Lemmatize terms | WordNet (English) / Identity (Tigrinya*) |
| **2.3** | Compute frequencies | `Counter` on tokenized/lemmatized terms |
| **2.4** | Measure distances from strategic points (start and end) | `first_pos/total`, `(total-last_pos)/total` per term |
| **2.5** | Compute compound relevance indices (50% frequency + 50% earliness) | `0.5 * freq_score + 0.5 * earliness_score` |

*Tigrinya lemmatization: No standard tool exists; identity mapping used. Can be extended with rule-based stemmer.

## Project Structure

```
nlp_graphic_tool/
├── app.py              # Streamlit graphic UI
├── nlp_pipeline.py     # Core NLP logic (all 5 requirements)
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Installation

```bash
cd nlp_graphic_tool
pip install -r requirements.txt
```

**Note:** For Tigrinya support, `tigrinya-nlp` is required. For English, `nltk` is required. Both are in `requirements.txt`.

## Run the Application

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Features

- **Dual language:** Tigrinya and English
- **Input modes:** Text area or file upload (.txt, .md)
- **Interactive tabs:** Frequencies, Distances, Compound Relevance, Visualizations
- **Charts:** Bar charts, scatter plots, pie charts (Plotly)
- **Export:** Download results as CSV
- **Modern UI:** Dark theme, responsive layout

## Technical Details

### Distance from Strategic Points (2.4)

For each unique term:
- **Distance from start:** `first_occurrence_position / total_tokens`
- **Distance from end:** `(total_tokens - last_occurrence_position) / total_tokens`

### Compound Relevance (2.5)

- **Frequency score:** `term_freq / max_freq` (normalized 0–1)
- **Earliness score:** `1 - (avg_position / total_tokens)` — earlier terms score higher
- **Compound relevance:** `0.5 × freq_score + 0.5 × earliness_score`

## Deploy on Render (Free)

1. Push this project to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Web Service**.
3. Connect your repo and select `nlp_graphic_tool` as the root (or set it in settings).
4. Render will use `render.yaml` if present, or set manually:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Deploy. Free tier: 750 hrs/month; service sleeps after ~15 min inactivity.

## Dependencies

- `streamlit` — Web UI
- `pandas` — Data tables
- `plotly` — Interactive charts
- `tigrinya-nlp` — Tigrinya preprocessing (stopwords, tokenization)
- `nltk` — English tokenization, stopwords, lemmatization

## License

MIT — For academic submission.
