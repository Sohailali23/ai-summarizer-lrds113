# AI Writer with Text Summarization

Generate summaries from large articles using NLP (Hugging Face BART/T5 + Flask).

## What's inside

```
ai-summarizer/
├── app.py              # Flask backend: model, chunking, scoring, routes
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Text input / results UI
├── static/
│   ├── style.css
│   └── app.js
└── history.json         # created automatically once you save a summary
```

## 1. Use BART or T5 summarization model
Default model is `facebook/bart-large-cnn` (best quality). You can switch models
without touching code:

```bash
export SUMMARY_MODEL=t5-small     # fastest, good for quick testing
export SUMMARY_MODEL=t5-base      # balance of speed/quality
export SUMMARY_MODEL=facebook/bart-large-cnn   # default, best quality
```

## 2–4. Text input interface, long-article handling, run model

Install dependencies (first run downloads the model, ~1.6GB for BART):

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**, paste an article, and click **Summarize**.

Long articles are automatically split into ~700-word chunks on sentence
boundaries, each chunk is summarized, and the partial summaries are combined
and summarized once more for a coherent final result (see
`split_into_chunks` / `summarize_text` in `app.py`).

## 5. Word count and readability score

Every response includes:
- Original word count / summary word count
- Compression percentage
- A Flesch Reading Ease score (0–100) plus a plain-language label
  (Very Easy → Very Confusing)

## 6. Optional: history of summaries

Check **"Save to history"** before summarizing to persist the result to
`history.json`. View past summaries with the **View History** button, or
via `GET /history`. Clear it with `POST /history/clear`.

## 7. Deploy using Flask

**Local/dev:**
```bash
python app.py
```

**Production (gunicorn):**
```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

**Docker (optional):**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

**Cloud options:** Render, Railway, Fly.io, or a GPU-enabled instance (AWS
EC2 / GCP) if you want faster inference on large volumes of text.

## API reference

| Route | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/summarize` | POST | `{text, max_length, min_length, save_history}` → summary + stats |
| `/history` | GET | List saved summaries |
| `/history/clear` | POST | Clear saved history |
| `/health` | GET | Health check |

## Notes

- First request after startup will be slow while the model loads into
  memory — this is expected.
- No GPU required, but a GPU (`device=0` in `app.py`) will speed things up
  significantly for long articles or high traffic.
