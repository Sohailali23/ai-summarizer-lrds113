"""
AI Writer with Text Summarization
----------------------------------
Objective : Generate summaries from large articles using NLP.
Tools     : Python, Hugging Face Transformers, Flask
Model     : facebook/bart-large-cnn (default) or t5-small/t5-base

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import re
import json
import uuid
import datetime

from flask import Flask, render_template, request, jsonify, send_file

# ---------------------------------------------------------------------------
# 1. Model loading (BART or T5 summarization model)
# ---------------------------------------------------------------------------
# We lazy-load the model on first request so the server starts instantly and
# so the app still boots even if transformers/torch aren't installed yet
# (useful while you're wiring up the rest of the app).

MODEL_NAME = os.environ.get("SUMMARY_MODEL", "facebook/bart-large-cnn")
# Alternatives you can set via env var SUMMARY_MODEL:
#   "facebook/bart-large-cnn"  -> best quality, larger download
#   "t5-small"                 -> fastest, lower quality
#   "t5-base"                  -> good middle ground

_summarizer = None
_load_error = None


def get_summarizer():
    """Lazily load and cache the Hugging Face summarization pipeline."""
    global _summarizer, _load_error
    if _summarizer is not None:
        return _summarizer
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        from transformers import pipeline
        # device=-1 forces CPU; set to 0 if you have a CUDA GPU available.
        _summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)
        return _summarizer
    except Exception as exc:  # pragma: no cover
        _load_error = (
            f"Could not load summarization model '{MODEL_NAME}'. "
            f"Make sure 'transformers' and 'torch' are installed "
            f"(pip install -r requirements.txt). Original error: {exc}"
        )
        raise RuntimeError(_load_error)


# ---------------------------------------------------------------------------
# 2. Long-article handling: split into model-friendly chunks
# ---------------------------------------------------------------------------
# BART/T5 have a max input token length (~1024 for BART). We split on
# sentence boundaries and pack sentences into ~800-word chunks so we stay
# safely under the token limit, then summarize each chunk and optionally
# do a second "summary of summaries" pass for very long articles.

WORDS_PER_CHUNK = 700


def split_into_chunks(text, words_per_chunk=WORDS_PER_CHUNK):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count > words_per_chunk and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# 3. Run the model and return a summary (with long-article handling)
# ---------------------------------------------------------------------------

def summarize_text(text, max_length=150, min_length=40):
    summarizer = get_summarizer()
    chunks = split_into_chunks(text)

    if len(chunks) == 1:
        result = summarizer(
            chunks[0], max_length=max_length, min_length=min_length, do_sample=False
        )
        return result[0]["summary_text"].strip()

    # Multi-chunk article: summarize each piece, then summarize the
    # combined partial summaries for a coherent final result.
    partial_summaries = []
    for chunk in chunks:
        result = summarizer(
            chunk, max_length=120, min_length=30, do_sample=False
        )
        partial_summaries.append(result[0]["summary_text"].strip())

    combined = " ".join(partial_summaries)
    # Guard against the combined text itself being too long for one pass.
    if len(combined.split()) > WORDS_PER_CHUNK:
        combined = " ".join(split_into_chunks(combined, WORDS_PER_CHUNK)[:1])

    final = summarizer(
        combined, max_length=max_length, min_length=min_length, do_sample=False
    )
    return final[0]["summary_text"].strip()


# ---------------------------------------------------------------------------
# 4. Word count and a simple readability score (Flesch Reading Ease)
# ---------------------------------------------------------------------------

def count_words(text):
    return len(re.findall(r"\b\w+\b", text))


def count_sentences(text):
    sentences = re.split(r"[.!?]+", text.strip())
    return max(1, len([s for s in sentences if s.strip()]))


def count_syllables(word):
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    vowels = "aeiouy"
    syllables = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            syllables += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def flesch_reading_ease(text):
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    if word_count == 0:
        return 0.0
    sentence_count = count_sentences(text)
    syllable_count = sum(count_syllables(w) for w in words)

    score = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllable_count / word_count)
    )
    return round(max(0.0, min(100.0, score)), 1)


def readability_label(score):
    if score >= 90:
        return "Very Easy"
    if score >= 70:
        return "Easy"
    if score >= 60:
        return "Standard"
    if score >= 50:
        return "Fairly Difficult"
    if score >= 30:
        return "Difficult"
    return "Very Confusing"


# ---------------------------------------------------------------------------
# 5. Optional: save history of summaries (simple JSON file store)
# ---------------------------------------------------------------------------

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_history_entry(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]  # keep the most recent 50
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# 6. Flask app + routes
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", model_name=MODEL_NAME)


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True) if request.is_json else request.form
    text = (data.get("text") or "").strip()
    save_to_history = str(data.get("save_history", "false")).lower() == "true"

    if not text:
        return jsonify({"error": "Please provide some article text to summarize."}), 400

    if count_words(text) < 30:
        return jsonify({"error": "Please provide a longer article (at least ~30 words)."}), 400

    try:
        max_length = int(data.get("max_length", 150))
        min_length = int(data.get("min_length", 40))
    except (TypeError, ValueError):
        max_length, min_length = 150, 40

    try:
        summary = summarize_text(text, max_length=max_length, min_length=min_length)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    original_words = count_words(text)
    summary_words = count_words(summary)
    compression = round(100 * (1 - summary_words / max(1, original_words)), 1)
    score = flesch_reading_ease(summary)

    response = {
        "summary": summary,
        "original_word_count": original_words,
        "summary_word_count": summary_words,
        "compression_percent": compression,
        "readability_score": score,
        "readability_label": readability_label(score),
        "model": MODEL_NAME,
    }

    if save_to_history:
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "original_preview": text[:200] + ("..." if len(text) > 200 else ""),
            **response,
        }
        save_history_entry(entry)

    return jsonify(response)


@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())


@app.route("/history/clear", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return jsonify({"status": "cleared"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


# ---------------------------------------------------------------------------
# 7. Deploy using Flask
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. For production, use gunicorn, e.g.:
    #   gunicorn -w 2 -b 0.0.0.0:5000 app:app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
