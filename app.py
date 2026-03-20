"""
app.py — Flask backend for SpamScan
-------------------------------------
Loads spam_model.onnx + tfidf_vectorizer.pkl + label_encoder.pkl
and exposes a /predict endpoint consumed by the frontend.

Run locally:
    python app.py
    → opens at http://localhost:5000

Deploy to Render / Railway:
    - push this repo to GitHub
    - set start command: gunicorn app:app

Required files in the same folder:
    spam_model.onnx         ← your exported ONNX model
    tfidf_vectorizer.pkl    ← run save_artifacts.py to generate
    label_encoder.pkl       ← run save_artifacts.py to generate
    templates/index.html    ← frontend (already provided)
"""

import os
import pickle
import string
import numpy as np
from flask import Flask, request, jsonify, render_template

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import onnxruntime as ort

# ── NLTK data ─────────────────────────────────────────────────────────────────
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

# ── Load artifacts at startup ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ONNX inference session (CPU — works on all platforms, no GPU needed)
session = ort.InferenceSession(
    os.path.join(BASE_DIR, "spam_model.onnx"),
    providers=["CPUExecutionProvider"]
)
INPUT_NAME  = session.get_inputs()[0].name    # "input"  (set during tf2onnx export)
OUTPUT_NAME = session.get_outputs()[0].name   # output tensor

with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# ── Preprocessing — exact notebook pipeline ───────────────────────────────────
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    """
    Mirrors the notebook:
    lowercase → word_tokenize → remove stopwords & punctuation → lemmatize → join
    """
    text   = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens
              if t not in string.punctuation and t not in _stop_words]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return "".join(tokens)

def classify(text: str) -> dict:
    processed = preprocess(text)

    # TF-IDF vectorize — cast to float32 (ONNX requires float32, sklearn outputs float64)
    vec = tfidf.transform([processed]).toarray().astype(np.float32)  # shape: (1, vocab_size)

    # ONNX inference
    outputs  = session.run([OUTPUT_NAME], {INPUT_NAME: vec})
    raw_prob = float(outputs[0][0][0])   # sigmoid output → probability of SPAM

    # Decode label: ham=0, spam=1
    label_idx  = int(raw_prob > 0.5)
    verdict    = le.inverse_transform([label_idx])[0].upper()
    confidence = raw_prob if verdict == "SPAM" else 1.0 - raw_prob

    # Top TF-IDF tokens for signal display
    feature_names = tfidf.get_feature_names_out()
    top_indices   = np.argsort(vec[0])[::-1][:8]
    top_tokens    = [feature_names[i] for i in top_indices if vec[0][i] > 0]

    return {
        "verdict":    verdict,
        "confidence": round(confidence * 100, 1),
        "processed":  processed,
        "signals":    top_tokens,
    }

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data    = request.get_json(force=True)
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400
    if len(message) > 500:
        return jsonify({"error": "Message too long (max 500 chars)"}), 400

    try:
        result = classify(message)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)