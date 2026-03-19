from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ── Paths relative to app.py location ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load ONNX model ──────────────────────────────────────────────
session    = ort.InferenceSession(os.path.join(BASE_DIR, "spam_model.onnx"))
input_name = session.get_inputs()[0].name

# ── Load tokenizer ──────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
    tok = pickle.load(f)

print("Loaded successfully ✅", type(tok))

# ── Prediction helper ────────────────────────────────────────────
def predict(text: str) -> dict:
    seq    = tok.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN).astype(np.float32)
    result = session.run(None, {input_name: padded})
    prob   = float(result[0][0][0])       # spam probability (0→ham, 1→spam)
    label  = "SPAM" if prob >= 0.5 else "HAM"
    confidence = prob if prob >= 0.5 else 1 - prob
    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "raw":        round(prob, 4)
    }


# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = predict(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
