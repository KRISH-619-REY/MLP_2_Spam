"""
app.py — SpamScan Streamlit App
---------------------------------
Uses spam_model.onnx + tfidf_vectorizer.pkl + label_encoder.pkl
No TensorFlow required — only onnxruntime.

Required files in the same folder as app.py:
    spam_model.onnx         ← your exported ONNX model
    tfidf_vectorizer.pkl    ← run save_artifacts.py to generate
    label_encoder.pkl       ← run save_artifacts.py to generate

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    push all files to GitHub → connect on share.streamlit.io
"""

import streamlit as st
import numpy as np
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import onnxruntime as ort

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamScan — SMS Spam Detector",
    page_icon="🚨",
    layout="centered"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0a0a0f; color: #e8e8f0; }
.block-container { padding-top: 2rem; }

.title-block { text-align:center; padding: 2rem 0 1.5rem; }
.title-block h1 {
    font-family: 'Space Mono', monospace;
    font-size: 3rem; font-weight: 700;
    letter-spacing: -0.02em; margin-bottom: 0.4rem;
}
.subtitle {
    color: #6b6b85; font-size: 0.85rem;
    font-family: 'Space Mono', monospace; letter-spacing: 0.04em;
}
.verdict-spam {
    background: rgba(255,60,95,0.08);
    border: 1px solid rgba(255,60,95,0.4);
    border-radius: 12px; padding: 1.5rem;
    text-align: center; margin: 1rem 0;
}
.verdict-ham {
    background: rgba(0,232,162,0.08);
    border: 1px solid rgba(0,232,162,0.4);
    border-radius: 12px; padding: 1.5rem;
    text-align: center; margin: 1rem 0;
}
.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b6b85; margin-bottom: 0.3rem;
}
.verdict-word-spam {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem; font-weight: 700; color: #ff3c5f;
}
.verdict-word-ham {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem; font-weight: 700; color: #00e8a2;
}
.tag-spam {
    display: inline-block;
    background: rgba(255,60,95,0.1); border: 1px solid rgba(255,60,95,0.4);
    color: #ff3c5f; font-family: 'Space Mono', monospace;
    font-size: 0.68rem; padding: 3px 10px; border-radius: 5px; margin: 3px;
}
.tag-ham {
    display: inline-block;
    background: rgba(0,232,162,0.1); border: 1px solid rgba(0,232,162,0.4);
    color: #00e8a2; font-family: 'Space Mono', monospace;
    font-size: 0.68rem; padding: 3px 10px; border-radius: 5px; margin: 3px;
}
.mono-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #3a3a55;
    border-bottom: 1px solid #1a1a24;
    padding-bottom: 6px; margin: 1.2rem 0 0.8rem;
}
.stat-box {
    background: #111118; border: 1px solid #2a2a3a;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.stat-num { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; }
.stat-lbl {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: #3a3a55; margin-top: 2px;
}
.hist-item {
    background: #111118; border: 1px solid #2a2a3a;
    border-radius: 8px; padding: 10px 14px;
    margin-bottom: 6px; display: flex;
    align-items: center; gap: 12px; font-size: 0.82rem;
}
.hist-badge-spam {
    font-family: 'Space Mono', monospace; font-size: 0.62rem; font-weight: 700;
    background: rgba(255,60,95,0.15); color: #ff3c5f;
    padding: 2px 8px; border-radius: 4px; white-space: nowrap;
}
.hist-badge-ham {
    font-family: 'Space Mono', monospace; font-size: 0.62rem; font-weight: 700;
    background: rgba(0,232,162,0.15); color: #00e8a2;
    padding: 2px 8px; border-radius: 4px; white-space: nowrap;
}
.hist-msg { color: #6b6b85; flex: 1; }
.hist-conf { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #3a3a55; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── NLTK DOWNLOADS (cached) ───────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)

download_nltk()

# ── LOAD ARTIFACTS (cached — loads only once per session) ─────────────────────
@st.cache_resource
def load_artifacts():
    # ONNX session — no TensorFlow needed
    session = ort.InferenceSession(
        "spam_model.onnx",
        providers=["CPUExecutionProvider"]
    )
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return session, tfidf, le

MODEL_LOADED = False  
LOAD_ERROR = ""  

try:
    session, tfidf, le = load_artifacts()
    INPUT_NAME  = session.get_inputs()[0].name
    OUTPUT_NAME = session.get_outputs()[0].name
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR   = str(e)

# ── PREPROCESSING — exact notebook pipeline ───────────────────────────────────
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text   = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens
              if t not in string.punctuation and t not in _stop_words]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def classify(text: str):
    processed = preprocess(text)

    # float32 — ONNX requires float32, sklearn TF-IDF outputs float64
    vec = tfidf.transform([processed]).toarray().astype(np.float32)

    # ONNX inference
    outputs  = session.run([OUTPUT_NAME], {INPUT_NAME: vec})
    raw_prob = float(outputs[0][0][0])   # sigmoid → P(spam)

    label_idx  = int(raw_prob > 0.5)
    verdict    = le.inverse_transform([label_idx])[0].upper()
    confidence = raw_prob if verdict == "SPAM" else 1.0 - raw_prob

    # Top TF-IDF signal tokens
    feature_names = tfidf.get_feature_names_out()
    top_indices   = np.argsort(vec[0])[::-1][:8]
    top_tokens    = [feature_names[i] for i in top_indices if vec[0][i] > 0]

    return verdict, confidence, processed, top_tokens

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "spam": 0, "ham": 0}

# ── EXAMPLE MESSAGES ──────────────────────────────────────────────────────────
EXAMPLES = {
    "⚠ Spam 1": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "⚠ Spam 2": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question std txt rate T&Cs apply 08452810075 over18s",
    "✓ Ham 1":  "Hey, are you coming to the party tonight? Let me know if you need a ride!",
    "✓ Ham 2":  "Ok I'll be there by 7. Don't forget to bring the stuff for dinner we talked about yesterday.",
}

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1><span style="color:#ff3c5f">SPAM</span><span style="color:#e8e8f0">SCAN</span></h1>
  <div class="subtitle">ONNX · TF-IDF · SMSSpamCollection · No TensorFlow</div>
</div>
""", unsafe_allow_html=True)

# ── MISSING ARTIFACTS GUIDE ───────────────────────────────────────────────────
if not MODEL_LOADED:
    st.error(f"❌ Could not load model artifacts: `{LOAD_ERROR}`")
    st.markdown("""
### Setup — run this once before launching

```bash
python save_artifacts.py
```

Then make sure these files are all in the **same folder** as `app.py`:

| File | Source |
|---|---|
| `spam_model.onnx` | Already exported by your notebook |
| `tfidf_vectorizer.pkl` | Generated by `save_artifacts.py` |
| `label_encoder.pkl` | Generated by `save_artifacts.py` |
    """)
    st.stop()

# ── INPUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="mono-label">Input Message</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
example_clicked = None
with c1:
    if st.button("⚠ Spam 1", use_container_width=True): example_clicked = "⚠ Spam 1"
with c2:
    if st.button("⚠ Spam 2", use_container_width=True): example_clicked = "⚠ Spam 2"
with c3:
    if st.button("✓ Ham 1",  use_container_width=True): example_clicked = "✓ Ham 1"
with c4:
    if st.button("✓ Ham 2",  use_container_width=True): example_clicked = "✓ Ham 2"

default_text = EXAMPLES[example_clicked] if example_clicked else ""
user_input = st.text_area(
    label="SMS Message",
    value=default_text,
    placeholder="Paste or type your SMS message here...",
    height=130,
    max_chars=500,
    label_visibility="collapsed"
)
st.caption(f"{len(user_input)} / 500 characters {'⚠️' if len(user_input) > 450 else ''}")

analyze_clicked = st.button("▶  ANALYZE MESSAGE", type="primary", use_container_width=True)

# ── INFERENCE ─────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not user_input.strip():
        st.warning("Please enter a message to analyze.")
    else:
        with st.spinner("Running ONNX inference..."):
            try:
                verdict, confidence, processed, top_tokens = classify(user_input.strip())
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        is_spam = (verdict == "SPAM")
        pct     = int(confidence * 100)

        # Verdict card
        if is_spam:
            st.markdown(f"""
            <div class="verdict-spam">
              <div class="verdict-label">Model Classification</div>
              <div style="font-size:2.2rem;margin:0.2rem 0">🚨</div>
              <div class="verdict-word-spam">SPAM</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-ham">
              <div class="verdict-label">Model Classification</div>
              <div style="font-size:2.2rem;margin:0.2rem 0">✅</div>
              <div class="verdict-word-ham">HAM</div>
            </div>""", unsafe_allow_html=True)

        # Confidence bar
        st.markdown('<div class="mono-label">Confidence Score</div>', unsafe_allow_html=True)
        bar_color = "#ff3c5f" if is_spam else "#00e8a2"
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
                    font-family:'Space Mono',monospace;font-size:0.75rem;
                    color:#6b6b85;margin-bottom:6px;">
            <span>SIGMOID OUTPUT → CONFIDENCE</span>
            <span style="color:{bar_color};font-weight:700;">{pct}%</span>
        </div>""", unsafe_allow_html=True)
        st.progress(confidence)

        # Top TF-IDF signal tokens
        if top_tokens:
            st.markdown('<div class="mono-label">Top TF-IDF Signals</div>', unsafe_allow_html=True)
            tag_cls = "tag-spam" if is_spam else "tag-ham"
            chips   = "".join(f'<span class="{tag_cls}">{t}</span>' for t in top_tokens)
            st.markdown(f'<div style="padding:4px 0">{chips}</div>', unsafe_allow_html=True)

        # Preprocessed text (educational)
        with st.expander("🔬 View preprocessed text fed to TF-IDF"):
            st.markdown('<div class="mono-label">After: lowercase → tokenize → stopword removal → lemmatize → join</div>',
                        unsafe_allow_html=True)
            st.code(processed if processed else "(empty after preprocessing)", language=None)

        # Update stats & history
        st.session_state.stats["total"] += 1
        st.session_state.stats["spam" if is_spam else "ham"] += 1
        st.session_state.history.insert(0, {"msg": user_input.strip(), "verdict": verdict, "pct": pct})
        if len(st.session_state.history) > 6:
            st.session_state.history.pop()

# ── SESSION STATS ─────────────────────────────────────────────────────────────
if st.session_state.stats["total"] > 0:
    st.markdown('<div class="mono-label">Session Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, key, color, label in [
        (c1, "total", "#4f8aff", "Scanned"),
        (c2, "spam",  "#ff3c5f", "Spam"),
        (c3, "ham",   "#00e8a2", "Ham"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num" style="color:{color}">{st.session_state.stats[key]}</div>
                <div class="stat-lbl">{label}</div>
            </div>""", unsafe_allow_html=True)

# ── SCAN HISTORY ──────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="mono-label">Recent Scans</div>', unsafe_allow_html=True)
    for item in st.session_state.history:
        badge   = "hist-badge-spam" if item["verdict"] == "SPAM" else "hist-badge-ham"
        preview = item["msg"][:72] + ("..." if len(item["msg"]) > 72 else "")
        st.markdown(f"""
        <div class="hist-item">
            <span class="{badge}">{item['verdict']}</span>
            <span class="hist-msg">{preview}</span>
            <span class="hist-conf">{item['pct']}%</span>
        </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.65rem;
            color:#3a3a55;margin-top:2.5rem;padding-top:1rem;border-top:1px solid #1a1a24;">
    ONNX Runtime · MLP 128→64→32→1 (sigmoid) · TF-IDF · SMSSpamCollection
</div>
""", unsafe_allow_html=True)