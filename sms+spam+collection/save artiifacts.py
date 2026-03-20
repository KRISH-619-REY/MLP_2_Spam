"""
save_artifacts.py
------------------
Run this ONCE inside your tf_env before launching the Flask app.
It re-runs the exact preprocessing pipeline from your notebook,
fits a TF-IDF vectorizer and LabelEncoder on the full dataset,
and saves them as pickle files so app.py can load them.

Usage (inside tf_env):
    python save_artifacts.py

Output:
    tfidf_vectorizer.pkl   — fitted TfidfVectorizer
    label_encoder.pkl      — fitted LabelEncoder (ham=0, spam=1)
"""

import pandas as pd
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ── NLTK downloads ────────────────────────────────────────────────────────────
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv(
    "sms+spam+collection/SMSSpamCollection",
    sep="\t", header=None, names=["label", "message"]
)

# ── Preprocessing — EXACT same pipeline as notebook ──────────────────────────
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text   = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens
              if t not in string.punctuation and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return "".join(tokens)

df["message"] = df["message"].fillna("").apply(preprocess)

# ── Fit TF-IDF ────────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer()
tfidf.fit(df["message"])

# ── Fit LabelEncoder ──────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(df["label"])   # ham → 0, spam → 1

# ── Save ──────────────────────────────────────────────────────────────────────
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅  tfidf_vectorizer.pkl saved")
print("✅  label_encoder.pkl saved")
print(f"    Vocabulary size : {len(tfidf.vocabulary_)}")
print(f"    Label classes   : {list(le.classes_)}")