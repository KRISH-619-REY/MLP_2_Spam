import streamlit as st
import onnxruntime as ort
import numpy as np
import pickle

# Replace tensorflow's pad_sequences with this manual version
def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq[:maxlen]
        padded[i, -len(seq):] = seq
    return padded

# Load model and tokenizer
session    = ort.InferenceSession("spam_model.onnx")
input_name = session.get_inputs()[0].name

with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

MAXLEN = 100

st.title("🚨 SpamShield — SMS Detector")
st.write("Detect spam messages automatically")

text = st.text_area("Enter SMS message here")

if st.button("Analyse"):
    if text.strip():
        seq    = [word_index.get(word, 0) for word in text.split()]
        padded = pad_sequences([seq], maxlen=MAXLEN)
        result = session.run(None, {input_name: padded})
        prob   = float(result[0][0][0])
        label  = "SPAM 🚨" if prob >= 0.5 else "HAM ✅"
        conf   = prob if prob >= 0.5 else 1 - prob
        st.subheader(label)
        st.progress(conf)
        st.write(f"Confidence: {round(conf*100, 2)}%")
    else:
        st.warning("Please enter a message first.")