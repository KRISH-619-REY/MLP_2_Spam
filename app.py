import streamlit as st
import onnxruntime as ort
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
session    = ort.InferenceSession("app/spam_model.onnx")
input_name = session.get_inputs()[0].name

with open("app/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 100

st.title("🚨 SpamShield — SMS Detector")
st.write("Detect spam messages automatically")

text = st.text_area("Enter SMS message here")

if st.button("Analyse"):
    if text.strip():
        seq    = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAXLEN).astype(np.float32)
        result = session.run(None, {input_name: padded})
        prob   = float(result[0][0][0])
        label  = "SPAM 🚨" if prob >= 0.5 else "HAM ✅"
        conf   = prob if prob >= 0.5 else 1 - prob
        st.subheader(label)
        st.progress(conf)
        st.write(f"Confidence: {round(conf*100, 2)}%")
    else:
        st.warning("Please enter a message first.")