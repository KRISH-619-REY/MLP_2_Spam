import pickle

with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

with open("word_index.pkl", "wb") as f:
    pickle.dump(tok.word_index, f)

print("✅ word_index.pkl created successfully")