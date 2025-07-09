import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained LSTM model
model = load_model(r"C:\Users\user\OneDrive\Desktop\Yaman khaled obiedat\nlp_pro1\lstm_model\model.h5")


# Load the tokenizer
with open(r"C:\Users\user\OneDrive\Desktop\Yaman khaled obiedat\nlp_pro1\lstm_model\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open(r"C:\Users\user\OneDrive\Desktop\Yaman khaled obiedat\nlp_pro1\lstm_model\label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# Try to load max_length from JSON, else use default 100
try:
    with open("lstm_model/max_length.json", "r") as f:
        max_len_data = json.load(f)
    MAX_LEN = max_len_data.get("max_length", 100)
except FileNotFoundError:
    MAX_LEN = 151

def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    probs = model.predict(padded)[0]
    pred_index = int(np.argmax(probs))
    pred_class = label_encoder.inverse_transform([pred_index])[0]
    prob_dict = {label_encoder.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(probs)}
    return pred_class, prob_dict

if __name__ == "__main__":
    sample = input("Enter a text to classify: ")
    category, probabilities = predict_text(sample)
    print(f"\nPredicted Category: {category}\n")
    print("Probabilities:")
    for cls, prob in probabilities.items():
        print(f"{cls}: {prob:.4f}")
