import numpy as np
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("lstm_model/model.h5")

with open("lstm_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("lstm_model/label_map.json", "r") as f:
    label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

MAX_LEN = 100

def predict_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    probabilities = model.predict(padded)[0]
    predicted_index = np.argmax(probabilities)
    predicted_class = label_map[predicted_index]
    return predicted_class, dict(zip(label_map.values(), map(float, probabilities)))
