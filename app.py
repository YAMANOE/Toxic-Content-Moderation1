import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Use external LSTM model directory
LSTM_DIR = r"C:/Users/user/OneDrive/Desktop/Yaman khaled obiedat/nlp_pro1/lstm_model"

# Load the LSTM model and resources
@st.cache_resource
def load_lstm_resources():
    model = load_model(os.path.join(LSTM_DIR, "model.h5"))
    with open(os.path.join(LSTM_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(LSTM_DIR, "label_encoder.pickle"), "rb") as f:
        label_encoder = pickle.load(f)

    try:
        with open(os.path.join(LSTM_DIR, "max_length.json"), "r") as f:
            max_len_data = json.load(f)
        max_len = max_len_data.get("max_length", 151)
    except FileNotFoundError:
        max_len = 151

    return model, tokenizer, label_encoder, max_len

# Load BLIP image captioning model
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Simulate LLaMA Guard decision
@st.cache_data
def llama_guard_check(text):
    dangerous_keywords = ["kill", "suicide", "attack", "explosion", "child", "rape"]
    text_lower = text.lower()
    for word in dangerous_keywords:
        if word in text_lower:
            return True  # Dangerous
    return False  # Not dangerous

# Predict using LSTM
def predict_lstm(text, model, tokenizer, label_encoder, max_len):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    probs = model.predict(padded)[0]
    pred_index = int(np.argmax(probs))
    pred_class = label_encoder.inverse_transform([pred_index])[0]
    prob_dict = {label_encoder.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(probs)}
    return pred_class, prob_dict

# Get caption from image
def get_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# App layout
st.set_page_config(page_title="Toxic Content Moderator", layout="centered")
st.title("Hybrid Content Moderator ")

model, tokenizer, label_encoder, MAX_LEN = load_lstm_resources()
processor, blip_model = load_blip_model()

input_mode = st.radio("Choose input type:", ["Text", "Image"], horizontal=True)

if input_mode == "Text":
    user_input = st.text_area("Enter a sentence:")
    if st.button("Classify") and user_input.strip():
        llama_result = llama_guard_check(user_input)
        st.markdown(f"**LLaMA Decision:** {'Dangerous' if llama_result else 'Safe'}")

        if not llama_result:
            pred_class, probs = predict_lstm(user_input, model, tokenizer, label_encoder, MAX_LEN)
            st.markdown(f"**LSTM Prediction:** `{pred_class}`")
            st.markdown("**Class Probabilities:**")
            st.json(probs)
            st.success("LLaMA and LSTM decisions are aligned.")
        else:
            st.error("Content flagged as dangerous by LLaMA. Skipping LSTM classification.")

elif input_mode == "Image":
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Generating caption..."):
            caption = get_caption(image, processor, blip_model)
        st.markdown(f"**Image Caption:** {caption}")

        llama_result = llama_guard_check(caption)
        st.markdown(f"**LLaMA Decision:** {'Dangerous' if llama_result else 'Safe'}")

        if not llama_result:
            pred_class, probs = predict_lstm(caption, model, tokenizer, label_encoder, MAX_LEN)
            st.markdown(f"**LSTM Prediction:** `{pred_class}`")
            st.markdown("**Class Probabilities:**")
            st.json(probs)
            st.success("LLaMA and LSTM decisions are aligned.")
        else:
            st.error("Content flagged as dangerous by LLaMA. Skipping LSTM classification.")
