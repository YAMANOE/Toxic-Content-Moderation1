import streamlit as st
from PIL import Image
from lstm_model.predict import predict_text
from llama_guard import check_text_safety
from blip_caption import generate_caption

st.set_page_config(page_title="Toxic Content Moderator")

st.title("Toxic Content Moderation System")
st.write("Enter text or upload an image to check for toxic content.")

input_text = st.text_area("Enter text")
uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

final_text = None

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Generating caption for the image...")
    final_text = generate_caption(image)
    st.write("Caption:", final_text)
elif input_text.strip():
    final_text = input_text

if final_text:
    st.write("Stage 1: LLaMA Guard Safety Check")
    safety = check_text_safety(final_text)

    if safety == "unsafe":
        st.write("Content is unsafe. Stopping further processing.")
    else:
        st.write("Content is safe. Running classification...")
        predicted_class, probabilities = predict_text(final_text)
        st.write("Predicted Category:", predicted_class)
        st.write("Probabilities:")
        for label, prob in probabilities.items():
            st.write(f"{label}: {prob:.2f}")
