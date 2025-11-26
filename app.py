import streamlit as st
import joblib
import numpy as np
from PIL import Image
import pandas as pd
import json

# Load model and label encoder
rf_model = joblib.load('models/rf_model_save.pkl')
label_encoder = joblib.load('models/label_encoder_model_save.pkl')

# Load usage data
with open('datasets/emoji_usage_complete.json', 'r', encoding='utf-8') as f:
    emoji_usage = json.load(f)

# Load emoji mapping (match on 'name' not 'label')
emoji_df = pd.read_csv('datasets/emoji_datas.csv')  # must include: 'name' and 'emoji'
emoji_map = dict(zip(emoji_df['name'].str.lower(), emoji_df['emoji']))

# Image feature extractor (without TensorFlow)
def extract_features(image, target_size=(64, 64)):
    image = image.convert('RGB').resize(target_size)
    img_array = np.array(image) / 255.0  # Normalize pixel values
    return img_array.flatten().reshape(1, -1)

# Streamlit UI
st.set_page_config(page_title="Emoji Classifier", layout="wide")
st.title("🤖 Emoji Classifier")

uploaded_file = st.file_uploader("Upload your emoji image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Center crop and resize
    w, h = image.size
    min_dim = min(w, h)
    image = image.crop(((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2))
    image = image.resize((128, 128))
    st.image(image, caption="🖼️ Uploaded Emoji", width=150)

    if st.button("🔍 Predict"):
        features = extract_features(image)
        probs = rf_model.predict_proba(features)[0]
        top_indices = np.argsort(probs)[-3:][::-1]
        top_labels = label_encoder.inverse_transform(top_indices)
        top_scores = probs[top_indices]

        # Show usage of top-1 prediction
        top_name = top_labels[0]
        usage = emoji_usage.get(top_name, "Usage not found.")
        st.subheader("💡 Emoji Usage")
        st.success(f"{emoji_map.get(top_name.lower(), '')} **{top_name}**: {usage}")

        st.subheader("🔝 Top-3 Predictions")

        for i in range(3):
            name = top_labels[i]
            prob = top_scores[i]
            emoji_char = emoji_map.get(name.lower(), "❓")  # match on lowercase name

            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"<div style='font-size: 40px'>{emoji_char}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{name}**")
                st.progress(prob)
