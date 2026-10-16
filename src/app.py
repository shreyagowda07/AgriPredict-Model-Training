
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "agripredict_improved_model.keras"
LABEL_PATH = PROJECT_DIR / "models" / "agripredict_improved_class_names.json"

st.set_page_config(
    page_title="AgriPredict AI",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 AgriPredict AI")
st.subheader("Plant Disease Detection")
st.write("Upload a leaf image to get an AI-based prediction.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

if not MODEL_PATH.exists() or not LABEL_PATH.exists():
    st.error("Model or class-name file is missing. Check your models folder.")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf image", width=350)

        if st.button("🔍 Detect Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_model()
                labels = load_labels()

                image = image.resize((128, 128))
                array = np.asarray(image, dtype=np.float32)
                array = np.expand_dims(array, axis=0)

                predictions = model.predict(array, verbose=0)[0]

                if len(predictions) != len(labels):
                    st.error("Model output and class labels do not match.")
                    st.stop()

                top_indices = np.argsort(predictions)[-3:][::-1]
                best_index = int(top_indices[0])
                plant, disease = labels[best_index].split("___", 1)

                st.success("Prediction completed!")
                st.markdown("### 🌱 Prediction Result")
                st.write("**Plant:**", plant.replace("_", " "))
                st.write("**Predicted condition:**", disease.replace("_", " "))
                st.metric(
                    "Model confidence",
                    f"{predictions[best_index] * 100:.2f}%"
                )

                st.markdown("### Top 3 predictions")
                for index in top_indices:
                    name = labels[int(index)].replace("___", " — ")
                    st.write(f"{name.replace('_', ' ')}: {predictions[index] * 100:.2f}%")

                st.caption(
                    "AI predictions are not a confirmed diagnosis. "
                    "Accuracy may vary with image quality and real-world conditions."
                )

    except Exception as error:
        st.error(f"Something went wrong: {error}")
