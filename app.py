
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import predict
import cv2

st.title(" Deepfake Image Classifier")
st.write("Upload an image to classify it as **Real** or **Fake** using EfficientNetB0")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_efficientnetb0.h5")

model = load_model()
class_names = ["Fake", "Real"]

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL to NumPy
    img_array = np.array(image)

    # Predict
    pred = predict(model, img_array)
    class_idx = np.argmax(pred)
    confidence = pred[class_idx]

    st.write(f"### Prediction: **{class_names[class_idx]}**")
    st.write(f"Confidence: `{confidence:.2%}`")
