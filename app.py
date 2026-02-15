import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("AI Suspicious Activity Detection using YOLOv8")

# Load model
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Run detection
    results = model(img)

    # Get annotated image
    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result")
