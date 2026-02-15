import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

import os
import shutil

# clear ultralytics cache
if os.path.exists("/home/adminuser/.cache/ultralytics"):
    shutil.rmtree("/home/adminuser/.cache/ultralytics")

# Load YOLOv8 Model
model = YOLO("yolov8n.pt", task="detect")


# Title
st.title("AI Suspicious Activity Detection using YOLOv8")
st.write("Upload an image to detect objects / suspicious indicators")

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run YOLO Detection
    results = model(img)

    # Draw Bounding Boxes
    annotated_img = results[0].plot()

    # Convert BGR → RGB for Streamlit
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Show Output
    st.image(annotated_img, caption="Detection Result", use_column_width=True)

    st.success("Detection Completed")
