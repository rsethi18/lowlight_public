import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
import gdown

# Function to download the model file from Google Drive using gdown
def download_model(url, destination):
    if not os.path.exists(destination):
        st.info(f"Downloading model file from {url}")
        gdown.download(url, destination, quiet=False)

# URL of the model file on Google Drive
model_url = 'https://drive.google.com/uc?export=download&id=16yGC7FY2UEHIHmOwbvqUR1VC1RHj1DKF'
model_path = 'best-medium.pt'

# Download the model file if it doesn't exist
download_model(model_url, model_path)

# Function to verify the downloaded model file
def verify_model_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            first_bytes = f.read(2)
        if first_bytes == b'\x80\x02':
            return True
        else:
            st.error(f"File at {file_path} is not a valid PyTorch model. First bytes: {first_bytes}")
            return False
    except Exception as e:
        st.error(f"Error verifying model file: {e}")
        return False

# Verify the model file
if not verify_model_file(model_path):
    st.error("Downloaded model file is invalid. Please check the download link and try again.")
    st.stop()

# Load the model
@st.cache_resource
def load_model():
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

def process_image(image, confidence_threshold):
    results = model(image)
    
    img = np.array(image)
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        if conf >= confidence_threshold:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{model.names[int(cls)]}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    return Image.fromarray(img)

st.title("Low-Light Vehicle and Pedestrian Detection")
st.markdown("* Vehicles includes cars, buses, and motorbikes")
st.markdown("* This model may produce inaccurate results in some cases")

option = st.radio("Choose an option:", ("Upload Image", "Demo"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25)
        
        if st.button("Detect Objects"):
            processed_image = process_image(image, confidence_threshold)
            st.image(processed_image, caption="Processed Image", use_column_width=True)
            
            buf = io.BytesIO()
            processed_image.save(buf, format="PNG")
            btn = st.download_button(
                label="Download processed image",
                data=buf.getvalue(),
                file_name="processed_image.png",
                mime="image/png"
            )

elif option == "Demo":
    st.subheader("Demo: Original and Processed (at 25% confidence)")
    
    input_image = Image.open("initial.png")
    output_image = Image.open("processed.png")
    
    st.image(input_image, caption="Original Image", use_column_width=True)
    st.image(output_image, caption="Processed Image", use_column_width=True)
