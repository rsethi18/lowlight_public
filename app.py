import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io

# Load the model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best-medium-weights.pt')

model = load_model()

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