import os
import json
import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Configuration
TFLITE_PATH = "models/ea_cnn.tflite"
IMG_SIZE = (100, 100)
CLASS_MAP_PATH = "app/class_mapping.json"

st.set_page_config(page_title="Fruits-360 EA-CNN", layout="wide")

st.sidebar.title("Fruits-360 EA-CNN")
st.sidebar.markdown("""
- Enhanced-Attention CNN ([DOI 10.1016/j.heliyon.2024.e28006](https://doi.org/10.1016/j.heliyon.2024.e28006))
- Using TensorFlow Lite for better compatibility
- Drag-and-drop fruit/vegetable image for classification.
""")

# Check if TFLite model exists
if not os.path.exists(TFLITE_PATH):
    st.error("TFLite model not found. Please run: `python3 models/convert_tflite.py`")
    st.stop()

# Check if class mapping exists
if not os.path.exists(CLASS_MAP_PATH):
    st.error("Class mapping not found. Please ensure app/class_mapping.json exists.")
    st.stop()

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_class_map():
    with open(CLASS_MAP_PATH) as f:
        return json.load(f)

interpreter = load_model()
class_map = load_class_map()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🍎 Fruits-360 Classifier (EA-CNN)")
file = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = arr[np.newaxis, ...].astype(np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], arr)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    
    top5_idx = np.argsort(preds)[-5:][::-1]
    top5_labels = [class_map[str(i)] for i in top5_idx]
    top5_scores = preds[top5_idx]
    pred_label = class_map[str(np.argmax(preds))]
    pred_conf = float(np.max(preds))
    
    st.image(img, caption=f"Prediction: {pred_label} ({pred_conf:.2%})", use_column_width=False)
    st.bar_chart({"Class": top5_labels, "Confidence": top5_scores})
    
    st.success(f"✅ Predicted: **{pred_label}** with {pred_conf:.2%} confidence")
    
    # Show top 5 predictions
    st.subheader("Top 5 Predictions:")
    for i, (label, score) in enumerate(zip(top5_labels, top5_scores)):
        st.write(f"{i+1}. {label}: {score:.2%}") 