import os
import json
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf  # noqa: F401
from src.config import MODEL_DIR, IMG_SIZE, CLASS_MAP_PATH

st.set_page_config(page_title="Fruits-360 EA-CNN", layout="wide")

st.sidebar.title("Fruits-360 EA-CNN")
st.sidebar.markdown("""
- Enhanced-Attention CNN ([DOI 10.1016/j.heliyon.2024.e28006](https://doi.org/10.1016/j.heliyon.2024.e28006))
- Drag-and-drop fruit/vegetable image for classification.
""")

metrics_path = "models/metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
    st.sidebar.subheader("Model Metrics")
    st.sidebar.json(metrics["classification_report"])
else:
    st.sidebar.info("Run `make evaluate` to see metrics here.")

if not os.path.exists(MODEL_DIR):
    st.error("Model not found. Please train first: `make train`")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_DIR)

@st.cache_data
def load_class_map():
    with open(CLASS_MAP_PATH) as f:
        return json.load(f)

model = load_model()
class_map = load_class_map()

st.title("🍎 Fruits-360 Classifier (EA-CNN)")
file = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = arr[np.newaxis, ...]
    preds = model.predict(arr)[0]
    top5_idx = np.argsort(preds)[-5:][::-1]
    top5_labels = [class_map[str(i)] for i in top5_idx]
    top5_scores = preds[top5_idx]
    pred_label = class_map[str(np.argmax(preds))]
    pred_conf = float(np.max(preds))
    st.image(img, caption=f"Prediction: {pred_label} ({pred_conf:.2%})", use_column_width=False)
    st.bar_chart({"Class": top5_labels, "Confidence": top5_scores})
    if st.button("Explain (LIME)"):
        from src.lime_visualise import explain
        # Save uploaded file to disk for LIME
        temp_path = "app/temp_uploaded_image.png"
        img.save(temp_path)
        explain(temp_path, "app/overlay.png")
        st.image("app/overlay.png", caption="LIME Explanation", use_column_width=False) 