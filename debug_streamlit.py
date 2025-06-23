#!/usr/bin/env python3
"""
Debug Streamlit prediction pipeline
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# Configuration
TFLITE_PATH = "models/ea_cnn.tflite"
IMG_SIZE = (100, 100)
CLASS_MAP_PATH = "app/class_mapping.json"

def test_prediction_pipeline():
    """Test the exact prediction pipeline used in Streamlit"""
    print("=== Testing Streamlit Prediction Pipeline ===")
    
    # Check if files exist
    if not os.path.exists(TFLITE_PATH):
        print(f"TFLite model not found at {TFLITE_PATH}")
        return
    
    if not os.path.exists(CLASS_MAP_PATH):
        print(f"Class mapping not found at {CLASS_MAP_PATH}")
        return
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Load class mapping
    with open(CLASS_MAP_PATH) as f:
        class_map = json.load(f)
    
    # Create a test image (random for now)
    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(test_img)
    
    print(f"Test image shape: {np.array(img).shape}")
    print(f"Test image dtype: {np.array(img).dtype}")
    print(f"Test image range: [{np.array(img).min()}, {np.array(img).max()}]")
    
    # Apply Streamlit preprocessing
    arr = np.array(img)  # No division by 255
    arr = arr[np.newaxis, ...].astype(np.float32)
    
    print(f"Preprocessed array shape: {arr.shape}")
    print(f"Preprocessed array dtype: {arr.dtype}")
    print(f"Preprocessed array range: [{arr.min()}, {arr.max()}]")
    
    # Check for NaN or inf values
    if np.any(np.isnan(arr)):
        print("WARNING: NaN values found in input array!")
    if np.any(np.isinf(arr)):
        print("WARNING: Inf values found in input array!")
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], arr)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    
    print(f"Raw predictions: {preds}")
    print(f"Predictions dtype: {preds.dtype}")
    print(f"Predictions shape: {preds.shape}")
    
    # Check for NaN or inf in predictions
    if np.any(np.isnan(preds)):
        print("WARNING: NaN values found in predictions!")
    if np.any(np.isinf(preds)):
        print("WARNING: Inf values found in predictions!")
    
    # Calculate confidence
    pred_conf = float(np.max(preds))
    print(f"Max prediction value: {pred_conf}")
    print(f"Max prediction type: {type(pred_conf)}")
    
    # Check if confidence is valid
    if np.isnan(pred_conf):
        print("ERROR: Confidence is NaN!")
    elif np.isinf(pred_conf):
        print("ERROR: Confidence is Inf!")
    else:
        print(f"Confidence: {pred_conf:.2%}")
    
    # Get class prediction
    pred_idx = np.argmax(preds)
    pred_label = class_map[str(pred_idx)]
    print(f"Predicted class: {pred_label} (index {pred_idx})")
    
    # Test with a real image from the dataset
    print("\n=== Testing with Real Dataset Image ===")
    from src.config import DATA_DIR
    from src.dataload import get_datasets
    
    _, _, test_ds, class_names = get_datasets(data_dir=DATA_DIR)
    
    for i, (x, y) in enumerate(test_ds):
        if i >= 1:  # Just test first batch
            break
        
        sample = x[0:1].numpy()  # Take first sample
        
        print(f"Sample shape: {sample.shape}")
        print(f"Sample range: [{sample.min()}, {sample.max()}]")
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], sample.astype(np.float32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        print(f"Sample predictions: {preds}")
        
        # Calculate confidence
        pred_conf = float(np.max(preds))
        print(f"Sample confidence: {pred_conf:.2%}")
        
        if np.isnan(pred_conf):
            print("ERROR: Sample confidence is NaN!")
        elif np.isinf(pred_conf):
            print("ERROR: Sample confidence is Inf!")

if __name__ == "__main__":
    test_prediction_pipeline() 