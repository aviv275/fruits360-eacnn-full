#!/usr/bin/env python3
"""
Debug script to test model predictions and identify issues
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import MODEL_DIR, DATA_DIR
from src.dataload import get_datasets

def test_tflite_predictions():
    """Test TFLite model predictions on test data"""
    print("=== Testing TFLite Model ===")
    
    # Load TFLite model
    tflite_path = "models/ea_cnn.tflite"
    if not os.path.exists(tflite_path):
        print(f"TFLite model not found at {tflite_path}")
        return
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Load test data
    _, _, test_ds, class_names = get_datasets(data_dir=DATA_DIR)
    
    # Test a few samples
    predictions = []
    for i, (x, y) in enumerate(test_ds):
        if i >= 5:  # Test first 5 batches
            break
        
        for j in range(min(3, x.shape[0])):  # Test first 3 samples from each batch
            sample = x[j:j+1]
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], sample.numpy().astype(np.float32))
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            
            true_label = class_names[np.argmax(y[j])]
            pred_label = class_names[np.argmax(preds)]
            pred_conf = float(np.max(preds))
            
            print(f"Sample {i}-{j}: True={true_label}, Pred={pred_label} ({pred_conf:.2%})")
            print(f"  Raw predictions: {preds}")
            predictions.append((true_label, pred_label, pred_conf))
    
    return predictions

def test_keras_model():
    """Test Keras model predictions on test data"""
    print("\n=== Testing Keras Model ===")
    
    if not os.path.exists(MODEL_DIR):
        print(f"Keras model not found at {MODEL_DIR}")
        return
    
    # Load Keras model
    model = tf.keras.models.load_model(MODEL_DIR)
    
    # Load test data
    _, _, test_ds, class_names = get_datasets(data_dir=DATA_DIR)
    
    # Test a few samples
    predictions = []
    for i, (x, y) in enumerate(test_ds):
        if i >= 5:  # Test first 5 batches
            break
        
        preds = model.predict(x, verbose=0)
        
        for j in range(min(3, x.shape[0])):
            true_label = class_names[np.argmax(y[j])]
            pred_label = class_names[np.argmax(preds[j])]
            pred_conf = float(np.max(preds[j]))
            
            print(f"Sample {i}-{j}: True={true_label}, Pred={pred_label} ({pred_conf:.2%})")
            print(f"  Raw predictions: {preds[j]}")
            predictions.append((true_label, pred_label, pred_conf))
    
    return predictions

def test_random_images():
    """Test with random noise to see if model is stuck"""
    print("\n=== Testing with Random Images ===")
    
    # Load TFLite model
    tflite_path = "models/ea_cnn.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test with random noise
    for i in range(5):
        # Generate random image
        random_img = np.random.rand(1, 100, 100, 3).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], random_img)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        pred_label = f"Class {np.argmax(preds)}"
        pred_conf = float(np.max(preds))
        
        print(f"Random {i}: Pred={pred_label} ({pred_conf:.2%})")
        print(f"  Raw predictions: {preds}")

def test_streamlit_preprocessing():
    """Test the exact preprocessing used in Streamlit app"""
    print("\n=== Testing Streamlit Preprocessing ===")
    
    # Load TFLite model
    tflite_path = "models/ea_cnn.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load class mapping
    with open("app/class_mapping.json") as f:
        class_map = json.load(f)
    
    # Test with a sample image from test data
    _, _, test_ds, _ = get_datasets(data_dir=DATA_DIR)
    
    for i, (x, y) in enumerate(test_ds):
        if i >= 1:  # Just test first batch
            break
        
        sample = x[0:1]  # Take first sample
        
        # Apply Streamlit preprocessing (divide by 255)
        sample_streamlit = sample.numpy() / 255.0
        
        # Test original preprocessing
        interpreter.set_tensor(input_details[0]['index'], sample.numpy().astype(np.float32))
        interpreter.invoke()
        preds_original = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Test Streamlit preprocessing
        interpreter.set_tensor(input_details[0]['index'], sample_streamlit.astype(np.float32))
        interpreter.invoke()
        preds_streamlit = interpreter.get_tensor(output_details[0]['index'])[0]
        
        print(f"Original preprocessing: {preds_original}")
        print(f"Streamlit preprocessing: {preds_streamlit}")
        print(f"Difference: {np.abs(preds_original - preds_streamlit).max()}")

if __name__ == "__main__":
    print("Starting prediction debugging...")
    
    # Test TFLite model
    tflite_preds = test_tflite_predictions()
    
    # Test Keras model
    keras_preds = test_keras_model()
    
    # Test random images
    test_random_images()
    
    # Test preprocessing
    test_streamlit_preprocessing()
    
    print("\n=== Summary ===")
    if tflite_preds:
        print(f"TFLite predictions tested: {len(tflite_preds)}")
    if keras_preds:
        print(f"Keras predictions tested: {len(keras_preds)}") 