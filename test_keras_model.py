#!/usr/bin/env python3
"""
Test the original Keras model
"""

import numpy as np
import tensorflow as tf
from src.config import MODEL_DIR, DATA_DIR
from src.dataload import get_datasets

def test_keras_model():
    """Test the original Keras model"""
    print("=== Testing Keras Model ===")
    
    if not tf.io.gfile.exists(MODEL_DIR):
        print(f"Keras model not found at {MODEL_DIR}")
        return
    
    # Load Keras model
    model = tf.keras.models.load_model(MODEL_DIR)
    print(f"Model loaded successfully")
    print(f"Model summary:")
    model.summary()
    
    # Load test data
    _, _, test_ds, class_names = get_datasets(data_dir=DATA_DIR)
    
    # Test a few samples
    for i, (x, y) in enumerate(test_ds):
        if i >= 3:  # Test first 3 batches
            break
        
        print(f"\n--- Batch {i} ---")
        print(f"Input shape: {x.shape}")
        print(f"Input range: [{x.numpy().min():.3f}, {x.numpy().max():.3f}]")
        
        # Check for NaN in input
        if np.any(np.isnan(x.numpy())):
            print("WARNING: NaN values found in input!")
        
        # Make predictions
        preds = model.predict(x, verbose=0)
        
        for j in range(min(2, x.shape[0])):
            print(f"Sample {j}:")
            print(f"  True label: {class_names[np.argmax(y[j])]}")
            print(f"  Predictions: {preds[j]}")
            print(f"  Predicted label: {class_names[np.argmax(preds[j])]}")
            print(f"  Confidence: {np.max(preds[j]):.2%}")
            
            # Check for NaN in predictions
            if np.any(np.isnan(preds[j])):
                print("  WARNING: NaN values in predictions!")

if __name__ == "__main__":
    test_keras_model() 