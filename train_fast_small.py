#!/usr/bin/env python3
"""
Fast training script for small 10-fruit dataset
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import configuration
from src.config_fast_small import *

def load_class_mapping():
    """Load class mapping from JSON file"""
    with open(CLASS_MAPPING_FILE, 'r') as f:
        class_mapping = json.load(f)
    return class_mapping

def create_fast_model():
    """Create a fast CNN model for small dataset"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Convolutional layers
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        # Global average pooling for faster training
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')  # 10 classes
    ])
    
    return model

def create_data_generators():
    """Create data generators with augmentation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def train_model():
    """Main training function"""
    print("="*60)
    print("FAST TRAINING - SMALL 10-FRUIT DATASET")
    print("="*60)
    
    # Load class mapping
    class_mapping = load_class_mapping()
    print(f"Classes: {list(class_mapping.values())}")
    
    # Create data generators
    print("\nCreating data generators...")
    train_generator, val_generator = create_data_generators()
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    
    # Create model
    print("\nCreating model...")
    model = create_fast_model()
    
    # Compile model with legacy optimizer for M1/M2 Macs
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODEL_SAVE_DIR / f"{MODEL_NAME}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final model
    model.save(str(MODEL_SAVE_DIR / f"{MODEL_NAME}_final.h5"))
    print(f"Model saved to {MODEL_SAVE_DIR / f'{MODEL_NAME}_final.h5'}")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    evaluate_on_test_set(model, val_generator)
    
    return model, history

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_on_test_set(model, val_generator):
    """Evaluate model on test set"""
    print("\n" + "="*40)
    print("MODEL EVALUATION")
    print("="*40)
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=1)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Predictions
    predictions = model.predict(val_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    
    # Print some sample predictions
    class_names = list(load_class_mapping().values())
    print(f"\nSample predictions:")
    for i in range(min(10, len(predicted_classes))):
        true_class = class_names[true_classes[i]]
        pred_class = class_names[predicted_classes[i]]
        confidence = np.max(predictions[i])
        print(f"  True: {true_class:15} | Pred: {pred_class:15} | Confidence: {confidence:.3f}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the model
    model, history = train_model() 