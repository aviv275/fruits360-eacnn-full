#!/usr/bin/env python3
"""
Improved training script for small 10-fruit dataset with transfer learning
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
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

def create_transfer_learning_model():
    """Create a transfer learning model using MobileNetV2"""
    # Load pre-trained MobileNetV2 (without top layer)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')  # 10 classes
    ])
    
    return model, base_model

def create_data_generators():
    """Create data generators with strong augmentation"""
    # Strong data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
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
    """Main training function with transfer learning"""
    print("="*60)
    print("IMPROVED TRAINING - SMALL 10-FRUIT DATASET")
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
    print("\nCreating transfer learning model...")
    model, base_model = create_transfer_learning_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODEL_SAVE_DIR / f"{MODEL_NAME}_transfer_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen base model
    print(f"\nPhase 1: Training with frozen base model for {EPOCHS} epochs...")
    start_time = time.time()
    
    history1 = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune the base model
    print(f"\nPhase 2: Fine-tuning base model...")
    base_model.trainable = True
    
    # Freeze early layers, fine-tune later layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history2 = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final model
    model.save(str(MODEL_SAVE_DIR / f"{MODEL_NAME}_transfer_final.h5"))
    print(f"Model saved to {MODEL_SAVE_DIR / f'{MODEL_NAME}_transfer_final.h5'}")
    
    # Plot training history
    plot_training_history(history1, history2)
    
    # Evaluate on validation set
    evaluate_on_validation_set(model, val_generator)
    
    return model, history1, history2

def plot_training_history(history1, history2):
    """Plot training history for both phases"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Combine histories
    epochs1 = len(history1.history['accuracy'])
    epochs2 = len(history2.history['accuracy'])
    
    # Accuracy
    ax1.plot(range(epochs1), history1.history['accuracy'], label='Phase 1 Training', color='blue')
    ax1.plot(range(epochs1), history1.history['val_accuracy'], label='Phase 1 Validation', color='blue', linestyle='--')
    ax1.plot(range(epochs1, epochs1 + epochs2), history2.history['accuracy'], label='Phase 2 Training', color='red')
    ax1.plot(range(epochs1, epochs1 + epochs2), history2.history['val_accuracy'], label='Phase 2 Validation', color='red', linestyle='--')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(range(epochs1), history1.history['loss'], label='Phase 1 Training', color='blue')
    ax2.plot(range(epochs1), history1.history['val_loss'], label='Phase 1 Validation', color='blue', linestyle='--')
    ax2.plot(range(epochs1, epochs1 + epochs2), history2.history['loss'], label='Phase 2 Training', color='red')
    ax2.plot(range(epochs1, epochs1 + epochs2), history2.history['val_loss'], label='Phase 2 Validation', color='red', linestyle='--')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'transfer_learning_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_on_validation_set(model, val_generator):
    """Evaluate model on validation set"""
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
    model, history1, history2 = train_model() 