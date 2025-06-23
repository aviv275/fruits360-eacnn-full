import tensorflow as tf  # noqa: F401
from tensorflow.keras import layers, models
from typing import Tuple

def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Enhanced Attention Conv Block."""
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Attention gate
    gate = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, gate])
    # Mixed pooling
    maxp = layers.MaxPooling2D(2)(x)
    avgp = layers.AveragePooling2D(2)(x)
    x = layers.Concatenate()([maxp, avgp])  # doubles channels
    return x

def build_eacnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Builds the EA-CNN model."""
    inputs = layers.Input(shape=input_shape)
    x = conv_block(inputs, 32)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='softplus')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name='EA-CNN')
    return model 