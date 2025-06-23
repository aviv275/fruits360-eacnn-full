import os
import json
from typing import Tuple
import tensorflow as tf  # noqa: F401
from tensorflow.keras import layers
from .config import IMG_SIZE, BATCH_SZ, DATA_DIR, CLASS_MAP_PATH

def get_class_names(data_dir: str) -> list[str]:
    train_dir = os.path.join(data_dir, 'train')
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return class_names

def export_class_mapping(class_names: list[str], path: str):
    mapping = {i: name for i, name in enumerate(class_names)}
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=2)

def get_datasets(img_size: Tuple[int, int] = IMG_SIZE, batch_size: int = BATCH_SZ, data_dir: str = DATA_DIR):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    val_dir = os.path.join(data_dir, 'val')
    if os.path.exists(val_dir):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            label_mode='categorical',
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        val_ds = None
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    class_names = train_ds.class_names
    export_class_mapping(class_names, CLASS_MAP_PATH)
    AUTOTUNE = tf.data.AUTOTUNE
    aug = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.Rescaling(1./255)
    ])
    def preprocess(x, y):
        return aug(x), y
    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    if val_ds:
        val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names 