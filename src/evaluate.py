import os
import json
import numpy as np
import tensorflow as tf  # noqa: F401
from sklearn.metrics import confusion_matrix, classification_report
from .config import MODEL_DIR, DATA_DIR, CLASS_MAP_PATH
from .dataload import get_datasets

def main():
    _, _, test_ds, class_names = get_datasets(data_dir=DATA_DIR)
    model = tf.keras.models.load_model(MODEL_DIR)
    y_true = []
    y_pred = []
    for x, y in test_ds:
        preds = model.predict(x)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Evaluation complete. Metrics written to models/metrics.json.")

if __name__ == '__main__':
    main() 