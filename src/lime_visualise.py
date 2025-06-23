import sys
import numpy as np
import tensorflow as tf  # noqa: F401
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
from .config import MODEL_DIR, IMG_SIZE

def explain(image_path: str, output_path: str = 'overlay.png'):
    model = tf.keras.models.load_model(MODEL_DIR)
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = arr[np.newaxis, ...]
    explainer = lime_image.LimeImageExplainer()
    def predict_fn(x):
        return model.predict(x)
    explanation = explainer.explain_instance(
        arr[0], predict_fn, top_labels=1, hide_color=0, num_samples=1000
    )
    label = explanation.top_labels[0] if hasattr(explanation, 'top_labels') and explanation.top_labels else 0
    temp, mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=5, hide_rest=False
    )
    overlay = mark_boundaries(temp, mask)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    overlay_img.save(output_path)
    print(f"LIME overlay saved to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/lime_visualise.py <image_path> [output_path]")
        sys.exit(1)
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'overlay.png'
    explain(image_path, output_path) 