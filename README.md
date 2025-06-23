# Fruits-360 Enhanced-Attention CNN (EA-CNN)

## Overview
This project implements an Enhanced-Attention CNN (EA-CNN) for fruit and vegetable classification using the Fruits-360 100×100 dataset. It includes training, evaluation, LIME explainability, TFLite export, and a Streamlit web app for drag-and-drop inference.

**EA-CNN Reference:**
> "Enhanced attention convolutional neural network for fruit and vegetable classification"<br>
> DOI: [10.1016/j.heliyon.2024.e28006](https://doi.org/10.1016/j.heliyon.2024.e28006)

---

## Quick Start

1. **Download Dataset**
   - Download `fruits-360_100x100.zip` from Kaggle and unzip so that:
     - `data/raw/fruits-360/train/` and `data/raw/fruits-360/test/` exist.

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**
   ```bash
   python src/train.py
   ```

4. **Evaluate the Model**
   ```bash
   python src/evaluate.py
   ```

5. **Export to TFLite**
   ```bash
   python models/convert_tflite.py
   ```

6. **Run the Streamlit App**
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## Project Structure
```
root/
│  README.md
│  requirements.txt
│  Makefile
│  .gitignore
│
├─data/
│  make_dataset.py
│  raw/  # Place Fruits-360 here (train/, test/)
│
├─src/
│  config.py
│  dataload.py
│  eacnn.py
│  train.py
│  evaluate.py
│  lime_visualise.py
│
├─models/
│  convert_tflite.py
│  ea_cnn_savedmodel/
│  ea_cnn.tflite
│
└─app/
   streamlit_app.py
   class_mapping.json
```

---

## Model Diagram
The EA-CNN architecture is visualized using `tf.keras.utils.plot_model` in `src/eacnn.py`.

---

## Deployment
- **SavedModel:** Created in `models/ea_cnn_savedmodel/` after training.
- **TFLite:** Exported as `models/ea_cnn.tflite` via `models/convert_tflite.py` (default: fp32; int8 quantization code included as comments).
- **Streamlit App:** Loads the SavedModel for inference. If missing, prompts user to train first.

---

## Makefile Targets
- `train`     : Train the EA-CNN model
- `evaluate`  : Evaluate and generate metrics
- `tflite`    : Export to TFLite
- `app`       : Launch Streamlit app
- `clean`     : Remove models, logs, and outputs

---

## Citation
If you use this code, please cite the original EA-CNN paper:
> DOI: [10.1016/j.heliyon.2024.e28006](https://doi.org/10.1016/j.heliyon.2024.e28006) 