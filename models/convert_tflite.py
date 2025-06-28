import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf  # noqa: F401

# Path to the improved transfer learning model and output TFLite model
KERAS_MODEL_PATH = "models/fast_small_model_transfer_final.h5"
TFLITE_PATH = "models/ea_cnn.tflite"

def convert():
    print(f"Converting improved transfer learning model to TFLite...")
    
    # Load the Keras model
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Float32 export
    tflite_model = converter.convert()
    
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {TFLITE_PATH}")
    print("This improved model should perform much better!")

    # # INT8 quantization (uncomment to use, requires representative dataset)
    # def representative_data_gen():
    #     for _ in range(100):
    #         # Provide a batch of input images as np.float32
    #         yield [np.random.rand(1, 100, 100, 3).astype(np.float32)]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_data_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # tflite_quant_model = converter.convert()
    # with open('models/ea_cnn_int8.tflite', 'wb') as f:
    #     f.write(tflite_quant_model)
    # print("INT8 TFLite model saved to models/ea_cnn_int8.tflite")

if __name__ == '__main__':
    convert() 