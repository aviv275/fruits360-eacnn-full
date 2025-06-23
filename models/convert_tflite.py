import tensorflow as tf  # noqa: F401
from src.config import MODEL_DIR, TFLITE_PATH

def convert():
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    # Float32 export
    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {TFLITE_PATH}")

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