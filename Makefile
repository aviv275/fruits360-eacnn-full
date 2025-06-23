.PHONY: train evaluate tflite app clean

train:
	python src/train.py

evaluate:
	python src/evaluate.py

tflite:
	python models/convert_tflite.py

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf models/ea_cnn_savedmodel models/ea_cnn.tflite logs/ models/metrics.json app/__pycache__/ src/__pycache__/ 