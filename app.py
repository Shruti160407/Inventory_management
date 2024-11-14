from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_processing import preprocess_data

app = Flask(__name__)

# Load the trained model
model = load_model('model/sca_prediction_model.h5')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract ECG file from the request
    ecg_file = request.files['ecg_data']
    temporal_data = np.array(request.form['temporal_data'], dtype=np.float32)
    structured_data = np.array(request.form['structured_data'], dtype=np.float32)

    # Preprocess ECG data
    ecg_data = preprocess_data(ecg_file)  # Assuming ecg_file is a CSV

    # Reshape inputs to match model input requirements
    ecg_data = ecg_data.reshape(1, 300, 1)  # Modify according to your data dimensions
    temporal_data = temporal_data.reshape(1, 100, 1)
    structured_data = structured_data.reshape(1, 10)

    # Make prediction
    prediction = model.predict([ecg_data, temporal_data, structured_data])

    # Return prediction result
    return jsonify({'prediction': int(prediction[0] > 0.5)})

if __name__ == "__main__":
    app.run(debug=True)
