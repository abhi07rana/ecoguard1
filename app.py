from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load the trained Keras model
model_path = 'ecoguard_model.keras'
model = load_model(model_path)

# Initialize a scaler (optional, depends on your model's preprocessing)
scaler = StandardScaler()

@app.route('/')
def home():
    return "EcoGuard Model API - Running"

@app.route('/upload_and_predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format, only CSVs are allowed!'}), 400

    try:
        # Read the CSV file
        file_contents = file.stream.read().decode('utf-8')
        df = pd.read_csv(pd.compat.StringIO(file_contents))

        # Check if necessary columns are present
        required_columns = ['pollution_level', 'air_quality_index', 'temperature', 'humidity']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': f'Missing required columns in CSV. Expected columns: {", ".join(required_columns)}'}), 400

        # Preprocess the data (e.g., normalize or scale)
        features = df[required_columns].values
        features = scaler.fit_transform(features)  # Standardizing or normalizing features if needed

        # Predict the health risk using the model
        predictions = model.predict(features)
        predicted_classes = np.argmax(predictions, axis=1)  # Assuming multi-class classification

        # Map predicted classes to labels (you can adjust this based on your model's output)
        class_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        predicted_labels = [class_labels.get(cls, "Unknown") for cls in predicted_classes]

        return jsonify({
            'message': 'Prediction made successfully from uploaded CSV.',
            'predicted_classes': predicted_labels,
            'predicted_confidence': predictions.tolist()  # Or a confidence score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
