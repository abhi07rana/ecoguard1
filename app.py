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

@app.route('/upload', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)

        # Check for required columns
        required_columns = ['pollution_level', 'air_quality_index', 'temperature', 'humidity']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'Missing required columns in CSV. Expected columns: pollution_level, air_quality_index, temperature, humidity'}), 400

        # Extract only the required features (pollution_level, air_quality_index, temperature, humidity)
        data = df[['pollution_level', 'air_quality_index', 'temperature', 'humidity']].values

        # Normalize the data (this depends on how your model was trained)
        # If your model was trained using StandardScaler, uncomment the following line:
        # data = scaler.fit_transform(data)  # Normalize data

        # Reshape data to match the model's expected input shape (None, 2, 1, 1)
        reshaped_data = data.reshape((-1, 2, 1, 1))  # Adjust reshape based on your model's input shape

        # Normalize data if needed (e.g., divide by 255 if the model was trained with normalized data)
        reshaped_data = reshaped_data / 255.0  # Example normalization, adjust as per your model

        # Make the prediction
        prediction = model.predict(reshaped_data)
        
        # Debugging: print the raw prediction
        print("Raw prediction output:", prediction)

        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class
        
        # Map predicted class to a label (adjust based on your model's output)
        class_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        result = class_labels.get(predicted_class, "Unknown")

        # Get the confidence as a percentage and round to 2 decimal places
        confidence = float(np.max(prediction)) * 100
        confidence = round(confidence, 2)

        return jsonify({
            'predicted_class': result, 
            'confidence': confidence,  # Confidence as percentage
            'raw_prediction': prediction.tolist()  # Returning raw prediction for debugging
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
