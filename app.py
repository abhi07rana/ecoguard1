from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)  # This will allow all domains by default

# Load the trained Keras model
model_path = 'ecoguard_model.keras'
model = load_model(model_path)

# Define the image input shape (modify this based on your model's expected input)
input_shape = (64, 64, 3)  # Change this to match your model's input shape

@app.route('/')
def home():
    return "EcoGuard Model API - Running"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the environmental health risk based on the input image.
    Expected input: Form-data with 'file' as the image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the image file
        img = image.load_img(file, target_size=input_shape[:2])  # Resize the image to the input shape
        img_array = image.img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image

        # Make the prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class

        # Map predicted class to a label (if needed)
        class_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}  # Example class mapping
        result = class_labels.get(predicted_class, "Unknown")

        return jsonify({'predicted_class': result, 'confidence': float(np.max(prediction))})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles file uploads and saves the uploaded file to the server.
    Expected input: Form-data with 'file' as the image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the file to a specific location on the server
        upload_folder = 'uploads/'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        return jsonify({'message': f'File uploaded successfully! Saved to {filepath}'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
