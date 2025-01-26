from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import StringIO

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)  # This will allow all domains by default

# Load the trained Keras model
model_path = 'ecoguard_model.keras'
model = load_model(model_path)

# Define the image input shape (modify this based on your model's expected input)
input_shape = (64, 64, 3)  # Change this to match your model's input shape

# Route to upload file and predict in a single request
@app.route('/upload_and_predict', methods=['POST'])
def upload_and_predict():
    """
    Upload a CSV file and use it to make predictions.
    Expected input: Form-data with 'file' as the CSV.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the uploaded file is a CSV
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format, only CSVs are allowed!'}), 400

    try:
        # Read the CSV file directly into a pandas DataFrame
        file_contents = file.stream.read().decode('utf-8')  # Read the file as a string
        csv_data = StringIO(file_contents)  # Convert string to StringIO object (acts like a file)
        df = pd.read_csv(csv_data)  # Read CSV from StringIO into DataFrame

        # Assuming the CSV contains data for prediction (process this as needed)
        # You can now preprocess the data based on the model's requirements

        # Example: Assume the CSV contains a column for image file paths or raw image data
        # Process the CSV and make predictions

        # Example for image column processing (adjust according to your CSV content)
        # If your CSV has a column named 'image_path', for example, you would process images here.
        if 'image_path' in df.columns:
            img = image.load_img(df['image_path'][0], target_size=input_shape[:2])  # Resize image
            img_array = image.img_to_array(img)  # Convert image to array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array /= 255.0  # Normalize the image

            # Make the prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class

            # Map predicted class to a label (adjust this according to your model)
            class_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}  # Example class mapping
            result = class_labels.get(predicted_class, "Unknown")

            return jsonify({
                'message': f'Predictions made successfully from uploaded CSV.',
                'predicted_class': result,
                'confidence': float(np.max(prediction))
            })
        else:
            return jsonify({'error': 'No valid image column found in CSV for prediction.'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
