import os
import logging
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Enable Logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder='templates')

# Load AI Model with Better Error Handling
try:
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    model.compile()  # Ensures model is ready for predictions
    logging.info("AI Model Loaded Successfully")
except Exception as e:
    logging.error(f"Error Loading AI Model: {e}")
    model = None  # Prevents app from crashing


# Function to preprocess image for AI model
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')  # Ensures RGB format
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logging.error(f"Error Preprocessing Image: {e}")
        return None


# Function to perform AI-based object recognition
def recognize_objects(image_path):
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        logging.error("Failed to preprocess image")
        return None

    if model is None:
        logging.error("AI Model is not loaded")
        return None

    try:
        predictions = model.predict(processed_image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        logging.error(f"Error Processing Image in AI Model: {e}")
        return None


# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Fix Upload API
@app.route('/upload', methods=['POST'])
def upload_image():
    logging.info("Received image upload request")

    if 'file' not in request.files:
        logging.error("No file found in request")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("Empty file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    # Save the file temporarily
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)  # Ensure directory exists
    file_path = os.path.join(upload_folder, file.filename)
    
    try:
        file.save(file_path)
        logging.info(f"File saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        return jsonify({'error': 'Error saving file'}), 500

    # Perform object recognition
    try:
        results = recognize_objects(file_path)
        if results is None:
            return jsonify({'error': 'AI model failed to process image'}), 500

        search_results = f'https://www.bing.com/images/search?q={results[0][1]}'

        return jsonify({
            'recognized_objects': [
                {'name': result[1], 'confidence': float(result[2])} 
                for result in results
            ],
            'search_results_url': search_results,
            'uploaded_image_path': file_path
        })
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 500


from flask import send_from_directory

# Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
