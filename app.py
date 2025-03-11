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

# Load AI Model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess image for AI model
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to perform AI-based object recognition
def recognize_objects(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    return decoded_predictions

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
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)  # Ensure directory exists
    file.save(file_path)
    logging.info(f"File saved at {file_path}")

    # Perform object recognition
    try:
        results = recognize_objects(file_path)
        search_results = f'https://www.bing.com/images/search?q={results[0][1]}'

        return jsonify({
            'recognized_objects': [
                {'name': result[1], 'confidence': float(result[2])} 
                for result in results
            ],
            'search_results_url': search_results
        })
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
