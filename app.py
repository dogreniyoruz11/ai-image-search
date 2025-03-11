import os
import requests
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__, template_folder='templates')

# Load pre-trained AI model for image recognition (TensorFlow + OpenCV)
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

# Home route to render UI
@app.route('/')
def index():
    return render_template('index.html')

# API Route for Image Search
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Perform object recognition
    results = recognize_objects(file_path)
    
    # Search image on Bing (Sample implementation)
    search_results = f'https://www.bing.com/images/search?q={results[0][1]}'
    
    return jsonify({
        'recognized_objects': [{
            'name': result[1],
            'confidence': float(result[2])
        } for result in results],
        'search_results_url': search_results
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
