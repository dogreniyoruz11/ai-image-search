import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
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
    logging.info("✅ AI Model Loaded Successfully")
except Exception as e:
    logging.error(f"❌ Error Loading AI Model: {e}")
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
        logging.error(f"❌ Error Preprocessing Image: {e}")
        return None


# Function to perform AI-based object recognition
def recognize_objects(image_path):
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        logging.error("❌ Failed to preprocess image")
        return None

    if model is None:
        logging.error("❌ AI Model is not loaded")
        return None

    try:
        predictions = model.predict(processed_image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        logging.error(f"❌ Error Processing Image in AI Model: {e}")
        return None


# Home route
@app.route('/')
def index():
    return render_template('index.html')


# ✅ Fixed Upload API
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)  # ✅ Ensure folder exists
    file.save(file_path)
    
    # Perform object recognition
    results = recognize_objects(file_path)
    
    if not results:
        return jsonify({'error': 'AI model failed to process image'}), 500

    query = results[0][1].replace(" ", "+")

    # ✅ Fix: Add multiple search engines
    search_links = {
        "Google": f"https://www.google.com/search?tbm=isch&q={query}",
        "Bing": f"https://www.bing.com/images/search?q={query}",
        "Yandex": f"https://yandex.com/images/search?text={query}",
        "Pinterest": f"https://www.pinterest.com/search/pins/?q={query}"
    }

    return jsonify({
        'recognized_objects': [
            {'name': result[1], 'confidence': float(result[2])}
            for result in results
        ],
        'search_results': search_links,
        'uploaded_image_url': request.host_url + 'uploads/' + file.filename  # ✅ Ensures image preview works
    })


# ✅ Serve Uploaded Images (Fix for Image Preview Issue)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
