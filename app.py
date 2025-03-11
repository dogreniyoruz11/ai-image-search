import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf

# Enable Logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder='templates')

# Ensure "uploads" directory exists
os.makedirs("uploads", exist_ok=True)

# Load AI Model for Image Captioning
try:
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    model.compile()
    logging.info("✅ AI Model Loaded Successfully")
except Exception as e:
    logging.error(f"❌ Error Loading AI Model: {e}")
    model = None

# ✅ AI-Based Image Caption Generator
def generate_image_caption(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        return decoded_predictions[0][1]  # Best matching label
    except Exception as e:
        logging.error(f"❌ Error generating caption: {e}")
        return "Unknown Image"

# ✅ Home Route
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Upload & Generate Reverse Search Links
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename.replace(" ", "_")  # Fix space issue in URLs
    file_ext = filename.split('.')[-1].lower()

    # ✅ Supported File Types
    allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'heic', 'tiff'}
    if file_ext not in allowed_extensions:
        return jsonify({'error': f"Unsupported file type: {file_ext}"}), 400

    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    # ✅ Get the full image URL
    image_url = f"{request.host_url}uploads/{filename}"

    # ✅ Reverse Search Links (Fixed Bing Issue)
    search_links = {
        "Google Lens": f"https://lens.google.com/uploadbyurl?url={image_url}",
        "Yandex Reverse Search": f"https://yandex.com/images/search?source=collections&rpt=imageview&url={image_url}",
        "Bing Visual Search (Upload Manually)": "https://www.bing.com/visualsearch"  # Bing does NOT support direct URLs
    }

    return jsonify({
        'uploaded_image_url': image_url,
        'reverse_search_links': search_links
    })

# ✅ Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
