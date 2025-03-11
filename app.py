import os
import logging
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Enable Logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder='templates')

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/enhanced", exist_ok=True)

# Load AI Model for Image Enhancement & Captioning
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

# ✅ AI-Based Image Enhancement (Upscaling)
def enhance_image(image_path):
    try:
        image = cv2.imread(image_path)
        image_upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        enhanced_path = image_path.replace("uploads", "uploads/enhanced")
        cv2.imwrite(enhanced_path, image_upscaled)
        return enhanced_path
    except Exception as e:
        logging.error(f"❌ Error enhancing image: {e}")
        return image_path

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

    # ✅ AI-Based Enhancements
    enhanced_image_path = enhance_image(file_path)
    image_caption = generate_image_caption(file_path)

    # ✅ Get the full image URL
    image_url = f"{request.host_url}uploads/{filename}"
    enhanced_image_url = f"{request.host_url}uploads/enhanced/{filename}"

    # ✅ Reverse Search Links (Fixed Bing & TinEye URL Format)
    search_links = {
        "🔍 Google Lens": f"https://lens.google.com/uploadbyurl?url={image_url}",
        "🔍 Bing Visual Search": f"https://www.bing.com/images/search?q=imgurl:{image_url}&view=detailv2",
        "🔍 Yandex Reverse Search": f"https://yandex.com/images/search?source=collections&rpt=imageview&url={image_url}",
        "🔍 Pinterest Image Search": f"https://www.pinterest.com/search/pins/?q={image_url}",
        "🔍 TinEye Reverse Image": f"https://www.tineye.com/search/?url={image_url}",
        "🔍 Reddit Image Search": f"https://www.reddit.com/search?q={image_url}",
        "🔍 Etsy Reverse Search": f"https://www.etsy.com/search?q={image_url}"
    }

    # ✅ Save Recent Searches (Only Keep Last 5 Images)
    recent_searches = []
    if os.path.exists("recent_searches.json"):
        with open("recent_searches.json", "r") as f:
            try:
                recent_searches = json.load(f)
            except:
                recent_searches = []

    recent_searches.append({"image_url": image_url, "caption": image_caption})
    recent_searches = recent_searches[-5:]  # Keep only last 5 searches

    with open("recent_searches.json", "w") as f:
        json.dump(recent_searches, f)

    return jsonify({
        'reverse_search_links': search_links,
        'uploaded_image_url': image_url,
        'enhanced_image_url': enhanced_image_url,
        'image_caption': image_caption,
        'recent_searches': recent_searches
    })

# ✅ Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
