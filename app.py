import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory

# Enable Logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder='templates')

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)

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
    filename = file.filename.replace(" ", "_")
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    image_url = f"{request.host_url}uploads/{filename}"

    # ✅ Reverse Search Links
    search_links = {
        "Google Lens": f"https://lens.google.com/uploadbyurl?url={image_url}",
        "Yandex Reverse Search": f"https://yandex.com/images/search?source=collections&rpt=imageview&url={image_url}",
        "Bing Search": f"https://www.bing.com/images/search?q=imgurl:{image_url}&view=detailv2",
        "Pinterest Image Search": f"https://www.pinterest.com/search/pins/?q={image_url}"
    }

    return jsonify({
        'message': "✅ Image Uploaded Successfully!",
        'reverse_search_links': search_links,
        'uploaded_image_url': image_url
    })

# ✅ Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
