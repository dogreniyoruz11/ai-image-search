import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory

# Enable Logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder='templates')

# Ensure "uploads" directory exists
os.makedirs("uploads", exist_ok=True)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and Generate Reverse Search Links
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    # Generate Reverse Image Search Links
    search_links = {
        "🔍 Google Search": f"https://www.google.com/searchbyimage?image_url={request.host_url}uploads/{filename}",
        "🔍 Bing Search": f"https://www.bing.com/images/search?q=imgurl:{request.host_url}uploads/{filename}&view=detailv2",
        "🔍 Yandex Search": f"https://yandex.com/images/search?rpt=imageview&url={request.host_url}uploads/{filename}",
        "🔍 Pinterest Search": f"https://www.pinterest.com/search/pins/?q={request.host_url}uploads/{filename}"
    }

    return jsonify({
        'reverse_search_links': search_links,
        'uploaded_image_url': f"{request.host_url}uploads/{filename}"
    })

# Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
