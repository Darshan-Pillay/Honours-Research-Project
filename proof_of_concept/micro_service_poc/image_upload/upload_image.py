from flask import Flask, jsonify
import requests

app = Flask(__name__)

@app.get("/upload_image")
def upload_image():
    image_to_upload = open('example_image_1.JPG', 'rb')
    files = {'image': image_to_upload}

    try:
        response = requests.post("http://localhost:5000/process_image", files=files)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500