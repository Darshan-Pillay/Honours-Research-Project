from flask import Flask, jsonify
from flask import request
from PIL import Image

app = Flask(__name__)

@app.post("/process_image")
def process_image():
    file = request.files['image']
    image = Image.open(file.stream)
    image.show()
    return jsonify({'msg': 'success', 'size': [image.width, image.height]})