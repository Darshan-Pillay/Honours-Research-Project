import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.post("/send")
def send_data():
    data = request.get_json()
    headers = { 'Content-Type': 'application/json' }

    try:
        response = requests.post(url="http://receiver:5000/receive", headers=headers, data=json.dumps(data), timeout=20)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500