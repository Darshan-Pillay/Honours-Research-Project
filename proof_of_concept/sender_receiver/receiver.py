from flask import Flask, jsonify
from flask import request

app = Flask(__name__)

@app.post("/receive")
def received_data():
    json = request.get_json()
    response = jsonify(json)
    response.status = 200
    return response