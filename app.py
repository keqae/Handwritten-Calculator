from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.main import pipeline
import base64
import numpy as np
import cv2


app = Flask(__name__)
CORS(app)

# load the html page
@app.route('/')
def home():
    return render_template('index.html')

# api endpoint
@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        user_input = data['image']

        # remove prefix
        image_data = user_input.split(",")[1]

        image_bytes = base64.b64decode(image_data) # decode into bytes

        image_np = np.frombuffer(image_bytes, np.uint8) # convert into np.ndarray

        image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE) # decompress

        print(type(image))
        print(image.shape)

    
        # take uploaded image and solve
        result_dict = pipeline(image)

        # debug prints
        if result_dict is None:
            print("Result dictionary is None")

        print(f"Result: {result_dict}")

        return jsonify(result_dict)

    except Exception as e:
        print(f"Error in backend: {e}")


if __name__ == '__main__':
    # make it accessible for cloud containers (like codespace)
    app.run(host='0.0.0.0', port=5000, debug=True)
