from flask import Flask, request, jsonify, render_template
from src.main import pipeline

app = Flask(__name__)

# load the html page
@app.route('/')
def home():
    return render_template('index.html')

# api endpoint
@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    user_input = data['image']

    # take uploaded image and solve
    result = pipeline(user_input)

    return jsonify({"result": result})

if __name__ == '__main__':
    # make it accessible for cloud containers (like codespace)
    app.run(host='0.0.0.0', port=5000, debug=True)
