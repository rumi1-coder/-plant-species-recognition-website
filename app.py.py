from flask import Flask, request, jsonify
from flower_train import get_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = get_model()

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.resize((224, 224))
    img_arr = np.array(img)
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    pred = model.predict(img_arr)
    flower_type = categories[np.argmax(pred)]
    return jsonify({'flower_type': flower_type})

if __name__ == '__main__':
    app.run(debug=True)
