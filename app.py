
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "handwritten_digit_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "Handwritten Digit Recognition",
        "status": "API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        pixels = np.array(data["pixels"], dtype=np.float32)

        if pixels.shape != (28, 28):
            return jsonify({
                "error": "Input must be a 28x28 image"
            }), 400

        # Accept pixel values from 0 to 255 or 0 to 1
        if pixels.max() > 1:
            pixels = pixels / 255.0

        image = pixels.reshape(1, 28, 28, 1)

        probabilities = model.predict(image, verbose=0)[0]
        digit = int(np.argmax(probabilities))

        return jsonify({
            "prediction": digit,
            "confidence": round(
                float(probabilities[digit]) * 100, 2
            )
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
