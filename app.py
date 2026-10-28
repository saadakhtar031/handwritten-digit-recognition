
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "https://doodle-digit-oracle.lovable.app",
                "https://doodle-digit-oracle.vercel.app"
            ]
        }
    }
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "handwritten_digit_model.tflite"
)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/", methods=["GET"])
def home():
    return "DigitMind AI TFLite API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        pixels = np.array(data["pixels"], dtype=np.float32)

        if pixels.shape != (28, 28):
            return jsonify({
                "error": "Expected a 28x28 pixel image"
            }), 400

        if pixels.max() > 1:
            pixels = pixels / 255.0

        image = pixels.reshape(1, 28, 28, 1)
        expected_dtype = input_details[0]["dtype"]
        image = image.astype(expected_dtype)

        interpreter.set_tensor(
            input_details[0]["index"], image
        )
        interpreter.invoke()

        probabilities = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])

        return jsonify({
            "digit": digit,
            "confidence": confidence,
            "probabilities": probabilities.tolist()
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
