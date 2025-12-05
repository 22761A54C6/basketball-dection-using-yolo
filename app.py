from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import uuid
import os
import numpy as np

app = Flask(__name__)

# Load YOLO ONNX model (same as Colab)
# model = YOLO("model_after_40_epochs.onnx")
model = YOLO("model_after_40_epochs.onnx", task="detect")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Convert uploaded image to OpenCV format
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Run YOLO prediction (same style as Colab)
    results = model(img)

    # Plot result (YOLO built-in)
    output_img = results[0].plot()

    # Save output image
    output_name = f"result_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join("static", output_name)
    cv2.imwrite(output_path, output_img)

    return render_template("index.html", output_image=output_name)


if __name__ == "__main__":
    app.run(debug=True)
