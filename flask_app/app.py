from flask import Flask, render_template, request, send_from_directory
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("waste_classifier.keras")
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    upload_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "uploads"
    )
    return send_from_directory(upload_folder, filename)


def preprocess_image(image_path):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, (128, 128))

    image = image / 255.0

    image = tf.expand_dims(image, axis=0)

    return image


@app.route("/predict", methods=["POST"])

def predict():

    image = request.files["image"]
    if image is None or image.filename == "":
        return "Please upload an image first."

    upload_folder = "flask_app/uploads"

    os.makedirs(upload_folder, exist_ok=True)

    image_path = os.path.join(upload_folder, image.filename)

    image.save(image_path)

    # Preprocess the uploaded image
    processed_image = preprocess_image(image_path)

    print("Image saved at:", image_path)
    print("Processed image shape:", processed_image.shape)
    print("Processed image dtype:", processed_image.dtype)

    # Make prediction
    predictions = model.predict(processed_image, verbose=0)

    # Get probabilities for this image
    probabilities = predictions[0]

    # Find the class with the highest probability
    predicted_index = np.argmax(probabilities)

    # Convert index to class name
    predicted_class = classes[predicted_index]

    # Get confidence
    confidence = probabilities[predicted_index]

    print("Predicted class:", predicted_class)
    print("Confidence:", confidence)

    return render_template(
    "index.html",
    predicted_class=predicted_class,
    confidence=confidence * 100,
    image_filename=os.path.basename(image_path)
    )

if __name__ == "__main__":
    app.run(debug=True)
