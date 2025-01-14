from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask_cors import CORS
from io import BytesIO
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('art_validation_model.h5')

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}}) 

SIZE = 512

def predict_image(image_bytes):
    """
    Function to predict if the image is handmade or factory-made.
    """
    img = Image.open(BytesIO(image_bytes))
    img = img.resize((SIZE, SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    score = prediction[0][0]

    if score > 0.8:
        result = "Handmade Art "
    elif 0.51 <= score <= 0.8:
        result = "Likely to be handmade, Our experts will verify it"
    else:
        result = "Factory-made Art"

    return { "result": result}


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to handle image upload and prediction.
    """
    # Check if an image file is part of the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']

    # Check if the image has a filename
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check file extension to ensure valid image type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' in image.filename and image.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Please upload a PNG, JPG, or JPEG image."}), 400

    try:
        # Read the image as bytes
        image_bytes = image.read()

        # Call the prediction function
        prediction = predict_image(image_bytes)
        return jsonify(prediction)

    except Exception as e:
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
