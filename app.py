from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import base64
import cv2

# Load the trained model
model = tf.keras.models.load_model('face_shape_model.h5')

app = Flask(__name__)

# Function to decode the base64 image
def decode_image(base64_str):
    image_data = base64.b64decode(base64_str.split(",")[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    return img

# Predict face shape and hairstyle suggestion
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image = decode_image(data['image'])
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    face_shape_idx = np.argmax(prediction)
    face_shapes = ["heart", "long", "oval", "round", "square"]
    
    # Simple rule-based hairstyle suggestion (customize as needed)
    hairstyle_suggestion = {
        "heart": "Side Swept Bangs",
        "long": "Layered Waves",
        "oval": "Pixie Cut",
        "round": "Shaggy Bob",
        "square": "Soft Curls"
    }
    
    predicted_shape = face_shapes[face_shape_idx]
    suggested_hairstyle = hairstyle_suggestion[predicted_shape]
    print(suggested_hairstyle)

    return jsonify({
        'face_shape': predicted_shape,
        'hairstyle': suggested_hairstyle
    })

if __name__ == '__main__':
    app.run(debug=True)
