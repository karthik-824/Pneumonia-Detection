import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

# Initialize Flask app
app = Flask(__name__)


if not os.path.exists('uploads'):
    os.makedirs('uploads')

# ---------------------------------------------------------
# MODEL ARCHITECTURE 
# ---------------------------------------------------------
input_layer = Input(shape=(256, 256, 3))  
base_model = MobileNetV2(include_top=False, weights=None, input_tensor=input_layer)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)

# Create and load model
model = Model(inputs=input_layer, outputs=output)
model.load_weights("final_trained_model.h5")  
print("Model loaded successfully!")

def get_class_name(class_no):
    return "Normal" if class_no == 0 else "Pneumonia"

def get_result(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None  

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((256, 256))  
    image = np.array(image) / 255.0  
    input_img = np.expand_dims(image, axis=0)

    # Back to the standard, full-power predict method!
    result = model.predict(input_img)
    class_index = np.argmax(result, axis=1)[0]
    return class_index

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join('uploads', secure_filename(f.filename))
    f.save(file_path)

    value = get_result(file_path)
    if value is None:
        return jsonify({'error': 'Invalid image file'}), 400

    result = get_class_name(value)
    return jsonify({'result': result})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
