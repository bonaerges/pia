#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
import base64
import random

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename, send_from_directory
from PIL import Image
import os, io
import numpy as np
import tensorflow as tf


app = Flask(__name__)

app.config['DATASET_FOLDER'] = 'dataset/train/'

# Path to signature.json and model file
ASSETS_PATH = os.path.join(".", "yoga_model")
model = tf.keras.models.load_model(ASSETS_PATH)


@app.route('/', methods=['GET'])
def index():
    # Renderiza la página de carga
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Preparar la imagen
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((200, 200))
        image = np.expand_dims(image, axis=0)
        image = np.array(image) / 255.0

        # Predecir
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions)

        # Obtener el nombre de la clase
        class_names = ["downdog", "goddess", "plank", "tree", "warrior2"]
        predicted_label = class_names[predicted_class[0]]

    # Seleccionar una imagen de ejemplo de la clase predicha y convertirla en base64
        current_directory = os.getcwd()
    # Construir el full path

        example_folder = os.path.join(app.config['DATASET_FOLDER'], predicted_label)
        # Asegurar que la lista de archivos no esté vacía
        if os.listdir(example_folder):
            random_filename = random.choice(os.listdir(example_folder))
            # Uso de secure_filename para garantizar un nombre de archivo seguro
            secure_random_filename = secure_filename(random_filename)
            example_image_path = os.path.join(current_directory+"/"+example_folder, secure_random_filename)
            print(f"example_image_path: {example_image_path}")
            with open(example_image_path, "rb") as image_file:
                encoded_example_image = base64.b64encode(image_file.read()).decode('utf-8')


        else:
            # Manejar el caso en que el directorio esté vacío
            print("No hay archivos en el directorio.")

        # Convertir la imagen subida en base64
        file.seek(0)
        encoded_uploaded_image = base64.b64encode(file.read()).decode('utf-8')

        return render_template('result.html',
                               uploaded_image=encoded_uploaded_image,
                               class_example_image=encoded_example_image,
                               class_name=predicted_label, confidence=f'{confidence:.2f}')

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)