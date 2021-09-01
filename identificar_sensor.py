from __future__ import division, print_function

# Importar Keras
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

# Importar Flask 
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Importar librerias adicionales 
import os
import numpy as np
import cv2

# Definir el tamaño de las imágenes
width_shape = 224
height_shape = 224

# Definir las clases a clasificar
names = ['KY002','KY006','KY010','KY011','KY021','KY035','KY039']

# Instanciar Flask
app = Flask(__name__)

# Ruta del modelo entrenado
MODEL_PATH = 'modelos\sensores_VGG16.h5'

# Cargar el modelo
model = load_model(MODEL_PATH)

print('El modelo se cargó correctamente. Ingresar a http://127.0.0.1:5000/')

# Predicción de la imagen
def model_predict(img_path, model):

    img=cv2.resize(cv2.imread(img_path), (width_shape, height_shape), interpolation = cv2.INTER_AREA)
    x=np.asarray(img)
    x=preprocess_input(x)
    x = np.expand_dims(x,axis=0)
    
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Página principal
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Obtener el archivo del request
        f = request.files['file']

        # Grabar el archivo en ./subidas
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'subidas', secure_filename(f.filename))
        f.save(file_path)

        # Se realiza la predicción
        preds = model_predict(file_path, model)

        print('PREDICCIÓN', names[np.argmax(preds)])
        
        # Enciar el resultado de la predicción
        result = str(names[np.argmax(preds)])              
        return result
    return None


if __name__ == '__main__':
    app.run(debug=False, threaded=False)

