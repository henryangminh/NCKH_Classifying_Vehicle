from flask import Flask, jsonify, session, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
import os, shutil
import cv2
import numpy as np
import pandas as pd
import datetime, time
import pickle
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, save_img
from sklearn.svm import SVC
from sklearn.externals import joblib

APP_NAME = "flask_app"
UPLOAD_FOLDER = './static/img/upload_images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Change to GPU
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class_mapping = ['Bus', 'Microbus', 'Minivan', 'SUV', 'Sedan', 'Truck']
AlexNet_model = None
GoogleNet_model = None
svm_model = None


class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "flask_app"
        return super().find_class(module, name)

sess = tf.Session()
set_session(sess)
with open("model/AlexNet.json", "r") as json_file:
    model_json = json_file.read()
    AlexNet_model = model_from_json(model_json)
AlexNet_model.load_weights("model/AlexNet_weight.h5")
AlexNet_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

GoogleNet_model = load_model('model/GoogleNet.hdf5')
model_extract_feature = Model(GoogleNet_model.input, GoogleNet_model.get_layer(GoogleNet_model.layers[-2].name).output)
svm_model = joblib.load('model/SVMmodel.pkl')


global graph
graph = tf.get_default_graph()
def convertRGBtoBGR(img_arr):
    ''' OpenCV reads images in BGR format whereas in keras,
    it is represented in RGB.
    This function convert img from BGR to RGB and vice versa
    '''
    return img_arr[...,::-1]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask("flask_app", static_folder='static')
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    result_imgs = []
    current_img = None
    return app.send_static_file('html/index.html')

@app.route('/path')
def path():
    return os.getcwd()

@app.route("/GoogLeNet", methods=['POST'])
def GoogLeNet():
    fileUpload = request.files['files']
    resized_shape = (224, 224)
    SAVE_PATH = app.config['UPLOAD_FOLDER']
    
    # Create folder if not existed
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    name = (int)(datetime.datetime.utcnow().timestamp())
    saved_img = f"{SAVE_PATH}/{name}.png"
    
    # if user does not select file, browser also
    # submit an empty part without filename
    if fileUpload.filename == '':
        return "No selected file"
    if fileUpload and allowed_file(fileUpload.filename):
        fileUpload.save(saved_img)

    img = load_img(saved_img)
    img = img.resize(resized_shape)
    img = img_to_array(img)
    with graph.as_default():
      set_session(sess)
      feature = model_extract_feature.predict(np.expand_dims(img,axis=0))
      result_return = svm_model.predict_proba(feature)
    max_index = np.argmax(result_return[0])
    #save_img(result,predict_img)
    #cv2.imwrite(result,img)
    return jsonify({'Class': class_mapping[max_index], 'Prob': float(result_return[0][max_index]*100)})

@app.route("/AlexNet", methods=['POST'])
def AlexNet():
    fileUpload = request.files['files']
    resized_shape = (64, 64)
    SAVE_PATH = app.config['UPLOAD_FOLDER']
    
    # Create folder if not existed
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    name = (int)(datetime.datetime.utcnow().timestamp())
    saved_img = f"{SAVE_PATH}/{name}.png"
    
    # if user does not select file, browser also
    # submit an empty part without filename
    if fileUpload.filename == '':
        return "No selected file"
    if fileUpload and allowed_file(fileUpload.filename):
        fileUpload.save(saved_img)

    img = load_img(saved_img)
    img = img.resize(resized_shape)
    img = img_to_array(img)


    #save_img(result,predict_img)
    #cv2.imwrite(result,img)
    with graph.as_default():
      set_session(sess)
      res = AlexNet_model.predict(np.expand_dims(img,axis=0))
    clss = np.argmax(res[0])
    return jsonify({'Class': class_mapping[clss], 'Prob': float(res[0][clss]*100)})

if __name__ == '__main__':
    app.run(debug=True)
