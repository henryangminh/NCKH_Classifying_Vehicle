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
from keras.models import Model
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, save_img
from annoy import AnnoyIndex
#from my_model import *
from ultility.config import *
from ultility.prepare_data import *
from ultility.ultility import *

APP_NAME = "flask_app"
UPLOAD_FOLDER = './static/img/upload_images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

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

@app.route('/prediction')
def prediction():
    page = (int)(request.args.get('page'))
    if page is None:
        page=1
    if page > 1:
        pass
    else:
        if current_img is None:
            return ""
        img = cv2.imread(current_img)
        feats = crop_feature(img, verbose=False)
        max_feat = feats[np.argmax([feats[i][0] for i in range(len(feats))])] # Get feature with the highest prob
        vehicle_query_result = vehicle_query.get_nns_by_vector(max_feat[1], n=500)
        result_imgs = featureDB.iloc[vehicle_query_result]['name']
    return paging(page)
@app.route("/upload", methods=['POST'])
def upload():
    fileUpload = request.files['files']

    SAVE_PATH = "./static/img/upload_images"

##    #read image file string data
    filestr = fileUpload.read()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
##    img = cv2.imread(fileUpload)
##    img = convertRGBtoBGR(img)
##    img = cv2.resize(img,(resized_shape[1], resized_shape[0]))

##    predict_img = model.predict(img/255)

    name = (int)(datetime.datetime.utcnow().timestamp())
    
    # Create folder if not existed
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    result = f"{SAVE_PATH}/{name}.png"
    
    #save_img(result,predict_img)
    cv2.imwrite(result,img)
    return jsonify(search_vehicles(img,euclidean))

if __name__ == '__main__':
    app.run(debug=True)
