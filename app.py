from __future__ import division, print_function
# coding=utf-8
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import random 
from tqdm import tqdm 
IMG_SIZE=200

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = r'DenseNet_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary



def pred_list(pred):
  pred_list=[]
  for i in pred[0]:
    x=format(i,'.8f')
    pred_list.append(x)
  return pred_list


def disease(pred_list):
  x=np.argmax(pred_list)
  if x==0:
    print("Disease: Covid")
    output='Covid'
  elif x==1:
    print("Disease: Pneumonia")
    output='Pneumonia'
  elif x==2:
    print("Disease: Turberculosis")
    output='Turberculosis'
  elif x==3:
    print("Normal")
    output='Normal'
  return output
def predict(img_path,model):
    #img = image.load_img(img_path, target_size=(200,200))
    img=cv2.imread(img_path,cv2.IMREAD_COLOR)
    img=cv2.resize(img,(32,32))
    x=img
    img=np.expand_dims(img,axis=0)

    pred=model.predict(img)
    pred_list1=pred_list(pred)
    output=disease(pred_list1)
    
    return output
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        output = predict(file_path, model)

        return output
    return None



if __name__ == '__main__':
    app.run(debug=True)