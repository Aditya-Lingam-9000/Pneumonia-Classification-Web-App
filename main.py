# imporing required libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from util import classify, set_background

# for this project i used requirements as 
'''
python-3.11
tensorflow-2.15
keras-2.15
Protobuf 4.23.4
'''

# set title
st.title('Pneumonia Classification Web App')
# set subtitle
st.markdown('Upload an image of a chest X-ray to classify whether it shows pneumonia or not.')
# set background
base_path = os.path.dirname(__file__)
bg_path = os.path.join(base_path, 'img_bg.png')
if os.path.exists(bg_path):
    st.markdown(set_background(bg_path), unsafe_allow_html=True)

# file uploading
file = st.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])

# Custom layer to handle the 'groups' argument error in Keras 3
# This occurs when loading models saved in older Keras versions

# loading the model
model_path = os.path.join(base_path, 'pneumonia_classifier.h5')
# We use compile=False because the model might have been saved in a different Keras version
# and the compilation state might not be fully compatible.
model = load_model(model_path)

# loading class_names
labels_path = os.path.join(base_path, 'labels.txt')
with open(labels_path, 'r') as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]

# processing and displaying the image

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image,use_column_width=True)

    # classify image
    class_name,conf_score = classify(image,model,class_names)
    st.write('Class:',class_name)
    st.write(f'Confidence Score:{(conf_score*100):.2f}')


# dont run directly 
# instead run below in terminal
# streamlit run d:\CV\CV_Projects\pneumonia_classification_web_app\main.py

