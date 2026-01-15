import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image,model,class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image = np.array(image)
    
    # expand dimensions to (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    
    # normalize the image
    image = image / 255.0
    
    # make prediction
    prediction = model.predict(image)
    
    # get the index of the class with the highest probability
    class_index = np.argmax(prediction[0])
    
    # get the class name
    class_name = class_names[class_index]
    
    # get the confidence score
    confidence_score = prediction[0][class_index]
    
    return class_name, confidence_score