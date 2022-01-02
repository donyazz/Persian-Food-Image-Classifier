#!/usr/bin/env python
# coding: utf-8

# In[10]:


from fastai.vision.widgets import *
from fastai.vision.all import *

import pathlib

import streamlit as st
import os

print('OS: ' + os.name)

if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Persian Food Classifier")
    
st.title("Persian Food Classifier")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.predict()

    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(128, 128), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            self.predict()
        else:
            st.write(f'Click the button to classify')

    def predict(self):
        pred, pred_idx, probs = self.learn_inference.predict(self.img)
        st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')


if __name__ == '__main__':
    file_name = 'export.pkl'

    predictor = Predict(file_name)

# In[ ]:




