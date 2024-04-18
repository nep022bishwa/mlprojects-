#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:27:35 2024

@author: bishwaneupane
"""

import numpy as np
from flask import Flask,request,render_template
from sklearn.pipeline import Pipeline
import pickle

#first create an app object using Flask class

app=Flask(__name__)


#load the trained model

model=pickle.load(open('lr_model.pkl','rb'))

#Define the route to the home and 
#Decorator below links the relative route of the url to the function decorating it 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    ref_features=[float(x) for x in request.form.values()]
    features=[np.array(ref_features)]# convert the features in [[a,b]]
    prediction=model.predict(features)
    
    output_predict=round(prediction[0],2)
    
    return render_template('index.html',prediction_text='Total amount spent is {}'.format(output_predict))




if __name__=="__main__":
    app.run()