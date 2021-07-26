# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 21:26:03 2021

@author: Abdul Wajid
"""

#import
import uvicorn
import argparse
from fastapi import FastAPI
import pickle
import preprocess_kgptalkie as ps
import re
import spacy
from fastapi.middleware.cors import CORSMiddleware
nlp = spacy.load("en_core_web_sm")  

app = FastAPI()


#get model
model = pickle.load(open('model.pkl', 'rb'))
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

#First API
@app.get('/')
def index():
    return{'message':'Check'}

#Main API
@app.get('/predict')
def predict_rating(input: str):

    def get_clean(x):
        x = str(x).lower().replace('\\', '').replace('_', ' ')
        x = ps.cont_exp(x)
        x = ps.remove_emails(x)
        x = ps.remove_urls(x)
        x = ps.remove_html_tags(x)
        x = ps.remove_accented_chars(x)
        x = ps.remove_special_chars(x)
        x = re.sub("(.)\\1{2,}", "\\1", x)
        return x

    cleanInput = get_clean(input) 
    prediction = model.predict(loaded_vectorizer.transform([cleanInput]))
    
    #Edit this as you like
    if(prediction == [5]):
        prediction = "5"
    elif(prediction == [4]):
        prediction= "4"
    elif(prediction == [3]):
        prediction = "3"
    elif(prediction == [2]):
        prediction = "2"
    else:
        prediction = "1"
    return{
        'prediction':prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
