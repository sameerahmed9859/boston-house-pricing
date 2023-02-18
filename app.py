import pickle
import pandas as pd
import numpy as np
from flask import Flask,app,jsonify,url_for,render_template,request
from django.shortcuts import render

app= Flask(__name__)

# load the model 
lrmodel=pickle.load(open('lrmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=lrmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=lrmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="the House Price Prediction is {}".format(output))

if __name__==__name__:
    app.run(debug=True)