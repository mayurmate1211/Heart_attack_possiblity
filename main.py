from distutils.log import debug
from re import I
from flask import Flask,jsonify,request 
app=Flask(__name__)
import pandas as pd 
import numpy as np
import pickle
logistic_model=pickle.load(open('heart_logistic.pkl',"rb"))

@app.route('/')
def main():
    return jsonify({"messeage":"welcome to my applicaation"})

@app.route('/heart_data')
def target_prediction():
    data=request.get_json()
    df=pd.DataFrame(data)

    prediction=logistic_model.predict(df)
    print(prediction)
    
    l=[]
    for i in prediction:
        l.append(int(i))

    return jsonify({"Prediction of target variable is :",l})



if __name__== "__main__":
    app.run(debug=True)


