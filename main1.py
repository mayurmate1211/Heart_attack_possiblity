from distutils.log import debug
from turtle import update
from flask import Flask,jsonify,request 
app=Flask(__name__)
import pandas as pd
import numpy as np
import pickle 

logistic_model=pickle.load(open('heart_logistic.pkl','rb'))

@app.route('/')
def my():
    return jsonify({'Messeage':"Welcome in my Application"})
@app.route("/prediction_heart")

def heart():
    data=request.get_json()
    df=pd.DataFrame(data)
    result=logistic_model.predict(df)
    print(result)
    dict1={}

    for i,j in enumerate(result):
        if j==1:
            dict1.update({"Prediction":"There is more chance of heart attack"})
        else:
            dict1.update({"Prediction":"There is less chance of heart attack"})
            
    return dict1


if __name__=="__main__":
    app.run(debug=True)