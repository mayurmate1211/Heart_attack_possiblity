import pandas as pd
import numpy as np
from flask import Flask,render_template,request
import pickle
app=Flask(__name__)

model=pickle.load(open('heart_logistic.pkl','rb'))

@app.route('/')
def check():
    return render_template('index.html')

@app.route('/predict',methods=["POST","GET"])  

def heart():

    var_age=int(request.form.get('age'))
    var_sex=int(request.form.get('sex'))
    var_cp=int(request.form.get('cp'))
    var_trestbps=int(request.form.get('trestbps'))
    var_chol=int(request.form.get('chol'))
    var_fbs=int(request.form.get('fbs'))
    var_restecg=int(request.form.get('restecg'))
    var_thalach=int(request.form.get('thalach'))
    var_exang=int(request.form.get('exang'))
    var_oldpeak=float(request.form.get('oldpeak'))
    var_slope=int(request.form.get('slope'))
    var_ca=int(request.form.get('ca'))
    var_thal=int(request.form.get('thal'))

    result=model.predict([[var_age,var_sex,var_cp,var_trestbps,var_chol,var_fbs,var_restecg,var_thalach,var_exang,var_oldpeak,var_slope,var_ca,var_thal]])
    print(result[0])


    if result[0]==1:
        return "The person has heart attack possiblity"
    else:
        return "The person does not have heart attack possiblity"











if __name__== "__main__":
    app.run(debug=True ,host="0.0.0.0",port=8080)