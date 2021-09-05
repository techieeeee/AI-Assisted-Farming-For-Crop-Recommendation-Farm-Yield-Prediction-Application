
from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
import random
import pandas as pd
#model = pickle.load(open('risk.pkl','rb'))


import requests

import json
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "Ls5s3x3bTi6ep25E2mudO-C2o4xPCOb5M_sJSIYOABzQ"
token_response = requests.post('https://iam.eu-gb.bluemix.net/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]
print("mltoken",mltoken)

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
#payload_scoring = {"input_data": [{"fields": [array_of_input_fields], "values": [array_of_values_to_be_scored, another_array_of_values_to_be_scored]}]}


model = pickle.load(open('yield_gujrat.pkl','rb'))
ohe=pickle.load(open('ohe.pkl','rb'))
sc=pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/iot')
def iot():
    return render_template("card2.html")
@app.route('/assesment')
def a():
    return render_template("index.html")

@app.route('/yield1')
def yield1():
    return render_template("base.html")

@app.route('/iot1', methods = ['POST'])
def iot1():
    print("inside admin")
    #q= request.form["demo"]
    # print(q)
    #a,b,c,d,e,f=1,2,3,4,5,6
    global t
    a=random.randint(0,140)
    b=random.randint(5,145)
    c=random.randint(5,205)
    d=round(random.uniform(8,44),2)
    e=round(random.uniform(14,100),2)
    f=round(random.uniform(3,10),2)
    g=round(random.uniform(20,299),2)
    t = [[int(a),int(b),int(c),d,e,f,g]]
    labels=['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','Ph','Rainfall']
    values=[a,b,c,d,e,f,g]
    print(values)
    return render_template("card2.html",a="Nitrogen = {}".format(a),b="Phosphorus = {}".format(b),
                           c="Potassium = {}".format(c),d="Temperature =  {}".format(d),
                           e="Humidity = {}".format(e),
                          f="Ph = {}".format(f),g="Rainfall = {}".format(g),labels=labels,values=values)
    
    #return render_template("circle.html",labels=labels,values=values)
@app.route('/crop',methods=['POST'])
def crop():
    payload_scoring = {"input_data": [{"fields":["N","P","K","temperature","humidity","ph","rainfall"],"values":t}]}
    response_scoring = requests.post('https://eu-gb.ml.cloud.ibm.com/ml/v4/deployments/49a70f4f-d984-4014-8345-da8ec8e2f69d/predictions?version=2021-09-01', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    predictions = response_scoring.json()
    print(predictions)
    pred = predictions['predictions'][0]['values'][0][0]
    print(pred)
    index=['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
    pred=index[pred]
    
    return render_template("crop.html",pred=pred)

@app.route('/risk', methods = ['POST'])
def yield2():
    area= request.form["area"]
    production= request.form["production"]
    district= request.form["District"]
    crop= request.form["crop"]
    l=[[area,production,district,crop]]
    d=pd.DataFrame(l,columns=['area','production','district','crop'])
    res=pd.DataFrame(ohe.transform(d[['district','crop']]).toarray())
    d=d.join(res)
    d.columns=['Area','Production','district','crop','Ahmedabad','Amreli','Banaskantha','Bharuch','Bhavnagar','Dang','Gandhinagar', 'Jamnagar', 'Junagadh', 'Kheda', 'Kutchh',
           'Mehsana', 'Panchmahal','Rajkot','Sabarkantha', 'Surat',
           'Surendranagar', 'Vadodara', 'Valsad','BAJRA','CASTOR','COTTON','GNUT','JOWAR','SESAMUM','WHEAT']
    d=d.drop(['district','crop'],axis=1)
    final=sc.transform(d)
    pred=model.predict(final)
    #y=[[int(x1),int(x2),int(x3),int(x4),int(x5),int(x6),int(x7),int(x8),int(r1),int(r2),int(r3),int(t1),int(t2),int(t3),int(t4),int(u1),int(u2),int(u3),int(q),int(s),int(v),int(w)]]        
    #a = model.predict(y)
    return render_template("predbad.html", z = pred[0])



if __name__ == '__main__':
    app.run(port=8000)
    
    
    
