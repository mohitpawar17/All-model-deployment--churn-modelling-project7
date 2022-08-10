import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)


#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    credit = float(request.args.get('credit'))
    age=float(request.args.get('age'))
    tenure=float(request.args.get('tenure'))
    balance=float(request.args.get('balance'))
    products=float(request.args.get('products'))
    estimatedsalary=float(request.args.get('estimatedsalary'))
    geography=float(request.args.get('geography'))
    gender=float(request.args.get('gender'))
    active=float(request.args.get('active'))
    creditcard=float(request.args.get('creditcard'))
    model1=float(request.args.get('model1'))

    if model1==0:
      model=pickle.load(open('project7_decision_model.pkl','rb'))
    elif model1==1:
      model=pickle.load(open('project7_svm.pkl','rb'))
    elif model1==2:
      model=pickle.load(open('project7_random_forest.pkl','rb'))
    elif model1==3:
      model=pickle.load(open('project7_knn.pkl','rb'))
    elif model1==4:
      model=pickle.load(open('project7_naive.pkl','rb'))
      

    dataset= pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:13].values
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,1] = labelencoder_X.fit_transform(X[:,1])
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,2] = labelencoder_X.fit_transform(X[:,2])
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[credit,age,tenure,balance,products,estimatedsalary,geography,gender,active,creditcard]]))
    if prediction==0:
      message="Customer has Exited"
    else:
      message="Customer has Not exited"
    
        
    return render_template('index.html', prediction_text='Model  has predicted : {}'.format(message))

if __name__=='main':
  app.run(debug=True)
