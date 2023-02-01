# Loading required libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import VotingClassifier
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
    
# Writing App title 
st.title('Bank Customer Churn Prediction')

        
# Setting App description
st.markdown("""
     :dart:  This web app is made to predict customer churn in a ficitional Bank use case.
    The application is functional for online prediction. \n
    """)
st.markdown("<h3></h3>", unsafe_allow_html=True)
st.info("Let's do prediction...")
st.subheader("Customer Data")
 
#Making sliders and feature variables
creditscore = st.number_input('Enter credit score :', min_value=200, max_value=1000, value=350)
geography = st.selectbox('Select location :', ('Germany', 'France','Spain'))       
gender = st.selectbox('Select Gender :', ('Male', 'Female'))
age = st.number_input('Enter age :', min_value=18, max_value=100, value=20)
tenure = st.number_input('Enter tenure :', min_value=0, max_value=10, value=3)
balance = st.number_input(label = 'Enter account balance :',step = 1.,format="%.2f")   
products = st.number_input('Enter no. of products :', min_value=1, max_value=4, value=2)
hascard = st.selectbox('Select if customer has a credit card or not :', ('Yes', 'No'))
isactive =  st.selectbox('Select whether the customer is an active member of the bank or not :', ('Yes', 'No')) 
salary = st.number_input(label = 'Enter salary :',step = 1.,format="%.2f")  

       

#Converting features into DataFrame                        
features = {
  'CreditScore':creditscore,
  'Geography':geography,
  'Gender':gender,
  'Age':age,
  'Tenure':tenure,
  'Balance':balance,
  'NumOfProducts':products,
  'HasCrCard':hascard,
  'IsActiveMember':isactive,
  'EstimatedSalary':salary,
  
  }
st.write('Overview of input is shown below :')  
features_df  = pd.DataFrame([features])
st.table(features_df)   


# Defining preprocessing function

def preprocess(data):
    if(data['HasCrCard'][0] == 'Yes'):
        data['HasCrCard'][0]= 1
    else:
        data['HasCrCard'][0]= 0
    
    if(data['IsActiveMember'][0] == 'Yes'):
        data['IsActiveMember'][0]= 1
    else:
        data['IsActiveMember'][0]= 0
        
    geo = []
    if(data['Geography'][0] == "France"):
        geo = [0.0,1.0,0.0]
    elif(data['Geography'][0] == "Spain"):
        geo = [0.0,0.0,1.0]
    elif(data['Geography'][0] == "Germany"):
        geo = [1.0,0.0,0.0]
    
    gender = []
    if(data['Gender'][0] == 'Female'):
        gender = [0.0,1.0]
    elif(data['Gender'][0] == 'Male'):
        gender = [1.0,0.0]
        
    
    onhot_data = geo + gender
    onhot_data = pd.DataFrame(onhot_data).T
    onhot_data.columns = ['Geography_Germany','Geography_France','Geography_Spain','Gender_Male','Gender_Female']
    data = data.join(onhot_data)
    
    cat_df = data[["Geography_Germany", "Geography_France","Geography_Spain",   "Gender_Male","Gender_Female" ,"HasCrCard","IsActiveMember"]]
    
    X= data.drop(labels=["Geography_France","Geography_Germany","Geography_Spain",   "Gender_Female","Gender_Male" ,"HasCrCard","IsActiveMember","Geography","Gender"],axis=1)
    cols = X.columns
    index = X.index
    X = transformers.transform(X)
    X = pd.DataFrame(X, columns = cols, index = index)
    X = pd.concat([X,cat_df], axis = 1)
    
    prediction_val = []
    for i in range(len(X.columns)):
        prediction_val.append(X.iloc[0][i])
    prediction_val = [prediction_val]
    return prediction_val

# Loading scaler and ensemble model
transformers = pickle.load(open("transformer.pickle","rb"))
model = pickle.load(open("model.pickle","rb"))



# Prediction
y=preprocess(features_df)
pred=model.predict(y)
prediction=pred[0]
st.write('Will customer leave the bank?') 


if st.button('Predict'):
    if prediction == 1:
        st.warning('Yes, the customer will leave the bank soon.')
    elif prediction == 0:
        st.success('No, the customer is happy with bank services.') 
 
 
 










                                   
