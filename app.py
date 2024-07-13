import streamlit as st
import joblib
import pandas as pd

model = joblib.load("titanic_model.pkl")

st.title('Titanic Model Frontend')

pcls = st.select_slider('Choose passenger class',[1,2,3])
age = st.slider('Input Age',0,100)
sib = st.slider('Input Siblings',0,10)
parch = st.slider('Input parents/children',0,2)

fare = st.number_input('Fare amount',0,100)


def predict_survivers():

    column_names = ['Pclass','Age','Parch','Fare','SibSp']
    row = [pcls,age,parch,fare,sib]
    X = pd.DataFrame([row],columns = column_names)
    y_pred = model.predict(X)
    print(y_pred)
    if y_pred ==1:
        st.success('Passenger Survived')
    else:
        st.success('Passenger didnot Survived')


st.button('Predict', on_click = predict_survivers)