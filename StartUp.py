import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import streamlit as st 
import joblib
from sklearn.linear_model import LinearRegression

data= pd.read_csv('https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv')
data.to_csv('Startup.csv')

st.markdown("<h1 style = 'color:#0802A3; text-align: center; font-family: Arial Black'>STARTUP PROJECT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style = 'margin: -30px; color:#000000; text-align: center; font-family: cursive '>Built By Ismail Ibitoye</h4>", unsafe_allow_html=True)

st.image('pngwing.com (2).png', width=200, use_column_width=True)


st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<p style=font-family:Comic Sans>By analyzing a diverse set of parameters, including Market Expense, Administrative Expense, and Research and Development Spending, our team seeks to develop a robust predictive model that can offer valuable insights into the future financial performance of startups. This initiative does not only empowers investors and stakeholders to make data-driven decisions but also provides aspiring entrepreneurs with a comprehensive framework to evaluate the viability of their business models and refine their strategies for long-term success</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(data,use_container_width=True)

st.sidebar.image('pngwing.com (1).png', caption= 'Welcome User')
st.sidebar.write('Feature Input')
rd_spend= st.sidebar.number_input('Research and Development Expense',data['R&D Spend'].min(), data['R&D Spend'].max()+1000)
admin= st.sidebar.number_input('Administrative Expense',data['Administration'].min(), data['Administration'].max()+1000)
mkt_Spend= st.sidebar.number_input('Marketing Expense',data['Marketing Spend'].min(), data['Marketing Spend'].max()+1000)

st.write('Input Variables')
input_var= pd.DataFrame({'R&D Spend':[rd_spend], 'Administration':[admin], 'Marketing Spend': [mkt_Spend]})
st.dataframe(input_var)

model = joblib.load('startUPModel.pkl')
predicter=st.button('Predict Profit')

if predicter:
  prediction= model.predict(input_var)
  st.success(f"The Predicted Value for your company is{prediction}")
  st.balloons()