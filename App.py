import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('iris_model.pkl', 'rb'))
iris = load_iris()
def classify(num):
    if num<0.5:
        return 'Setosa'
    elif num <1.5:
        return 'Versicolor'
    else:
        return 'Virginica'

def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Spark Foundation</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Iris</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Flower Name</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Iris Classification ")
    activities=['Predict']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    sl=st.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.slider('Select Petal Length', 0.0, 10.0)
    pw=st.slider('Select Petal Width', 0.0, 10.0)
    inputs=[[sl,sw,pl,pw]]

    if st.button("Classify"):
        if option=='Predict':
            st.success(classify(model.predict(inputs)))

    if st.button("About"):
      st.header("By Prakhar Mehrishi")
      st.subheader("Intern , The Spark Foundation")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Iris Classifier</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
