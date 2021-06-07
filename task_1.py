# -*- coding: utf-8 -*-
"""Task 1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_-Hdf-2M_MtDlnT4FlcGc69C0qcEJBnV
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

df['target'] = iris.target
df.head()

iris.target_names

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df.head()

from sklearn.model_selection import train_test_split

X = df.drop(['target','flower_name'], axis='columns')
X.head()

y = df.target
y.head()

import matplotlib.pyplot as plt

a = iris.data
b = iris.target

plt.plot(a[:, 0][b==0], a[:, 1][b==0], 'r.', label='setosa')
plt.plot(a[:, 0][b==1], a[:, 1][b==1], 'g.', label='Versicolour')
plt.plot(a[:, 0][b==2], a[:, 1][b==2], 'b.', label='Virginica')
plt.legend()
plt.show()

plt.plot(a[:, 0][b==0]* a[:, 1][b==0], a[:, 1][b==0]* a[:, 2][b==0], 'r.', label='setosa')
plt.plot(a[:, 0][b==1]* a[:, 1][b==1], a[:, 1][b==1]* a[:, 2][b==1], 'g.', label='Versicolour')
plt.plot(a[:, 0][b==2]* a[:, 1][b==2], a[:, 1][b==2]* a[:, 2][b==2], 'b.', label='Virginica')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_test,y_test)

model.score(X_test,y_test)

y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

!pip install pydotplus
!apt-get install graphviz -y

from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Visualize the graph
dot_data = StringIO()
export_graphviz(model, out_file=dot_data, feature_names=iris.feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

from google.colab import drive
drive.mount('/content/drive')

import pickle
print("[INFO] Saving model...")
saved_model=pickle.dump(model,open('/content/drive/My Drive/iris_model.pkl', 'wb'))

model = pickle.load(open('/content/drive/My Drive/iris_model.pkl','rb'))  
# Load the pickled model 
#Dec_from_pickle = pickle.loads(saved_model) 
  
# Use the loaded pickled model to make predictions 
model.predict(X_test)

import joblib
filename = '/content/drive/My Drive/iris.sav'
joblib.dump(model, filename)
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)

!pip install streamlit

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

!pip install pyngrok

!ngrok authtoken 1sU6x13yZdrrriEUXNhkFkpKzQH_2PvSmEmcvPtqzeeXmWNbw

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st 
# from PIL import Image
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.datasets import load_iris
# st.set_option('deprecation.showfileUploaderEncoding', False)
# # Load the pickled model
# model = pickle.load(open('/content/drive/My Drive/iris_model.pkl', 'rb'))
# iris = load_iris()
# def classify(num):
#     if num<0.5:
#         return 'Setosa'
#     elif num <1.5:
#         return 'Versicolor'
#     else:
#         return 'Virginica'
# 
# def main():
#     
#     html_temp = """
#    <div class="" style="background-color:blue;" >
#    <div class="clearfix">           
#    <div class="col-md-12">
#    <center><p style="font-size:40px;color:white;margin-top:10px;">Spark Foundation</p></center> 
#    <center><p style="font-size:30px;color:white;margin-top:10px;">Iris</p></center> 
#    <center><p style="font-size:25px;color:white;margin-top:10px;">Flower Name</p></center> 
#    </div>
#    </div>
#    </div>
#    """
#     st.markdown(html_temp,unsafe_allow_html=True)
#     st.header("Iris Classification ")
#     activities=['Predict']
#     option=st.sidebar.selectbox('Which model would you like to use?',activities)
#     st.subheader(option)
#     sl=st.slider('Select Sepal Length', 0.0, 10.0)
#     sw=st.slider('Select Sepal Width', 0.0, 10.0)
#     pl=st.slider('Select Petal Length', 0.0, 10.0)
#     pw=st.slider('Select Petal Width', 0.0, 10.0)
#     inputs=[[sl,sw,pl,pw]]
# 
#     if st.button("Classify"):
#         if option=='Predict':
#             st.success(classify(model.predict(inputs)))
# 
#     if st.button("About"):
#       st.header("By Prakhar Mehrishi")
#       st.subheader("Intern , The Spark Foundation")
#     html_temp = """
#     <div class="" style="background-color:orange;" >
#     <div class="clearfix">           
#     <div class="col-md-12">
#     <center><p style="font-size:20px;color:white;margin-top:10px;">Iris Classifier</p></center> 
#     </div>
#     </div>
#     </div>
#     """
#     st.markdown(html_temp,unsafe_allow_html=True)
# if __name__=='__main__':
#   main()

!nohup streamlit run  app.py &

from pyngrok import ngrok
url=ngrok.connect(port='8050')
url

!streamlit run --server.port 80 app.py
