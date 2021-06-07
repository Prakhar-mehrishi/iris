import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

df['target'] = iris.target
df.head()


df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')
y = df.target

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_test,y_test)





