import pandas as pd
import numpy as np
iris_data = pd.read_csv("D:\Python\Iris.csv")
print(iris_data.head())
print(iris_data.isna().sum())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
iris_data['Species'] = le.fit_transform(iris_data['Species'])
print(iris_data.head())

x = iris_data.iloc[:,:-1]
y = iris_data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.8,random_state = 32)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
print("Confusion matrix : \n",confusion_matrix(y_test,y_pred))
print("Accuracy score: ",accuracy_score(y_test,y_pred))

import seaborn as sns
sns.heatmap(confusion_matrix(y_test,y_pred))