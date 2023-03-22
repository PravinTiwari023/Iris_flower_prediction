import pandas as pd

data = pd.read_csv('iris_dataset.csv')

X= data.drop('variety',axis=1)

Y= data['variety']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# # Linear regression
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# lr_score = lr.score(X_test, y_test)

# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_score = logreg.score(X_test, y_test)

# Decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)

# K-nearest neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)

import streamlit as st

st.title('Machine Learning Algorithms')

# st.write('Linear Regression Score:', lr_score)
st.write('Logistic Regression Score:', logreg_score)
st.write('Decision Tree Score:', dt_score)
st.write('KNN Score:', knn_score)