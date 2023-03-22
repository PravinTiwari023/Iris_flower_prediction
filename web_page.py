import pandas as pd
import streamlit as st
# from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from main import set_background

# Setting BG Image and removing WaterMark of streamlit
set_background('pexels-pixabay-262713.jpg')

# # Loading Iris dataset
# data = pd.read_csv('iris_dataset.csv')
#
# if st.checkbox('Show Raw data'):
#     st.header('Raw data')
#     st.dataframe(data)
#
# # Count of each kind of flower
# data1 = data[data['variety'] == 'Setosa']
# data2 = data[data['variety'] == 'Virginica']
# data3 = data[data['variety'] == 'Versicolor']
#
# st.sidebar.write('Number of Setosa flower: 50')
# st.sidebar.write('Number of Virginica flower: 50')
# st.sidebar.write('Number of Versicolor flower: 50')

# Load the Iris dataset
iris = pd.read_csv('iris_dataset.csv')
X = iris.drop('variety',axis=1)
Y = iris['variety']

if st.checkbox('Check Data'):
    st.subheader('Iris Dataset')
    st.dataframe(X)

# Train a Random Forest classifier on the Iris dataset
rfc = RandomForestClassifier()
rfc.fit(X, Y)


# Define the Streamlit app
def app():
    st.title("Iris Flower Classification")
    st.write(
        "This app uses a Random Forest classifier to predict the species of an iris flower based on its characteristics.")

    # Create input widgets for the user to enter the characteristics of an iris flower
    sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.0, 0.1)
    sepal_width = st.slider("Sepal width", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal length", 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider("Petal width", 0.1, 2.5, 1.0, 0.1)

    # Make a prediction based on the user's input
    prediction = rfc.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # Show the predicted species of the iris flower
    species = iris.target_names[prediction[0]]
    st.write("Predicted species:", species)


# Run the Streamlit app
if __name__ == '__main__':
    app()