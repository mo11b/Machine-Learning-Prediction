import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('prices_model.pickle', 'rb') as file:
        model = pickle.load(file)
    return model

regressor = load_model()

data_columns = ["bed", 
                "bath", 
                "house_size", 
                "connecticut", 
                "delaware", 
                "maine", 
                "massachusetts", 
                "new hampshire", 
                "new jersey", 
                "new york", 
                "pennsylvania", 
                "puerto rico", 
                "rhode island", 
                "vermont", 
                "virgin islands"]

def show_predict_page():
    st.title("House Price Prediction")

    states = ["connecticut", "delaware", "maine", "massachusetts", "new hampshire", "new jersey", "new york",
              "pennsylvania", "puerto rico", "rhode island", "vermont", "virgin islands"]

    state = st.selectbox("Select State", states)

    bath = st.slider("Number of Bathrooms", 1, 8, 2)
    bed = st.slider("Number of Bedrooms", 1, 8, 2)
    house_size = st.slider("House Size (sqft)", 300, 5500, 1000)

    ok = st.button("Calculate Price")
    if ok:
        state_lower = state.lower()
        loc_index = data_columns.index(state_lower)

        x = np.zeros(len(data_columns))
        x[0] = bed
        x[1] = bath
        x[2] = house_size
        if loc_index >= 0:
            x[loc_index] = 1

        # Convert x to a 2D array for prediction
        x = x.reshape(1, -1)

        price = regressor.predict(x)
        st.subheader(f"The estimated house price is ${price[0]:.2f}")

show_predict_page()
