from fastapi import FastAPI, Form

import pickle
import numpy as np

app = FastAPI()

class HousePricePredictor:
    def __init__(self, model_file, data_columns):
        self.model = self.load_model(model_file)
        self.data_columns = data_columns

    def load_model(self, model_file):
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        return model

    def predict_price(self, input_data):
        x = np.zeros(len(self.data_columns))
        x[0] = input_data['bed']
        x[1] = input_data['bath']
        x[2] = input_data['house_size']

        state_lower = input_data['state'].lower()
        if state_lower in self.data_columns:
            loc_index = self.data_columns.index(state_lower)
            x[loc_index] = 1

        # Convert x to a 2D array for prediction
        x = x.reshape(1, -1)

        price = self.model.predict(x)
        return price[0]

model_file = 'prices_model.pickle'
data_columns = ["bed", "bath", "house_size", "connecticut", "delaware", "maine", "massachusetts", "new hampshire",
                "new jersey", "new york", "pennsylvania", "puerto rico", "rhode island", "vermont", "virgin islands"]

predictor = HousePricePredictor(model_file, data_columns)

@app.post("/predict/")
async def predict_house_price(
    state: str = Form(...),
    bath: int = Form(...),
    bed: int = Form(...),
    house_size: int = Form(...),
):
    input_data = {
        'state': state,
        'bath': bath,
        'bed': bed,
        'house_size': house_size
    }
    estimated_price = predictor.predict_price(input_data)
    return {"estimated_price": estimated_price}
