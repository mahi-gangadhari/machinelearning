import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import pickle
from io import BytesIO

# URL of the pickled model
model_url = 'https://raw.githubusercontent.com/Srikanth1316/Machine-Learning/master/model_pickle2'
response = requests.get(model_url, stream=True)
response.raise_for_status()

# Load the model using pickle
model = pickle.load(BytesIO(response.content))

# Function to predict house price
def predict_house_price(area, bedrooms, age):
    input_data = [[area, bedrooms, age]]
    predicted_price = model.predict(input_data)
    return predicted_price[0]

