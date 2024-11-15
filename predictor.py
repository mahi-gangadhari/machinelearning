import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import requests
import pickle

model_url = 'https://raw.githubusercontent.com/Srikanth1316/Machine-Learning/master/model_pickle2'
response = requests.get(model_url, stram = True)
response.raise_for_status()

    model = pickle.load(response.content)


def predict_house_price(area,bedrooms,age):
    input_data =[[area,bedrooms,age]]
    predicted_price = model.predict(input_data)
    return predicted_price[0]
