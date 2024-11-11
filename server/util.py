import json
import pickle
import numpy as np
import os

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1


    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    current_dir = os.path.dirname(__file__)

    columns_file_path = os.path.join(current_dir, "artifacts/columns.json")

    with open( columns_file_path, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model

    model_file_path = os.path.join(current_dir, "artifacts/banglore_home_prices_model.pickle")
    with open( model_file_path, 'rb') as f:
        __model = pickle.load(f)

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000,3,3))
    print(get_estimated_price('1st Phase JP Nagar', 1000,2,2))