import json
import pickle
import numpy as np

linear_model= None
data_columns = None
locations = None

def get_locations():
    return locations


def get_estimated_price(loc, sqft, bhk, bath):
    try:
        loc_index = data_columns.index(locations.lower())
    except:
        loc_index = -1
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] =1
    return round(linear_model.predict([x])[0],2)

## loading the files of ML
def load_artifacts():
    print ("load saved files in artifacts")
    global data_columns
    global locations
    global linear_model

    with open("G:/Case study/WebApp/House Price Prediction/server/artifacts/column.json","r") as f:
        data_columns =    json.load(f)['data_columns']
        locations = data_columns[3:]

    with open("server/artifacts/estate_model.pickle",'rb') as f:
        linear_model = pickle.load(f)
    print("loading artifatcs is done....")
if  __name__ == "__main__":
    load_artifacts()
    print(get_locations())
    print(get_estimated_price('1st Phase JP Nagar',1000,3,3))
    print(get_estimated_price('1st Phase JP Nagar',1000,3,4))
