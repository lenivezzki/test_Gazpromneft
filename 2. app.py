from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
from sklearn.metrics import mean_squared_error
import numpy as np

app = FastAPI()
model = None

class InputData(BaseModel):
    X5: float
    X8: float
    X11: float
    X13: float
    X14: float
    X15: float
    X16: float
    X17: float
    X33: float
    X36: float
    true_value: float

@app.on_event("startup")
def startup_event():
    global model
    global scaler_y
    global scaler_x

    with open ('model.pkl','rb') as file:
        model = pickle.load(file)

    with open ('scaler_y.pkl','rb') as file:
        scaler_y = pickle.load(file)

    with open ('scaler_x.pkl','rb') as file:
        scaler_x = pickle.load(file)

class ModelResponse(BaseModel):
    model_prediction: float

class ModelInfo(BaseModel):
    mse: float
    important_features: list
@app.post("/predict")
def predict(input_data: InputData):
    features = [[input_data.X5,
                 input_data.X8,
                 input_data.X11,
                 input_data.X13,
                 input_data.X14,
                 input_data.X15,
                 input_data.X16,
                 input_data.X17,
                 input_data.X33,
                 input_data.X36]]
    features = scaler_x.transform(features)
    prediction = model.predict(features)
    prediction = scaler_y.inverse_transform(prediction)
    return ModelResponse(model_prediction = prediction)

@app.post("/get_info_by_model")
def get_info_by_model(input_data: InputData):
    true_value = [[input_data.true_value]]
    features = [[input_data.X5,
             input_data.X8,
             input_data.X11,
             input_data.X13,
             input_data.X14,
             input_data.X15,
             input_data.X16,
             input_data.X17,
             input_data.X33,
             input_data.X36]]

    features = scaler_x.transform(features)
    prediction = model.predict(features)
    predicted_value = scaler_y.inverse_transform(prediction)
    mse = mean_squared_error(true_value, predicted_value)

    k_best_features = ['X5', 'X8', 'X11', 'X13', 'X14', 'X15', 'X16', 'X17', 'X33', 'X36']
    feature_importances = np.abs(model.coef_)
    sorted_indices = np.argsort(feature_importances[0])[::-1]
    sorted_feature_names = [k_best_features[i] for i in sorted_indices]

    return ModelInfo(mse = mse, important_features = sorted_feature_names)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
