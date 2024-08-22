from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from typing import List

from application.train_model import process_data, predict

# Initializing FastAPI app
app = FastAPI()

# Loading pre-trained model
with open('model/lr_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Loading pre-trained encoder
with open('model/encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)


class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States",
                                alias="native-country")
    salary: str = Field(..., example="<=50K")


# Declaring data model for the returned prediction
class PredictResponse(BaseModel):
    prediction: List[str]


# Get method to output a welcome message
@app.get("/")
async def say_hello() -> str:
    return "Welcome to this project"


# POST method for inference
@app.post("/predict", response_model=PredictResponse)
async def call_predict(input_data: Data):
    try:
        # turning the Pydantic model into a dict, cleaning names and converting to df
        data_dict = input_data.dict()
        data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
        data = pd.DataFrame.from_dict(data)

        # processing data
        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        data_processed, _, _, _ = process_data(
            data, categorical_features=cat_features, training=False, encoder=encoder, encoder_path=None
        )
        # returning inference predictions
        predictions = predict(model, data_processed)
        return PredictResponse(prediction=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
