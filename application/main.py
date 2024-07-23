from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import List
import io

# Initializing FastAPI app
app = FastAPI()

# Load your pre-trained model
with open('model/lr_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Declaring data model for the returned prediction
class PredictResponse(BaseModel):
    prediction: List[str]

# Get method to output a welcome message
@app.get("/")
async def say_hello() -> str:
    return "Welcome to this project"

# POST method for inference
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        # Load the CSV file into a DataFrame
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), header="infer")
        # Perform prediction using the model
        predictions = model.predict(df)
        return PredictResponse(prediction=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
