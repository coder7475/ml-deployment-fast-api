from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Iris Classifier API", 
              description="API for classifying Iris flower", 
              version="1.0.0")

model = joblib.load('iris_model.joblib')

"""
data model for the input data
configuration
"""

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }


"""
designing the response input model
"""

class IrisResponse(BaseModel):
    predicted_species: str
    probability: float
    

"""

We have to define the mapping of the species

"""

SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Root route

@app.get("/")

async def root():
    return {
        "message": "Welcome to the Iris Classifier API"
    }

# Prediction endpoint
@app.post("/predict", response_model=IrisResponse)
# Function for `/predict` route
async def predict(iris_input: IrisInput):
    try:
        features = np.array([[
            iris_input.sepal_length, 
            iris_input.sepal_width, 
            iris_input.petal_length, 
            iris_input.petal_width
        ]])

        prediction = model.predict(features)[0]
        probability = np.max(model.predict_proba(features))
        
        return IrisResponse(
            predicted_species = SPECIES_MAP[prediction],
            probability=float(probability)
        )
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))