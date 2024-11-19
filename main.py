from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
import uvicorn


app = FastAPI(title="Iris Classifier API", 
              description="API for classifying Iris flower", 
              version="1.0.0")

model = joblib.load('iris_model.joblib')