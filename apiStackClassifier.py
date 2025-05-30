from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model = joblib.load('stacking_classifier_model.pkl')  # Ensure this path is correct

# Define request data model
class DepressionPredictRequest(BaseModel):
    Age: int
    Academic_Pressure: int
    CGPA: float
    Study_Satisfaction: int
    Sleep_Duration: float
    Dietary_Habits: int
    Suicidal_thoughts: int
    Work_Study_Hours: int
    Financial_Stress: int
    Family_History_of_Mental_Illness: int
    City: int
    Profession: int
    Degree: int

# Initialize FastAPI app
app = FastAPI()

# Enable CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction route
@app.post("/predict")
def predict_depression(data: DepressionPredictRequest):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of depression class
    return {
        "predicted_class": int(prediction),
        "probability_depression": float(probability)
    }
