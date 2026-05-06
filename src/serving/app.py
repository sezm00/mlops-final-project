from fastapi import FastAPI
import pandas as pd
import joblib
import os

app = FastAPI(title="MLOps Churn API")

MODEL_PATH = "data/processed/preprocessing_pipeline.pkl"

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    if model is None:
        return {"status": "model not loaded"}
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not available"}

    df = pd.DataFrame([data])
    prediction = model.predict(df).tolist()

    return {"prediction": prediction}