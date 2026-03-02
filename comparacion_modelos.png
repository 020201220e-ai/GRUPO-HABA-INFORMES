from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="California Housing Prediction API")

# Cargar el mejor modelo guardado
try:
    model = joblib.load("modelo.pkl")
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def home():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: HousingData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        # Convertir a DataFrame para mantener nombres de columnas si el pipeline lo requiere
        features = pd.DataFrame([data.dict().values()], columns=feature_names)
        prediction = model.predict(features)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
