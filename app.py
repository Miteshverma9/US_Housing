from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the trained model
with open("US_Housing.pkl", "rb") as f:
    model = pickle.load(f)

# Define the FastAPI app
app = FastAPI()

# Define a request body model
class PredictionRequest(BaseModel):
    
    DATE: int
    H_RATIO_3A_PCT_CHG: float
    HSN1F_3A_PCT_CHG: float
    PERMIT_3A_PCT_CHG: float
    STOCK_MKT_3A_PCT_CHG: float
    BAA_YEILD_10Y_2A_PCT_CHG: float
    US10Y_3A_PCT_CHG: float
    RPCE_A_PCT_CHG: float
    UEMP_3A_PCT_CHG: float
    RGDP_M_PCT_CHG: float

# Define a response model
class PredictionResponse(BaseModel):
    prediction: float

# Define an endpoint for making predictions
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    X = [[
        
        data.DATE,
        data.H_RATIO_3A_PCT_CHG,
        data.HSN1F_3A_PCT_CHG,
        data.PERMIT_3A_PCT_CHG,
        data.STOCK_MKT_3A_PCT_CHG,
        data.BAA_YEILD_10Y_2A_PCT_CHG,
        data.US10Y_3A_PCT_CHG,
        data.RPCE_A_PCT_CHG,
        data.UEMP_3A_PCT_CHG,
        data.RGDP_M_PCT_CHG
    ]]
    
    prediction = model.predict(X)[0]
    return {"prediction": prediction}