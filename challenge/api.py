from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator, Field
import pandas as pd
from datetime import datetime
from .model import DelayModel
import logging
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logs_consola import setup_logging

# Setup logging
setup_logging()
logging = logging.getLogger(__name__)

app = FastAPI()
model = DelayModel()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body},
    )

class PredictRequest(BaseModel):
    Fecha_I: str = Field(..., example="2023-01-01 12:00:00")
    OPERA: str = Field(..., example="Aerolineas Argentinas")
    TIPOVUELO: str = Field(..., example="N")
    MES: int = Field(..., gt=0, lt=13, example=3)

    @field_validator('MES')
    def check_mes(cls, v):
        if v < 1 or v > 12:
            raise ValueError('MES debe estar entre 1 y 12')
        return v

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    try:
        data = pd.DataFrame(request['flights'])
        features = model.preprocess(data)
        prediction = model.predict(features)
        return {"prediction": prediction}
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        response = {"error": str(e)}
        return JSONResponse(status_code=400, content=response)
