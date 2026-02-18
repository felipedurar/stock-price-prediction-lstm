from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TrainRequest(BaseModel):
    ticker: str
    lookback: int = 60
    horizon: int = 1

class TrainedModelOut(BaseModel):
    ticker: str
    model_version: str
    model_type: str
    artifact_path: str
    lookback: int
    horizon: int
    mae: float
    rmse: float
    mape: float
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True  # pydantic v2: allow reading from ORM object


class ActivateModelRequest(BaseModel):
    ticker: str
    model_version: str


class ActivateModelResponse(BaseModel):
    ok: bool = True
    ticker: str
    model_version: str