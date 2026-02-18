from pydantic import BaseModel
from typing import Literal, List, Dict, Any, Optional

class ModelParams(BaseModel):
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0

class ModelMetadata(BaseModel):
    ticker: str
    model_version: str
    model_type: Literal["lstm_regressor"]
    model_params: ModelParams

    lookback: int
    horizon: int = 1
    features: List[str] = ["close"]

    metrics: Optional[Dict[str, float]] = None
    train_params: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None