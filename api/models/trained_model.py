from datetime import datetime as DateTime
from typing import Optional
from sqlmodel import SQLModel, Field

class TrainedModel(SQLModel, table=True):
    __tablename__ = "model_versions"

    id: Optional[int] = Field(default=None, primary_key=True)

    ticker: str = Field(index=True)
    model_version: str = Field(index=True, unique=True)

    model_type: str
    artifact_path: str

    lookback: int
    horizon: int = 1

    mae: float
    rmse: float
    mape: float

    is_active: bool = Field(default=False)
    created_at: DateTime = Field(default_factory=DateTime.utcnow)