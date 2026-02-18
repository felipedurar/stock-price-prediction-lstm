from pydantic import BaseModel


class TrainRequest(BaseModel):
    ticker: str
    lookback: int = 60
    horizon: int = 1
