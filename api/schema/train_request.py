from pydantic import BaseModel


class TrainRequest(BaseModel):
    ticker: str
    lookback: int
