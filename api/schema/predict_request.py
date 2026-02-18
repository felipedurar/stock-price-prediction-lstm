from pydantic import BaseModel


class PredictRequest(BaseModel):
    ticker: str

