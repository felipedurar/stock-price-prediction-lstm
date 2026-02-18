from pydantic import BaseModel
from datetime import date, datetime

class StockPriceOut(BaseModel):
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    created_at: datetime

    class Config:
        from_attributes = True
        