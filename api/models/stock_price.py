from datetime import date as Date, datetime as DateTime
from sqlmodel import SQLModel, Field
from typing import Optional
from sqlalchemy import UniqueConstraint

class StockPrice(SQLModel, table=True):
    __tablename__ = "stock_prices"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    ticker: str = Field(index=True)

    date: Date = Field(index=True)
    open: float
    high: float
    low: float
    close: float
    volume: float

    created_at: DateTime = Field(default_factory=DateTime.utcnow)
    
    