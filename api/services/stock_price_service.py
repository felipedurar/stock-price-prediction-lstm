
from typing import List, Optional
from fastapi import Depends
from datetime import date as Date

from sqlmodel import Session, select, and_, func
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import func

from api.db import get_session
from api.models.stock_price import StockPrice

import numpy as np

class StockPriceService:
    def __init__(self, session: Session):
        self.session = session
        
    def bulk_upsert(self, rows: list[dict]) -> None:
        if not rows:
            return

        stmt = insert(StockPrice).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "date"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
            }
        )
        self.session.exec(stmt)
        self.session.commit()
    
    def get_closes_from_db(self, ticker: str) -> np.ndarray:
        stmt = (
            select(StockPrice.close)
            .where(StockPrice.ticker == ticker)
            .order_by(StockPrice.date.asc())
        )

        rows = self.session.exec(stmt).all()

        # Se vier como tupla: [(127.43,), (126.33,), ...]
        closes = [float(r) if not isinstance(r, tuple) else float(r[0]) for r in rows]

        return np.array(closes, dtype=np.float32)
    
    def get_last_closes(self, ticker: str, lookback: int) -> np.ndarray:
        stmt = (
            select(StockPrice.close)
            .where(StockPrice.ticker == ticker)
            .order_by(StockPrice.date.desc())
            .limit(lookback)
        )
        rows = self.session.exec(stmt).all()

        closes = [float(r) if not isinstance(r, tuple) else float(r[0]) for r in rows]
        closes = closes[::-1]  # reverse to chronological order (old -> new)

        return np.array(closes, dtype=np.float32)
    
    def get_latest_date(self, ticker: str) -> Date | None:
        stmt = (
            select(func.max(StockPrice.date))
            .where(StockPrice.ticker == ticker)
        )
        latest = self.session.exec(stmt).one()
        
        if isinstance(latest, tuple):
            latest = latest[0]
        return latest
    
    def list_tickers(self) -> List[str]:
        stmt = select(StockPrice.ticker).distinct().order_by(StockPrice.ticker.asc())
        rows = self.session.exec(stmt).all()
        return [r if not isinstance(r, tuple) else r[0] for r in rows]
    
    def get_prices(
        self,
        *,
        ticker: str,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        limit: int = 200,
        offset: int = 0,
        order: str = "asc",  # "asc" or "desc"
    ) -> List[StockPrice]:
        stmt = select(StockPrice).where(StockPrice.ticker == ticker)

        if start_date:
            stmt = stmt.where(StockPrice.date >= start_date)
        if end_date:
            stmt = stmt.where(StockPrice.date <= end_date)

        if order.lower() == "desc":
            stmt = stmt.order_by(StockPrice.date.desc())
        else:
            stmt = stmt.order_by(StockPrice.date.asc())

        stmt = stmt.offset(offset).limit(limit)

        return list(self.session.exec(stmt).all())
    
        
def get_stock_price_service(session: Session = Depends(get_session)) -> StockPriceService:
    return StockPriceService(session)
