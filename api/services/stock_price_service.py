
from fastapi import Depends

from sqlmodel import Session, select, and_, func
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import func

from api.db import get_session
from api.models.stock_price import StockPrice

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
        
def get_stock_price_service(session: Session = Depends(get_session)) -> StockPrice:
    return StockPriceService(session)
