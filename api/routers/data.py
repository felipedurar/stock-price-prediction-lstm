from fastapi import APIRouter, Depends, Query, HTTPException
from datetime import date as Date
from typing import Optional

from api.schema.reconcile_request import ReconcileRequest
from api.schema.price_response import StockPriceOut
from api.services.stock_data_ingestion_service import ingest_stock_data
from api.services.stock_price_service import StockPriceService, get_stock_price_service

router = APIRouter()

@router.post("/reconcile", summary="Reconcile stock prices (initial or incremental)")
async def data_reconcile(req: ReconcileRequest):
    ingest_stock_data(req.ticker)
    return {"ok": True}


@router.get("/tickers", summary="List tickers available in the database")
async def list_tickers(
    stock_price_service: StockPriceService = Depends(get_stock_price_service),
):
    tickers = stock_price_service.list_tickers()
    return {"tickers": tickers, "count": len(tickers)}


@router.get(
    "/prices",
    response_model=list[StockPriceOut],
    summary="Query OHLCV prices by ticker and optional date range",
)
async def query_prices(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    start_date: Optional[Date] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[Date] = Query(None, description="YYYY-MM-DD"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    stock_price_service: StockPriceService = Depends(get_stock_price_service),
):
    # Basic sanity checks
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date cannot be after end_date")

    rows = stock_price_service.get_prices(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
        order=order,
    )
    return rows