
import pandas as pd
import yfinance as yf

from datetime import date, timedelta
from sqlmodel import Session
from api.db import engine
from api.services.stock_price_service import StockPriceService

def normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Date is a normal column
    df = df.reset_index()

    # If columns are MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[0] != "Price" else col[1]
            for col in df.columns.to_list()
        ]

    # Standardize expected column names
    rename_map = {}
    if "Datetime" in df.columns and "Date" not in df.columns:
        rename_map["Datetime"] = "Date"

    df = df.rename(columns=rename_map)

    # Make sure Date is datetime64
    df["Date"] = pd.to_datetime(df["Date"])

    return df

def df_to_rows(df: pd.DataFrame, ticker: str) -> list[dict]:
    df = normalize_yf_df(df)

    rows = []
    for r in df.itertuples(index=False):
        rows.append({
            "ticker": ticker,
            "date": r.Date.date(),
            "open": float(r.Open),
            "high": float(r.High),
            "low": float(r.Low),
            "close": float(r.Close),
            "volume": float(getattr(r, "Volume", 0.0) or 0.0),
        })
    return rows

def sync_stock_data():
    with Session(engine) as session:
        stock_price_service = StockPriceService(session)
        tickers = stock_price_service.list_tickers()
        
    for c_ticker in tickers:
        print(f"Ingesting Stock Data for '{c_ticker}'...")
        ingest_stock_data(c_ticker)
        print(f"Finished Ingesting Stock Data for '{c_ticker}'")
    
def ingest_stock_data(ticker: str, period: str = "5y", overlap_days: int = 3):
    with Session(engine) as session:
        stock_price_service = StockPriceService(session)
        latest_date = stock_price_service.get_latest_date(ticker)

    # Check Download Interval
    if latest_date is None:
        # First Extract
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )
    else:
        start_date = latest_date - timedelta(days=overlap_days)
        end_date = date.today() + timedelta(days=1)

        df = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False
        )

    rows = df_to_rows(df, ticker)

    with Session(engine) as session:
        stock_price_service = StockPriceService(session)
        stock_price_service.bulk_upsert(rows)