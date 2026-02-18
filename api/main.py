from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import structlog

from api.db import init_db, engine
from api.logging_config import setup_logging
from api.routers import data, model, predict
from api.tasks import perform_initial_reconciliation

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from api.monitoring.prometheus import (
    API_REQUEST_COUNT,
    API_REQUEST_LATENCY
)


scheduler = AsyncIOScheduler(timezone="UTC")
logger = structlog.get_logger()

def setup_database():
    print("Setting Up Database...")

    # Create tables
    init_db()

def setup_scheduler():
    print("Setting Up Scheduler...")
    # Startup Update
    scheduler.add_job(
        perform_initial_reconciliation,
        trigger="date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=10),
        id="initial_reconciliation",
        replace_existing=True,
        misfire_grace_time=300,
    )
    
    # Daily Update
    scheduler.add_job(
        perform_initial_reconciliation,
        trigger=CronTrigger(hour=1, minute=0),
        id="daily_reconciliation",
        replace_existing=True,
        misfire_grace_time=300,
    )

    scheduler.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting up...")
    setup_logging()
    setup_database()
    setup_scheduler()
    print("Setup Completed!")

    try:
        yield
    finally:
        print("Shutting down...")
        scheduler.shutdown(wait=False)
        print("Scheduler stopped.")

app = FastAPI(
    title="Stock Price Prediction API",
    version="0.1.0",
    description="Stock Price Prediction using LSTM",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    API_REQUEST_COUNT.inc()
    API_REQUEST_LATENCY.observe(duration)

    return response

app.include_router(data.router, prefix="/api/v1/data", tags=["Data"])
app.include_router(model.router, prefix="/api/v1/model", tags=["Model"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["Predict"])

# --- Healthcheck ---
@app.get("/api/v1/health", tags=["Health"], status_code=200)
async def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
