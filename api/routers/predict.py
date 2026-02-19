from fastapi import APIRouter, Depends, HTTPException, Path, Query

from api.ml.inference.predictor import predict_and_inverse
from api.schema.predict_request import PredictRequest
from api.services.model_registry_service import ModelRegistryService, get_model_registry_service
from api.services.model_service import ModelService, get_model_service
from api.services.stock_price_service import StockPriceService, get_stock_price_service

import numpy as np

router = APIRouter()

@router.post("/", summary="Realiza a predição a partir de um modelo treinado")
async def predict_stock(
    req: PredictRequest,
    model_service: ModelService = Depends(get_model_service),
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
    stock_price_service: StockPriceService = Depends(get_stock_price_service),
):
    active_model = model_registry_service.get_active_model(ticker=req.ticker)
    if not active_model:
        raise HTTPException(status_code=404, detail="No active model for this ticker. Train and activate one first.")

    # Load model bundle
    model, scaler, meta = model_service.load_model_bundle(
        ticker=active_model.ticker,
        model_version=active_model.model_version
    )

    lookback = getattr(active_model, "lookback", None) or meta.lookback
    if not lookback:
        raise HTTPException(status_code=500, detail="Model lookback not found in registry or metadata.")

    # Get last closes from DB
    last_closes = stock_price_service.get_last_closes(req.ticker, lookback=lookback)
    if last_closes.shape[0] < lookback:
        raise HTTPException(status_code=400, detail=f"Not enough data in DB. Need {lookback} closes.")

    # Build X
    X = build_X_from_last_closes(last_closes, lookback=lookback, scaler=scaler)

    # Predict (returns array shape (1,1))
    pred_real = predict_and_inverse(model, X, scaler)
    predicted_close = float(pred_real[0, 0])

    return {
        "ticker": req.ticker,
        "model_version": active_model.model_version,
        "lookback": lookback,
        "predicted_close": predicted_close,
    }
    
def build_X_from_last_closes(last_closes: np.ndarray, lookback: int, scaler) -> np.ndarray:
    """
    last_closes: 1D array shape (lookback,)
    returns X: shape (1, lookback, 1) scaled
    """
    if last_closes.shape[0] != lookback:
        raise ValueError(f"Need {lookback} closes, got {last_closes.shape[0]}")

    x = last_closes.astype(np.float32).reshape(-1, 1)     # (lookback, 1)
    x_scaled = scaler.transform(x)                        # (lookback, 1)
    X = x_scaled.reshape(1, lookback, 1)                  # (1, lookback, 1)
    return X