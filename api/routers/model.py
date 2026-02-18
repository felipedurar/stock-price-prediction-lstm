from fastapi import APIRouter, Depends, Query, HTTPException

from api.ml.training.trainer import generate_metadata
from api.schema.model_registry import (
    TrainRequest,
    TrainedModelOut,
    ActivateModelRequest,
    ActivateModelResponse,
)
from api.services.model_registry_service import ModelRegistryService, get_model_registry_service
from api.services.model_service import ModelService, get_model_service
from api.services.stock_price_service import StockPriceService, get_stock_price_service

router = APIRouter()

@router.post("/train", summary="Treina um modelo para um ticker e registra nova versão")
async def train_model(
    req: TrainRequest,
    stock_price_service: StockPriceService = Depends(get_stock_price_service),
    model_service: ModelService = Depends(get_model_service),
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
):
    closes_vals = stock_price_service.get_closes_from_db(req.ticker)

    if len(closes_vals) < (req.lookback + req.horizon + 10):
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data to train. Need at least ~{req.lookback + req.horizon + 10} closes.",
        )

    model, scaler, metrics = model_service.run_training(closes_vals, req.lookback, req.horizon)

    metadata = generate_metadata(req.ticker, model, req.lookback, metrics)
    artifact_path = model_service.save_model_bundle(model, scaler, metadata)

    model_registry_service.register_model_version(
        ticker=req.ticker,
        model_version=metadata["model_version"],
        model_type="lstm_regressor",
        artifact_path=artifact_path,
        lookback=req.lookback,
        horizon=req.horizon,
        mae=float(metrics["MAE"]),
        rmse=float(metrics["RMSE"]),
        mape=float(metrics["MAPE"]),
        set_active=True,
    )

    return {
        "ok": True,
        "ticker": req.ticker,
        "model_version": metadata["model_version"],
        "metrics": metrics,
        "artifact_path": artifact_path,
    }


@router.get(
    "/versions",
    response_model=list[TrainedModelOut],
    summary="Lista versões treinadas de um ticker",
)
async def list_versions(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
):
    rows = model_registry_service.list_versions(ticker=ticker)
    return rows


@router.get(
    "/active",
    response_model=TrainedModelOut,
    summary="Retorna o modelo ativo de um ticker",
)
async def get_active_model(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
):
    row = model_registry_service.get_active_model(ticker=ticker)
    if not row:
        raise HTTPException(status_code=404, detail="No active model for this ticker.")
    return row


@router.post(
    "/activate",
    response_model=ActivateModelResponse,
    summary="Ativa uma versão do modelo para um ticker",
)
async def activate_model(
    req: ActivateModelRequest,
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
):
    try:
        model_registry_service.activate_model(ticker=req.ticker, model_version=req.model_version)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ActivateModelResponse(ticker=req.ticker, model_version=req.model_version)


@router.get(
    "/versions/{model_version}",
    response_model=TrainedModelOut,
    summary="Busca uma versão específica pelo model_version",
)
async def get_version(
    model_version: str,
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
):
    row = model_registry_service.get_by_version(model_version)
    if not row:
        raise HTTPException(status_code=404, detail="Model version not found.")
    return row