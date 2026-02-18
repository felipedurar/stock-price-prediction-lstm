from fastapi import APIRouter, Depends, Path, Query

from api.ml.training.trainer import generate_metadata
from api.schema.train_request import TrainRequest
from api.services.model_registry_service import ModelRegistryService, get_model_registry_service
from api.services.model_service import ModelService, get_model_service
from api.services.stock_price_service import StockPriceService, get_stock_price_service

router = APIRouter()

@router.post(
    "/train",
    summary="Realiza o treinamento do Modelo a partir de um Ticker especifico"
)
async def train_model(
    req: TrainRequest,
    stock_price_service: StockPriceService = Depends(get_stock_price_service),
    model_service: ModelService = Depends(get_model_service),
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service)
    ):
    """
    Realiza o treinamento do Modelo a partir de um Ticker especifico
    """
    print("Quering Data...")
    closes_vals = stock_price_service.get_closes_from_db(req.ticker)
    
    print("Training...")
    model, scaler, metrics = model_service.run_training(closes_vals, req.lookback, req.horizon)
    print("Finished Training")
    
    metadata = generate_metadata(req.ticker, model, req.lookback, metrics)
    print(metadata)
    artifact_path = model_service.save_model_bundle(model, scaler, metadata)
    print("Model Successfully Saved!")
    
    model_registry_service.register_model_version(
        ticker = req.ticker,
        model_version = metadata["model_version"],
        model_type = "lstm_regressor",
        artifact_path = artifact_path,
        lookback = req.lookback,
        horizon = req.horizon,
        mae = metrics["MAE"],
        rmse = metrics["RMSE"],
        mape = metrics["MAPE"],
        set_active=True
    )
    
    return { "ok": True }