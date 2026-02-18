from fastapi import APIRouter, Depends, Path, Query

from api.schema.train_request import TrainRequest
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
    ):
    """
    Realiza o treinamento do Modelo a partir de um Ticker especifico
    """
    print("Quering Data...")
    closes_vals = stock_price_service.get_closes_from_db(req.ticker)
    
    print("Training...")
    model, scaler, metrics = model_service.run_training_example(closes_vals)
    
    print("Finished Training")
    return { "ok": True }