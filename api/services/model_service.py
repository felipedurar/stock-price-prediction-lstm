import os, json, joblib, torch
from fastapi import Depends

from sqlmodel import Session, select, and_, func
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import func

from api.db import get_session
from api.ml.inference.predictor import inverse_y, predict_and_inverse
from api.ml.metrics.regression import mae, mape, rmse
from api.ml.preprocessing.scalling import prepare_data
from api.ml.registry.builders import build_model_from_metadata
from api.ml.registry.schemas import ModelMetadata
from api.ml.training.trainer import train_model
from api.models.stock_price import StockPrice
from datetime import datetime, timezone

import numpy as np

class ModelService:
    def __init__(self, session: Session):
        self.session = session
        
    def run_training(self, closes: np.ndarray, lookback=60):
        X_train, y_train, X_val, y_val, scaler = prepare_data(closes, lookback=lookback, train_ratio=0.8)

        model, best_val_loss = train_model(
            X_train, y_train, X_val, y_val,
            hidden_size=64, num_layers=2, dropout=0.2,
            lr=1e-3, epochs=20, batch_size=32
        )

        # Evaluate Real Price (Not normalized)
        y_val_real = inverse_y(y_val, scaler)
        pred_val_real = predict_and_inverse(model, X_val, scaler)

        metrics = {
            "MAE": mae(y_val_real, pred_val_real),
            "RMSE": rmse(y_val_real, pred_val_real),
            "MAPE": mape(y_val_real, pred_val_real),
        }

        print("Metrics:", metrics)
        
        return model, scaler, metrics
    
    def predict_next_close(self, model, scaler, last_closes: np.ndarray, lookback: int):
        """
        last_closes: array 1D com os últimos closes reais (sem normalizar)
        """
        assert len(last_closes) == lookback

        x = last_closes.astype(np.float32).reshape(-1, 1)
        x_scaled = scaler.transform(x)                 # (lookback,1)
        X = x_scaled.reshape(1, lookback, 1)          # (1, lookback, 1)

        pred = predict_and_inverse(model, X, scaler)  # (1,1)
        return float(pred[0, 0])
    
    def load_model_bundle(self, model_dir: str):
        # Metadata
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            meta_dict = json.load(f)

        meta = ModelMetadata.model_validate(meta_dict)

        # Scaler
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

        # Model
        model = build_model_from_metadata(meta)
        state = torch.load(os.path.join(model_dir, "weights.pt"), map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        return model, scaler, meta
    
    def save_model_bundle(self, model, scaler, metadata: dict, base_dir="trained_models"):
        ticker = metadata["ticker"]
        model_version = metadata["model_version"]

        out_dir = os.path.join(base_dir, ticker, model_version)
        os.makedirs(out_dir, exist_ok=True)

        # Weights
        torch.save(model.state_dict(), os.path.join(out_dir, "weights.pt"))

        # Scaler
        joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

        # Metadata
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
        
def get_model_service(session: Session = Depends(get_session)) -> ModelService:
    return ModelService(session)