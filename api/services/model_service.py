import joblib
import json
import os
import torch

import numpy as np
from fastapi import Depends
from sqlmodel import Session

from api.db import get_session
from api.ml.inference.predictor import inverse_y, predict_and_inverse
from api.ml.metrics.regression import mae, mape, rmse
from api.ml.preprocessing.scalling import prepare_data
from api.ml.registry.builders import build_model_from_metadata
from api.ml.registry.schemas import ModelMetadata
from api.ml.training.trainer import train_model
from api.storage.r2_client import upload_file

from api.storage.r2_client import download_file


class ModelService:
    def __init__(self, session: Session):
        self.session = session
        
    def run_training(self, closes: np.ndarray, lookback=60, horizon=1):
        X_train, y_train, X_val, y_val, scaler = prepare_data(closes, lookback=lookback, horizon=horizon, train_ratio=0.8)

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

    def load_model_bundle(self, ticker: str, model_version: str):
        tmp_dir = os.path.join("/tmp", ticker, model_version)
        os.makedirs(tmp_dir, exist_ok=True)

        remote_base = f"models/{ticker}/{model_version}"

        weights_path = os.path.join(tmp_dir, "weights.pt")
        scaler_path = os.path.join(tmp_dir, "scaler.pkl")
        metadata_path = os.path.join(tmp_dir, "metadata.json")

        # Download files
        download_file(f"{remote_base}/weights.pt", weights_path)
        download_file(f"{remote_base}/scaler.pkl", scaler_path)
        download_file(f"{remote_base}/metadata.json", metadata_path)

        # Load metadata
        with open(metadata_path, "r") as f:
            meta_dict = json.load(f)

        meta = ModelMetadata.model_validate(meta_dict)

        # Load scaler
        scaler = joblib.load(scaler_path)

        # Build model
        model = build_model_from_metadata(meta)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        return model, scaler, meta

    def save_model_bundle(self, model, scaler, metadata: dict):
        ticker = metadata["ticker"]
        model_version = metadata["model_version"]

        tmp_dir = os.path.join("/tmp", ticker, model_version)
        os.makedirs(tmp_dir, exist_ok=True)

        weights_path = os.path.join(tmp_dir, "weights.pt")
        scaler_path = os.path.join(tmp_dir, "scaler.pkl")
        metadata_path = os.path.join(tmp_dir, "metadata.json")

        # Save locally (temporary)
        torch.save(model.state_dict(), weights_path)
        joblib.dump(scaler, scaler_path)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload to R2 (preserving structure)
        remote_base = f"models/{ticker}/{model_version}"

        upload_file(weights_path, f"{remote_base}/weights.pt")
        upload_file(scaler_path, f"{remote_base}/scaler.pkl")
        upload_file(metadata_path, f"{remote_base}/metadata.json")

        return remote_base
        
def get_model_service(session: Session = Depends(get_session)) -> ModelService:
    return ModelService(session)