from __future__ import annotations

from typing import Optional, List
from fastapi import Depends
from sqlmodel import Session, select
from sqlalchemy import update

from api.db import get_session
from api.models.trained_model import TrainedModel

class ModelRegistryService:
    def __init__(self, session: Session):
        self.session = session

    def register_model_version(
        self,
        *,
        ticker: str,
        model_version: str,
        model_type: str,
        artifact_path: str,
        lookback: int,
        horizon: int,
        mae: float,
        rmse: float,
        mape: float,
        set_active: bool = False,
    ) -> TrainedModel:
        # Check Duplicates
        existing = self.get_by_version(model_version)
        if existing:
            if set_active:
                self.activate_model(ticker=ticker, model_version=model_version)
                existing = self.get_by_version(model_version)  # refresh
            return existing

        row = TrainedModel(
            ticker=ticker,
            model_version=model_version,
            model_type=model_type,
            artifact_path=artifact_path,
            lookback=lookback,
            horizon=horizon,
            mae=mae,
            rmse=rmse,
            mape=mape,
            is_active=False,
        )

        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)

        if set_active:
            self.activate_model(ticker=ticker, model_version=model_version)
            row = self.get_by_version(model_version)  # Get the updated one

        return row

    def list_versions(self, *, ticker: str) -> List[TrainedModel]:
        stmt = (
            select(TrainedModel)
            .where(TrainedModel.ticker == ticker)
            .order_by(TrainedModel.created_at.desc())
        )
        return list(self.session.exec(stmt).all())

    def get_active_model(self, *, ticker: str) -> Optional[TrainedModel]:
        stmt = (
            select(TrainedModel)
            .where(TrainedModel.ticker == ticker, TrainedModel.is_active == True)
            .limit(1)
        )
        return self.session.exec(stmt).first()

    def get_by_version(self, model_version: str) -> Optional[TrainedModel]:
        stmt = select(TrainedModel).where(TrainedModel.model_version == model_version).limit(1)
        return self.session.exec(stmt).first()

    def activate_model(self, *, ticker: str, model_version: str) -> None:
        # Deactivate all tickers first
        self.session.exec(
            update(TrainedModel)
            .where(TrainedModel.ticker == ticker)
            .values(is_active=False)
        )

        # Activate only the target
        result = self.session.exec(
            update(TrainedModel)
            .where(
                TrainedModel.ticker == ticker,
                TrainedModel.model_version == model_version,
            )
            .values(is_active=True)
        )

        # Check affected lines
        if result.rowcount == 0:
            self.session.rollback()
            raise ValueError(f"Model version not found for ticker={ticker}: {model_version}")

        self.session.commit()
        
def get_model_registry_service(session: Session = Depends(get_session)) -> ModelRegistryService:
    return ModelRegistryService(session)