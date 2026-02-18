from fastapi import APIRouter, Depends, Path, Query

from api.schema.reconcile_request import ReconcileRequest
from api.services.stock_data_ingestion_service import ingest_stock_data

router = APIRouter()

@router.post(
    "/reconcile",
    summary="Realiza a reconciliação dos dados de ações da bolsa"
)
async def data_reconcile(req: ReconcileRequest):
    """
    Realiza a reconciliação dos dados de ações da bolsa
    """
    print("Performing Reconciliation...")
    ingest_stock_data(req.ticker)
    print("Finished Performing Stock Data Ingestion")
    return { "ok": True }
