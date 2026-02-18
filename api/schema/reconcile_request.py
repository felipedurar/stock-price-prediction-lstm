from pydantic import BaseModel


class ReconcileRequest(BaseModel):
    ticker: str
