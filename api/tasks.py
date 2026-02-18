
from api.services.stock_data_ingestion_service import ingest_stock_data

def perform_initial_reconciliation():
    print("Performing Initial Reconciliation...")
    ingest_stock_data("AAPL")
    print("Finished Performing Stock Data Ingestion")
