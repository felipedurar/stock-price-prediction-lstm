
from api.services.stock_data_ingestion_service import sync_stock_data

def perform_initial_reconciliation():
    print("Performing Initial Reconciliation...")
    sync_stock_data()
    print("Finished Performing Stock Data Ingestion")
