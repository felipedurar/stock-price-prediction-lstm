# Stock Price Prediction using LSTM
Predict the next closing price of a stock using a Long Short-Term Memory (LSTM) neural network and serve the model through a FastAPI REST API with model versioning.

This project was developed as part of the FIAP Postgraduate – Tech Challenge (Phase 4).

## Overview
This application implements an end-to-end machine learning pipeline:
- Data ingestion
- Model training
- Model versioning
- Inference API

## Workflow
Workflow for generating new predictions using the Inference API:
![Stock Price Workflow](assets/workflow.drawio.png)

## Tech Stack
- FastAPI
- PyTorch
- SQLModel / SQLite
- Poetry
- Docker & Docker Compose
- yfinance, pandas, numpy

## How to Run
1. Install dependencies (Poetry)
```
poetry install
poetry run uvicorn api.main:app --reload
```
API will be available at:
```
http://localhost:8000/api/v1/docs
```

2. Run with Docker
```
docker-compose up --build
```

## Model Details
- Architecture: LSTM → Linear
- Input: last lookback closing prices
- Output: next closing price (horizon = 1)
- Scaling: MinMaxScaler (fit on training set only)
- Loss: MSE
- Evaluation metrics:
    - MAE
    - RMSE
    - MAPE

## Author
- Felipe Malaquias Durar
- Everton Vieira Rodrigues

## License
Educational project for FIAP Tech Challenge.