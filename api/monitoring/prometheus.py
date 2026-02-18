from prometheus_client import Counter, Histogram, Gauge

# API Metrics

API_REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests"
)

API_REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1,2, 5)
)

# Model Metrics

MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total number of model predictions"
)

MODEL_INFERENCE_TIME = Histogram(
    "model_inference_seconds",
    "Time spent in model inference",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1)
)

LAST_PREDICTED_VALUE = Gauge(
    "model_last_prediction_value",
    "Last predicted stock price"
)


