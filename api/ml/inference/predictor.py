import torch
import time
from api.monitoring.prometheus import (
    MODEL_PREDICTIONS_TOTAL,
    MODEL_INFERENCE_TIME,
    LAST_PREDICTED_VALUE
)

def predict_and_inverse(model, X, scaler, device=None):
    # start time for metrics
    start_time = time.time()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_scaled = model(X_t).cpu().numpy()  # (N,1)

    pred_real = scaler.inverse_transform(pred_scaled)

    # Metrics

    inference_time = time.time() - start_time

    MODEL_PREDICTIONS_TOTAL.inc()
    MODEL_INFERENCE_TIME.observe(inference_time)
    LAST_PREDICTED_VALUE.set(float(pred_real[0, 0]))

    return pred_real

def inverse_y(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled)