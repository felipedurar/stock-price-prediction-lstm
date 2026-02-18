
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_data(closes: np.ndarray, lookback: int = 60, train_ratio: float = 0.8):
    """
    closes: array 1D com valores de close em ordem temporal (antigo -> recente)
    """
    closes = closes.astype(np.float32).reshape(-1, 1)

    train_raw, val_raw = train_val_split_time(closes, train_ratio=train_ratio)

    scaler = MinMaxScaler()
    scaler.fit(train_raw)  # ✅ fit só no treino

    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)

    X_train, y_train = make_sequences(train_scaled, lookback)
    X_val, y_val = make_sequences(val_scaled, lookback)

    return X_train, y_train, X_val, y_val, scaler

def make_sequences(series: np.ndarray, lookback: int):
    """
    series: shape (N, 1) já normalizada
    returns:
      X: (N-lookback, lookback, 1)
      y: (N-lookback, 1)
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_val_split_time(arr: np.ndarray, train_ratio: float = 0.8):
    n = len(arr)
    cut = int(n * train_ratio)
    return arr[:cut], arr[cut:]