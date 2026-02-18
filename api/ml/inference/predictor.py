import torch

def predict_and_inverse(model, X, scaler, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_scaled = model(X_t).cpu().numpy()  # (N,1)

    pred_real = scaler.inverse_transform(pred_scaled)
    return pred_real

def inverse_y(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled)