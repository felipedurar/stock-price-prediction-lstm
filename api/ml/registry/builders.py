

from api.ml.registry.schemas import ModelMetadata
from api.ml.architectures.lstm_regressor import LSTMRegressor


# def build_model_fn(metadata: dict):
#     model_type = metadata.get("model_type")
#     params = metadata.get("model_params", {})

#     if model_type == "lstm_regressor":
#         # defaults seguros
#         input_size = int(params.get("input_size", 1))
#         hidden_size = int(params.get("hidden_size", 64))
#         num_layers = int(params.get("num_layers", 2))
#         dropout = float(params.get("dropout", 0.0))

#         return LSTMRegressor(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout,
#         )

#     raise ValueError(f"Unknown model_type: {model_type}")


def build_model_from_metadata(meta: ModelMetadata):
    p = meta.model_params
    return LSTMRegressor(
        input_size=p.input_size,
        hidden_size=p.hidden_size,
        num_layers=p.num_layers,
        dropout=p.dropout,
    )