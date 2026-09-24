"""Optional runtime acceleration without changing the forecasting recipe."""

from .optimization import optimized_runtime
from .online import predict_window

__all__ = ["optimized_runtime", "predict_window"]
