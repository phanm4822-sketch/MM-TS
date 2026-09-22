import torch
import torch.nn as nn


class ForecastNormalize(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.subtract_last = bool(subtract_last)

    def normalize(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            reference = x[:, -1:, :].detach()
        else:
            reference = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

        x_norm = x - reference
        x_norm = x_norm / stdev
        ctx = {
            "reference": reference,
            "stdev": stdev,
        }
        return x_norm, ctx

    def denormalize(self, x: torch.Tensor, ctx):
        if ctx is None:
            return x
        reference = ctx["reference"].to(device=x.device, dtype=x.dtype)
        stdev = ctx["stdev"].to(device=x.device, dtype=x.dtype)
        x_denorm = x
        x_denorm = x_denorm * stdev
        x_denorm = x_denorm + reference
        return x_denorm


def build_forecast_normalizer(
    num_features: int,
    eps: float = 1e-5,
    subtract_last: bool = False,
) -> ForecastNormalize:
    return ForecastNormalize(
        num_features=num_features,
        eps=eps,
        subtract_last=subtract_last,
    )
