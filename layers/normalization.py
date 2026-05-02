import torch
import torch.nn as nn


class ForecastNormalize(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        self.non_norm = bool(non_norm)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def normalize(self, x: torch.Tensor):
        if self.non_norm:
            return x, None
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            reference = x[:, -1:, :].detach()
        else:
            reference = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

        x_norm = x - reference
        x_norm = x_norm / stdev
        if self.affine:
            weight = self.affine_weight.view(*([1] * (x.ndim - 1)), -1).to(device=x.device, dtype=x.dtype)
            bias = self.affine_bias.view(*([1] * (x.ndim - 1)), -1).to(device=x.device, dtype=x.dtype)
            x_norm = x_norm * weight + bias
        ctx = {
            "reference": reference,
            "stdev": stdev,
        }
        return x_norm, ctx

    def denormalize(self, x: torch.Tensor, ctx):
        if self.non_norm or ctx is None:
            return x
        reference = ctx["reference"].to(device=x.device, dtype=x.dtype)
        stdev = ctx["stdev"].to(device=x.device, dtype=x.dtype)
        x_denorm = x
        if self.affine:
            weight = self.affine_weight.view(*([1] * (x.ndim - 1)), -1).to(device=x.device, dtype=x.dtype)
            bias = self.affine_bias.view(*([1] * (x.ndim - 1)), -1).to(device=x.device, dtype=x.dtype)
            x_denorm = x_denorm - bias
            x_denorm = x_denorm / (weight + self.eps * self.eps)
        x_denorm = x_denorm * stdev
        x_denorm = x_denorm + reference
        return x_denorm


def build_forecast_normalizer(
    num_features: int,
    eps: float = 1e-5,
    affine: bool = False,
    subtract_last: bool = False,
    non_norm: bool = False,
) -> ForecastNormalize:
    return ForecastNormalize(
        num_features=num_features,
        eps=eps,
        affine=affine,
        subtract_last=subtract_last,
        non_norm=non_norm,
    )
