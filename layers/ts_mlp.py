import torch
import torch.nn as nn


class PerChannelTSMLP(nn.Module):
    def __init__(self, num_vars: int, patch_len: int, embed_dim: int = 2048) -> None:
        super().__init__()
        if num_vars <= 0:
            raise ValueError(f"num_vars must be > 0, got {num_vars}")
        self.num_vars = int(num_vars)
        self.patch_len = int(patch_len)
        self.embed_dim = int(embed_dim)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.patch_len, self.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                )
                for _ in range(self.num_vars)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C*num_patches, patch_len]
        if x.dim() != 3:
            raise ValueError(f"expected 3D tensor [B, C*num_patches, patch_len], got {tuple(x.shape)}")
        bsz, total_tokens, patch_len = x.shape
        if patch_len != self.patch_len:
            raise ValueError(f"patch_len mismatch: got {patch_len}, expected {self.patch_len}")
        if total_tokens % self.num_vars != 0:
            raise ValueError(
                f"token count {total_tokens} not divisible by num_vars {self.num_vars}"
            )
        num_patches = total_tokens // self.num_vars
        x = x.view(bsz, self.num_vars, num_patches, patch_len)
        outs = []
        for idx, mlp in enumerate(self.mlps):
            out = mlp(x[:, idx, :, :])
            outs.append(out)
        out = torch.stack(outs, dim=1)
        return out.view(bsz, total_tokens, self.embed_dim)


def build_ts_mlp(
    patch_len: int,
    embed_dim: int = 2048,
    mode: str = "shared",
    num_vars: int | None = None,
) -> nn.Module:
    mode = str(mode).strip().lower()
    if mode == "shared":
        return nn.Sequential(
            nn.Linear(patch_len, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    if mode == "per_channel":
        if num_vars is None or int(num_vars) <= 0:
            raise ValueError("num_vars must be provided for per_channel TS MLP")
        return PerChannelTSMLP(int(num_vars), patch_len, embed_dim)
    raise ValueError(f"unknown ts_mlp_mode: {mode}")
