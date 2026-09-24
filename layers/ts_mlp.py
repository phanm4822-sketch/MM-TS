from torch import nn


def build_ts_mlp(patch_len: int, embed_dim: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(patch_len, embed_dim),
        nn.GELU(),
        nn.Linear(embed_dim, embed_dim),
    )
