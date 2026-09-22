from torch import nn
from models.MMTS import Model


class ForwardHarness(Model):
    def __init__(self):
        nn.Module.__init__(self)
        self._ts_embed_cache = {}

    @property
    def backbone(self):
        return self.test_backbone


def tiny_config():
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    text = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        attention_dropout=0.0,
        rope_scaling={"rope_type": "default", "mrope_section": [8, 12, 12]},
    )
    vision = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=128,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    cfg = Qwen3VLConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
    cfg._attn_implementation = cfg.text_config._attn_implementation = (
        cfg.vision_config._attn_implementation
    ) = "eager"
    return cfg
