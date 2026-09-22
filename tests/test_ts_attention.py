"""Tests for token ordering, attention masks and Qwen integration."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from helpers import ForwardHarness
from layers.ts_mlp import build_ts_mlp
from utils.qwen3_vl_patch import patch_qwen3_vl_ts_attn_bias
from utils.ts_attention import (
    build_history_attention_mask,
    positions_from_padding_mask,
    reorder_ts_tokens,
    validate_checkpoint_attention,
)
from utils.ts_bias import build_ts_attention_bias


torch.set_num_threads(1)


def tiny_backbone():
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
    config = Qwen3VLConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    return Qwen3VLModel(config).eval()


class LayoutTests(unittest.TestCase):
    def test_explicit_order_roundtrip_and_gradients(self):
        x = torch.tensor([[[10.0], [11.0], [12.0], [20.0], [21.0], [22.0]]], requires_grad=True)
        y = reorder_ts_tokens(x, 2, 3, "channel_major", "patch_major")
        self.assertEqual(y.flatten().tolist(), [10, 20, 11, 21, 12, 22])
        torch.testing.assert_close(reorder_ts_tokens(y, 2, 3, "patch_major", "channel_major"), x)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_identity_embeddings_and_shared_projection_stay_attached(self):
        harness = ForwardHarness()
        harness.args = SimpleNamespace(
            seq_len=6,
            patch_len=2,
            stride=2,
            num_vars=2,
            use_time_features=True,
            time_embed_scale=1.0,
            ts_token_layout="patch_major",
        )
        harness.device = torch.device("cpu")
        harness.ts_mlp = build_ts_mlp(2, 4)
        harness.ts_var_embed = nn.Embedding(2, 4)
        harness.ts_time_embed = nn.Embedding(3, 4)
        x = torch.randn(2, 6, 2)
        patch = harness._build_ts_embeddings(x, True)
        from layers.ts_features import encode_ts_embeddings

        channel = harness._add_time_features(
            encode_ts_embeddings(x, harness.ts_mlp, 2, 2, harness.device, True)
        )
        for p in range(3):
            for c in range(2):
                torch.testing.assert_close(patch[:, p * 2 + c], channel[:, c * 3 + p])

    def test_relation_entries_match_channel_and_patch_identity(self):
        C, P, B = 3, 2, 2
        global_rel = torch.arange(B * C * C, dtype=torch.float32).reshape(B, C, C)
        local_rel = 100 + torch.arange(B * P * C * C, dtype=torch.float32).reshape(B, P, C, C)
        for layout in ("channel_major", "patch_major"):
            with self.subTest(layout=layout):
                bias = build_ts_attention_bias(
                    (global_rel, global_rel, global_rel),
                    (local_rel, local_rel, local_rel),
                    P,
                    C,
                    torch.device("cpu"),
                    torch.float32,
                    (1.0, 0.0, 0.0),
                    token_layout=layout,
                )
                def index(p, c):
                    return c * P + p if layout == "channel_major" else p * C + c
                for p in range(P):
                    for q in range(P):
                        for c in range(C):
                            for d in range(C):
                                expected = local_rel[:, p, c, d] if p == q else global_rel[:, c, d]
                                torch.testing.assert_close(
                                    bias[:, index(p, c), index(q, d)], expected
                                )

    def test_checkpoint_semantics_cannot_change_silently(self):
        patch_major = SimpleNamespace(
            ts_token_layout="patch_major", ts_attention_mode="history_bidirectional"
        )
        channel_major = SimpleNamespace(ts_token_layout="channel_major", ts_attention_mode="causal")
        with self.assertRaisesRegex(ValueError, "Checkpoint attention configuration mismatch"):
            validate_checkpoint_attention(patch_major, {})
        validate_checkpoint_attention(channel_major, {})
        validate_checkpoint_attention(patch_major, vars(patch_major))
        with self.assertRaises(ValueError):
            validate_checkpoint_attention(channel_major, vars(patch_major))


class MaskTests(unittest.TestCase):
    def test_visibility_prefix_padding_and_empty_prefix(self):
        for prefix in (0, 4):
            padding = torch.ones(2, prefix + 6, dtype=torch.long)
            if prefix:
                padding[0, 0] = 0
                padding[1, 1:3] = 0
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                with self.subTest(prefix=prefix, dtype=dtype):
                    mask = build_history_attention_mask(padding, (prefix, prefix + 5), dtype)
                    self.assertEqual(mask.shape, (2, 1, prefix + 6, prefix + 6))
                    for b in range(2):
                        for q in range(prefix + 6):
                            for k in range(prefix + 6):
                                if not padding[b, q]:
                                    expected = q == k
                                else:
                                    expected = bool(padding[b, k]) and (q >= prefix or k <= q)
                                self.assertEqual(bool(mask[b, 0, q, k] == 0), expected)
                    self.assertTrue(torch.isfinite(torch.softmax(mask.float(), dim=-1)).all())

    def test_rejects_invalid_layout_or_mask_shape(self):
        with self.assertRaises(ValueError):
            reorder_ts_tokens(torch.zeros(1, 5, 2), 2, 3, "channel_major", "patch_major")
        with self.assertRaises(ValueError):
            build_history_attention_mask(torch.ones(1, 6), (1, 4), torch.float32)


class QwenIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(17)
        cls.model = tiny_backbone()
        patch_qwen3_vl_ts_attn_bias()

    def setUp(self):
        self.harness = ForwardHarness()
        self.harness.test_backbone = self.model
        self.harness.args = SimpleNamespace(ts_attention_mode="history_bidirectional")
        torch.manual_seed(29)
        self.embeds = torch.randn(2, 9, 128)
        self.padding = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1, 1]])
        self.ts_range = (3, 8)

    def run_model(self, embeds=None, bias=None, grad=False):
        return self.harness._run_backbone(
            self.embeds if embeds is None else embeds,
            self.padding,
            grad,
            ts_attn_bias=bias,
            ts_attn_layer=[0, 1],
            ts_attn_bias_scale=torch.tensor(-2.97),
            ts_range=self.ts_range,
        )

    def test_actual_attention_visibility_in_all_layers_with_and_without_bias(self):
        for with_bias in (False, True):
            with self.subTest(bias=with_bias):
                observed = []
                handles = [
                    layer.self_attn.register_forward_hook(
                        lambda module, args, output: observed.append(output[1].detach())
                    )
                    for layer in self.model.language_model.layers
                ]
                bias = torch.zeros(2, 9, 9) if with_bias else None
                if bias is not None:
                    bias[:, 3:, 3:] = torch.randn(2, 6, 6) * 0.1
                try:
                    output = self.run_model(bias=bias)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(len(observed), 3)
                for weights in observed:
                    self.assertTrue((weights[:, :, 3, 8] > 0).all())
                    self.assertTrue((weights[:, :, 8, 3] > 0).all())
                    self.assertTrue((weights[:, :, 3:, 2] > 0).all())
                    self.assertEqual(torch.count_nonzero(weights[:, :, :3, 3:]).item(), 0)
                    self.assertEqual(torch.count_nonzero(weights[0, :, 1:, 0]).item(), 0)
                    self.assertEqual(torch.count_nonzero(weights[1, :, 2:, 1]).item(), 0)
                    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))

    def test_padding_does_not_change_valid_outputs_or_rope_positions(self):
        expected_positions, _ = self.model.get_rope_index(None, attention_mask=self.padding)
        torch.testing.assert_close(positions_from_padding_mask(self.padding), expected_positions)
        perturb = self.embeds.clone()
        perturb[0, 0] += 100 * torch.randn(128)
        perturb[1, 1] += 100 * torch.randn(128)
        torch.testing.assert_close(self.run_model()[:, 3:], self.run_model(perturb)[:, 3:])

    def test_backward_through_custom_mask_and_structural_bias(self):
        x = self.embeds.clone().requires_grad_(True)
        bias = torch.zeros(2, 9, 9, requires_grad=True)
        output = self.run_model(x, bias=bias, grad=True)
        output[:, 3:].square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad[:, 3:].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(bias.grad).all())
        self.assertGreater(bias.grad[:, 3:, 3:].abs().sum().item(), 0)

    def test_complete_forecasting_path_with_residual_and_modality_ablations(self):
        harness = self.harness
        harness.device = torch.device("cpu")
        harness.vlm = self.model
        harness.ts_mlp = build_ts_mlp(2, 128)
        harness.ts_time_embed = nn.Embedding(3, 128)
        harness.ts_var_embed = nn.Embedding(2, 128)
        harness.pred_head = nn.Linear(3 * 128, 2)
        harness.ts_normalizer = SimpleNamespace(normalize=lambda x: (x, None))
        harness.args = SimpleNamespace(
            seq_len=6,
            patch_len=2,
            stride=2,
            num_vars=2,
            use_time_features=True,
            time_embed_scale=1.0,
            use_ts_residual=True,
            ts_attn_bias_layers="0,1",
            ts_token_layout="patch_major",
            ts_attention_mode="history_bidirectional",
            ts_pooling="flatten",
            ts_residual_mode="add",
        )
        text = (torch.randn(2, 3, 128), torch.tensor([[1, 1, 0], [1, 0, 0]]))
        harness._encode_unified_text = lambda x: text
        inputs = dict(
            x=torch.randn(2, 6, 2),
            img_grids=torch.rand(2, 2, 2, 3),
            vid_grids=torch.rand(2, 3, 2, 2, 3),
            img_tokens=torch.randn(2, 2, 128),
            img_token_mask=torch.tensor([[1, 0], [1, 1]]),
            vid_tokens=torch.randn(2, 3, 128),
            vid_token_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        )
        for vision, use_text, bias in (
            (True, True, True),
            (False, True, True),
            (True, False, True),
            (True, True, False),
            (False, False, False),
        ):
            with self.subTest(vision=vision, text=use_text, bias=bias):
                harness.args.use_vision, harness.args.use_text, harness.args.use_ts_attn_bias = (
                    vision,
                    use_text,
                    bias,
                )
                harness.ts_mlp.zero_grad(set_to_none=True)
                harness.pred_head.zero_grad(set_to_none=True)
                self.model.zero_grad(set_to_none=True)
                predictions = harness(**inputs)
                self.assertEqual(predictions.shape, (2, 2, 2))
                self.assertTrue(torch.isfinite(predictions).all())
                predictions.square().mean().backward()
                for module in (harness.ts_mlp, harness.pred_head):
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)


if __name__ == "__main__":
    unittest.main()
