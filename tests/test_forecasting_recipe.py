"""Tests for the forecasting head, signed bias and prompt tokenization."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from exp.exp_main import Exp_Main
from helpers import ForwardHarness
from layers.ts_features import decode_ts_predictions
from utils.config import BASE_DEFAULTS, MODEL_RECIPE, build_parser
from models.qwen3_vl_utils import encode_text_batch
from utils.prompts import UNIFIED_PROMPT_VERSION, build_unified_prompts
from utils.ts_attention import reorder_ts_tokens, validate_checkpoint_forecasting
from utils.ts_bias import _combine_stats


class ForecastingRecipeTests(unittest.TestCase):
    def test_flatten_preserves_every_patch_and_channel_in_both_layouts(self):
        cm = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]
        )
        expected = torch.tensor(
            [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]]
        )
        for layout in ("channel_major", "patch_major"):
            hidden = reorder_ts_tokens(cm, 2, 3, "channel_major", layout)
            got = decode_ts_predictions(
                hidden, (0, 5), 2, 3, nn.Identity(), False, token_layout=layout
            )
            torch.testing.assert_close(got, expected)

    def test_residual_preserves_sign_and_head_reads_temporal_tokens(self):
        h = ForwardHarness()
        h.args = SimpleNamespace(
            **{
                **BASE_DEFAULTS,
                "seq_len": 6,
                "patch_len": 2,
                "stride": 2,
                "num_vars": 2,
                "embed_dim": 4,
                "pred_len": 2,
                "use_time_features": False,
                "use_vision": False,
                "use_text": False,
                "use_ts_attn_bias": False,
            }
        )
        h.device = torch.device("cpu")
        h.ts_mlp, h.pred_head = h._build_ts_modules(h.device, torch.float32)
        self.assertIsInstance(h.pred_head, nn.Linear)
        self.assertEqual(h.pred_head.in_features, 12)
        h.vlm = nn.Linear(1, 1)
        h.test_backbone = nn.Identity()
        h.ts_normalizer = SimpleNamespace(normalize=lambda x: (x, None))
        ts = torch.arange(1.0, 25.0).reshape(1, 6, 4)
        h._build_ts_embeddings = lambda x, train: ts
        h._run_backbone = lambda fused_embeds, **kwargs: -2 * fused_embeds
        with torch.no_grad():
            h.pred_head.weight.fill_(1.0)
            h.pred_head.bias.zero_()
        got = h(torch.zeros(1, 6, 2))
        # PM channel 0: tokens 0,2,4; channel 1: tokens 1,3,5.
        expected = torch.tensor([[[-126.0, -174.0], [-126.0, -174.0]]])
        torch.testing.assert_close(got, expected)

    def test_signed_mixture_matches_paper_without_recentering_or_squashing(self):
        dtw = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
        cov = torch.tensor([[[-0.2, 0.8], [0.8, -0.2]]])
        pear = torch.tensor([[[1.0, -0.6], [-0.6, 1.0]]])
        got = _combine_stats(dtw, cov, pear, (1.0, 0.5, 2.0), torch.device("cpu"), torch.float32)
        expected = torch.tensor([[[2.9, -0.3], [-0.3, 2.9]]])
        torch.testing.assert_close(got, expected)

    def test_cli_defaults_match_the_paper_recipe(self):
        args = build_parser(BASE_DEFAULTS, "test").parse_args([])
        self.assertTrue(args.use_ts_residual)
        self.assertEqual(MODEL_RECIPE["ts_pooling"], "flatten")
        self.assertEqual(MODEL_RECIPE["ts_residual_mode"], "add")
        self.assertEqual(args.prompt_max_tokens, 192)
        for obsolete in (
            "ts_pooling",
            "ts_residual_alpha",
            "prompt_style",
            "ts_tail_k",
            "train_ts_mlp",
        ):
            self.assertFalse(hasattr(args, obsolete))

    def test_checkpoint_records_prompt_version_and_rejects_incompatible_recipe(self):
        h = Exp_Main.__new__(Exp_Main)
        h.args = SimpleNamespace(**BASE_DEFAULTS)
        saved = h._collect_run_config()
        self.assertEqual(saved["prompt_template_version"], UNIFIED_PROMPT_VERSION)
        validate_checkpoint_forecasting(h.args, saved)
        for name, old in (
            ("ts_pooling", "mean"),
            ("ts_residual_mode", "B"),
            ("pred_context_mode", "all"),
            ("ts_bias_norm", "per_stat_tanh"),
            ("prompt_style", "sample"),
            ("prompt_template_version", "old"),
        ):
            with self.subTest(field=name):
                with self.assertRaisesRegex(
                    ValueError, "Checkpoint forecasting configuration mismatch"
                ):
                    validate_checkpoint_forecasting(h.args, {**saved, name: old})

    def test_prompt_describes_each_current_window_and_correct_relations(self):
        windows = torch.stack((torch.arange(8.0).repeat(2, 1).T, -torch.arange(8.0).repeat(2, 1).T))
        texts = build_unified_prompts(windows, "ETTh1", 96)
        self.assertIn("sampling=hourly; channels=2", texts[0])
        self.assertIn("trend=rising", texts[0])
        self.assertIn("trend=falling", texts[1])
        self.assertIn("DTW similarity", texts[0])
        self.assertNotIn("DTW distance", texts[0])
        self.assertNotIn("immediately before", texts[0])
        self.assertEqual(texts[0].count("Forecast the next"), 1)


@unittest.skipUnless(
    os.environ.get("MMTS_TOKENIZER_DIR"), "set MMTS_TOKENIZER_DIR for real-tokenizer checks"
)
class QwenPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ["MMTS_TOKENIZER_DIR"], local_files_only=True
        )
        embedding = nn.Embedding(len(cls.tokenizer), 4)
        cls.model = SimpleNamespace(model=SimpleNamespace(get_input_embeddings=lambda: embedding))
        cls.processor = SimpleNamespace(tokenizer=cls.tokenizer)

    def test_total_budget_and_saved_retained_text_with_actual_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            h = ForwardHarness()
            h.args = SimpleNamespace(**{**BASE_DEFAULTS, "output_dir": directory, "pred_len": 96})
            h.dataset_name = "ETTh1"
            h.device = torch.device("cpu")
            h.vlm, h.processor = self.model, self.processor
            tokens, mask = h._encode_unified_text(torch.zeros(2, 96, 7))
            self.assertEqual(tokens.shape, (2, 192, 4))
            self.assertTrue((mask.sum(1) <= 192).all())
            record = json.loads((Path(directory) / "prompt_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(record["allocated_text_tokens"], 192)
            for example in record["examples"]:
                self.assertFalse(example["truncated"])
                self.assertEqual(example["prompt"], example["encoded_text"])
                self.assertEqual(len(example["input_ids"]), example["effective_token_count"])

    def test_truncation_records_actual_tokens_and_dynamic_padding_stays_within_budget(self):
        prompts = ["Forecast " * 80, "Forecast next steps."]
        tokens, mask, metadata = encode_text_batch(
            self.model,
            self.processor,
            prompts,
            "cpu",
            max_length=12,
            pad=False,
            truncate=True,
            return_metadata=True,
        )
        self.assertEqual(tokens.shape[1], 12)
        self.assertTrue(metadata[0]["truncated"])
        self.assertFalse(metadata[1]["truncated"])
        for i, entry in enumerate(metadata):
            expected_ids = self.tokenizer(prompts[i], truncation=True, max_length=12)["input_ids"]
            self.assertEqual(entry["input_ids"], expected_ids)
            self.assertEqual(entry["encoded_text"], self.tokenizer.decode(expected_ids))
            self.assertEqual(int(mask[i].sum()), len(expected_ids))


if __name__ == "__main__":
    unittest.main()
