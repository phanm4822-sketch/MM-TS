"""Tests for training, checkpoint loading and cache preparation."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration

from helpers import tiny_config
from data_provider.cache.common import build_relation_grids, build_split_ranges
from data_provider.cache import electricity, standard
from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from models.MMTS import Model
from utils.config import BASE_DEFAULTS


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_training_best_checkpoint_and_reload(self):
        with tempfile.TemporaryDirectory(prefix="mmts_test_") as directory:
            rng = np.random.default_rng(8)
            meta = dict(
                cache_protocol="global_standardize_fullfft_v5",
                seq_len=6,
                pred_len=2,
                patch_len=2,
                stride=2,
                num_vars=2,
                split_sizes=[4, 2, 2],
                source_dataset_name="synthetic",
            )
            data_path = str(Path(directory, "data.npz"))
            np.savez(
                data_path,
                x=rng.normal(size=(8, 6, 2)).astype("float32"),
                y=rng.normal(size=(8, 2, 2)).astype("float32"),
                img_grid=rng.random((8, 2, 2, 3)).astype("float32"),
                vid_grids=rng.random((8, 3, 2, 2, 3)).astype("float32"),
                meta_json=json.dumps(meta),
                mean=np.zeros((1, 2)),
                std=np.ones((1, 2)),
            )
            args = SimpleNamespace(
                **{
                    **BASE_DEFAULTS,
                    "root_path": directory,
                    "data_path": data_path,
                    "output_dir": directory,
                    "seq_len": 6,
                    "pred_len": 2,
                    "patch_len": 2,
                    "stride": 2,
                    "embed_dim": 128,
                    "batch_size": 2,
                    "epochs": 2,
                    "use_gpu": False,
                    "use_text": False,
                    "use_vision": False,
                    "lora_target_layers": "0,1",
                    "max_train_batches": 1,
                    "max_eval_batches": 1,
                }
            )
            with patch(
                "models.MMTS.load_qwen3_vl",
                return_value=(Qwen3VLForConditionalGeneration(tiny_config()), None),
            ):
                experiment = Exp_Main(args)
            experiment.train()
            checkpoint = Path(directory, "synthetic/checkpoints/best.latest.pt")
            self.assertTrue(checkpoint.is_file())
            before = experiment.test()
            with torch.no_grad():
                experiment.model.pred_head.weight.add_(10)
            experiment.load_checkpoint(str(checkpoint))
            self.assertEqual(before, experiment.test())
            config = experiment._collect_run_config()
            self.assertEqual(config["ts_token_layout"], "patch_major")
            self.assertEqual(config["prompt_template_version"], "mmts-window-v1")

    def test_frozen_backbone_still_propagates_input_and_bias_gradients(self):
        args = SimpleNamespace(
            **{
                **BASE_DEFAULTS,
                "seq_len": 6,
                "pred_len": 2,
                "patch_len": 2,
                "stride": 2,
                "embed_dim": 128,
                "num_vars": 2,
                "use_gpu": False,
                "use_text": False,
                "use_vision": False,
                "use_lora": False,
                "use_ts_residual": False,
            }
        )
        with patch(
            "models.MMTS.load_qwen3_vl",
            return_value=(Qwen3VLForConditionalGeneration(tiny_config()), None),
        ):
            model = Model(args, device="cpu")
        model(
            torch.randn(2, 6, 2), torch.rand(2, 2, 2, 3), torch.rand(2, 3, 2, 2, 3)
        ).square().mean().backward()
        self.assertGreater(model.ts_mlp[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(model.vlm.ts_bias_scale.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.language_model.parameters()))

    def test_npz_and_sharded_backends_deliver_the_same_windows(self):
        with tempfile.TemporaryDirectory(prefix="mmts_cache_test_") as directory:
            csv = Path(directory, "weather.csv")
            np.savetxt(
                csv,
                np.random.default_rng(15).normal(size=(100, 3)),
                delimiter=",",
                header="a,b,c",
                comments="",
            )
            paths = [Path(directory, "standard.npz"), Path(directory, "sharded")]
            for builder, output in zip((standard, electricity), paths):
                argv = [
                    "cache",
                    "--root_path",
                    directory,
                    "--data_path",
                    str(csv),
                    "--seq_len",
                    "8",
                    "--pred_len",
                    "2",
                    "--patch_len",
                    "4",
                    "--stride",
                    "2",
                    "--precompute_vision",
                    "false",
                    "--output_path",
                    str(output),
                    "--log_every",
                    "0",
                ]
                if builder is electricity:
                    argv += ["--build_workers", "1", "--shard_size", "20", "--cleanup", "false"]
                else:
                    argv += ["--auto_shard", "false"]
                with patch.object(sys, "argv", argv):
                    builder.main()
            args = SimpleNamespace(
                **{
                    **BASE_DEFAULTS,
                    "root_path": directory,
                    "seq_len": 8,
                    "pred_len": 2,
                    "patch_len": 4,
                    "stride": 2,
                    "use_vision": False,
                }
            )
            sets = []
            for path in [paths[0], paths[1] / "manifest.json"]:
                args.data_path = str(path)
                datasets = [data_provider(args, flag)[0] for flag in ["train", "val", "test"]]
                sets.append(datasets)
            for npz, shard in zip(*sets):
                self.assertEqual(len(npz), len(shard))
                for i in (0, len(npz) - 1):
                    for a, b in zip(npz[i], shard[i]):
                        # Contiguous shard windows and NPZ views can differ
                        # slightly in float32 reduction order.
                        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
                shard._cache.clear()
            electricity._FAST_MEMMAPS.clear()

    def test_split_targets_and_training_statistics_boundaries(self):
        args = SimpleNamespace(
            seq_len=96, pred_len=96, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
        )
        ranges, sizes, is_ett, _, train_end = build_split_ranges(args, "ETTh1.csv", 17420)
        self.assertEqual(sizes, [8449, 2785, 2785])
        self.assertTrue(is_ett)
        self.assertEqual(ranges[1][0] + args.seq_len, train_end)
        ranges, _, _, _, train_end = build_split_ranges(args, "weather.csv", 52696)
        self.assertEqual(ranges[1][0], train_end)
        args.seq_len, args.pred_len = 104, 24
        _, sizes, _, _, _ = build_split_ranges(args, "national_illness.csv", 966)
        self.assertEqual(sizes, [549, 74, 170])

    def test_relation_grids_are_finite_with_constant_channels(self):
        x = np.random.default_rng(18).normal(size=(12, 3)).astype("float32")
        x[:, 0] = 3.0
        img, vid = build_relation_grids(x, 4, 2)
        self.assertEqual(img.shape, (3, 3, 3))
        self.assertEqual(vid.shape, (5, 3, 3, 3))
        self.assertTrue(np.isfinite(img).all() and np.isfinite(vid).all())
        self.assertTrue(((img >= 0) & (img <= 1)).all())


if __name__ == "__main__":
    unittest.main()
