"""CPU equivalence and checkpoint compatibility checks (no model download)."""

import unittest

import numpy as np
import torch

from data_provider.cache.common import build_relation_grids
from runtime.optimization import _unused_deepstack
from runtime.relations import accelerated_relations
from utils.ts_bias import build_ts_attention_bias, _combine_stats


class RuntimeTests(unittest.TestCase):
    def test_relations_preserve_complete_grids(self):
        rng = np.random.default_rng(2026)
        for channels in (2, 7, 21):
            for constant in (False, True):
                window = rng.normal(size=(96, channels)).astype(np.float32)
                if constant:
                    window[:, 0] = 1
                expected = build_relation_grids(window, 16, 8)
                with accelerated_relations():
                    actual = build_relation_grids(window, 16, 8)
                for a, b in zip(actual, expected):
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertEqual(a.tobytes(), b.tobytes())

    def test_bias_matches_reference_blocks_and_gradients(self):
        torch.manual_seed(2026)
        for layout in ("channel_major", "patch_major"):
            for dtype in (torch.float32, torch.bfloat16):
                patches, channels = 3, 7
                image = tuple(
                    torch.randn(2, channels, channels, dtype=dtype, requires_grad=True)
                    for _ in range(3)
                )
                video = tuple(
                    torch.randn(
                        2, patches, channels, channels, dtype=dtype, requires_grad=True
                    )
                    for _ in range(3)
                )
                weights = (2.5, 2.0, 2.5)
                img = _combine_stats(*image, weights, device="cpu", dtype=dtype)
                vid = _combine_stats(*video, weights, device="cpu", dtype=dtype)
                reference = torch.empty(
                    2, patches * channels, patches * channels, dtype=dtype
                )
                indices = [
                    (
                        torch.arange(channels) * patches + p
                        if layout == "channel_major"
                        else p * channels + torch.arange(channels)
                    )
                    for p in range(patches)
                ]
                for i in range(patches):
                    for j in range(patches):
                        reference[:, indices[i][:, None], indices[j][None, :]] = (
                            vid[:, i] if i == j else img
                        )
                actual = build_ts_attention_bias(
                    image, video, patches, channels, "cpu", dtype, weights, layout
                )
                self.assertTrue(torch.equal(actual, reference))
                old_grad = torch.autograd.grad(
                    reference.float().sum(), image + video, retain_graph=True
                )
                new_grad = torch.autograd.grad(actual.float().sum(), image + video)
                for a, b in zip(old_grad, new_grad):
                    self.assertTrue(torch.equal(a, b))

    def test_checkpoint_schema_survives_inactive_modules(self):
        visual = torch.nn.Module()
        visual.deepstack_merger_list = torch.nn.ModuleList(
            [torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)]
        )
        visual.deepstack_visual_indexes = [1, 2]
        visual.requires_grad_(False)
        original = {k: v.clone() for k, v in visual.state_dict().items()}
        with _unused_deepstack(visual):
            self.assertEqual(list(visual.parameters()), [])
            self.assertEqual(set(visual.state_dict()), set(original))
            visual.load_state_dict(original, strict=True)
            changed = {k: v + 1 for k, v in original.items()}
            visual.load_state_dict(changed, strict=True)
            for k, v in visual.state_dict().items():
                self.assertTrue(torch.equal(v, changed[k]))
            with self.assertRaises(RuntimeError):
                visual.load_state_dict({}, strict=True)
        for k, v in visual.state_dict().items():
            self.assertTrue(torch.equal(v, changed[k]))
        self.assertEqual(visual.deepstack_visual_indexes, [1, 2])


if __name__ == "__main__":
    unittest.main()
