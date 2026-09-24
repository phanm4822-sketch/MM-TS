"""Visual CUDA graphs indexed by input grid and tensor shape."""

from contextlib import contextmanager
from itertools import accumulate

import torch
from .vision_attention import batched_visual_attention


class FrozenVisionGraph:
    def __init__(self, visual):
        self.visual = visual
        self.graphs = {}
        self.original = visual.forward

    def __call__(self, hidden_states, grid_thw, **kwargs):
        if torch.is_grad_enabled() or kwargs:
            return self.original(hidden_states, grid_thw, **kwargs)
        grid = tuple(
            tuple(int(v) for v in row) for row in grid_thw.detach().cpu().tolist()
        )
        key = (
            grid,
            tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.device,
        )
        if key not in self.graphs and len(self.graphs) >= 2:
            return self.original(hidden_states, grid_thw, **kwargs)
        if key not in self.graphs:
            v = self.visual
            pos = v.fast_pos_embed_interpolate(grid_thw)
            rot = v.rot_pos_emb(grid_thw)
            emb = torch.cat((rot, rot), dim=-1)
            positions = (emb.cos(), emb.sin())
            lengths = [h * w for t, h, w in grid for _ in range(t)]
            cu = torch.tensor(
                [0] + list(accumulate(lengths)),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            for block in v.blocks:
                block.attn._cache_segment_lengths = lengths
            static = hidden_states.clone()

            def body():
                z = v.patch_embed(static)
                z = z + pos
                z = z.reshape(z.shape[0], -1)
                deep = []
                for i, block in enumerate(v.blocks):
                    z = block(z, cu_seqlens=cu, position_embeddings=positions)
                    if i in v.deepstack_visual_indexes:
                        deep.append(
                            v.deepstack_merger_list[
                                v.deepstack_visual_indexes.index(i)
                            ](z)
                        )
                return v.merger(z), deep

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = body()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = body()
            self.graphs[key] = (graph, static, output, (pos, positions, cu))
        graph, static, output, _ = self.graphs[key]
        static.copy_(hidden_states)
        graph.replay()
        return output


@contextmanager
def graphed_visual(visual):
    with batched_visual_attention(visual):
        # Prepare segment lengths before entering the captured forward pass.
        wrapper = FrozenVisionGraph(visual)
        original = visual.forward
        visual.forward = wrapper
        try:
            yield wrapper
        finally:
            visual.forward = original
            wrapper.graphs.clear()
