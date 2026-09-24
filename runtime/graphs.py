"""CUDA graphs with CPU prompt/relation preparation outside capture."""

import inspect
from contextlib import contextmanager

import torch


class _Prepared(torch.nn.Module):
    def __init__(self, model, original, fields, text, bias):
        super().__init__()
        self.network = model
        self.original = original
        self.fields = fields
        self.text, self.bias = text, bias

    def forward(self, *inputs):
        count = len(self.fields)
        kwargs = dict(zip(self.fields, inputs[:count]))
        model = self.network
        old_text, old_stats = model._encode_unified_text, model._relation_stats
        if self.text:
            pair = inputs[count : count + 2]
            count += 2
            model._encode_unified_text = lambda x: pair
        if self.bias:
            image, video = inputs[count : count + 3], inputs[count + 3 : count + 6]

            def stats(grid):
                if grid.ndim == 4:
                    return image
                if grid.ndim == 5:
                    return video
                raise ValueError("relation grids must have rank 4 or 5")

            model._relation_stats = stats
        try:
            return self.original(**kwargs)
        finally:
            model._encode_unified_text, model._relation_stats = old_text, old_stats


@contextmanager
def graphed_core(model):
    """Keep at most one graph per mode; preserve tail batches and autocast.

    Capture is lazy, after channel-dependent modules have been initialized.
    Training capture preserves RNG so dropout starts at the same generator
    state as eager execution. No optimizer update runs during capture.
    """
    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError(
            "CUDA graph execution is validated on PyTorch 2.9.1; disable cuda_graphs on other versions"
        )
    original = model.forward
    signature = inspect.signature(original)
    holders = {}
    stats = {"captures": 0, "replays": 0, "eager_shape_fallbacks": 0}

    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        kwargs = {k: v for k, v in bound.arguments.items() if v is not None}
        training = bool(model.training and torch.is_grad_enabled())
        # eval() alone does not disable autograd outside the backbone.
        if not training and torch.is_grad_enabled():
            return original(**kwargs)
        if any(not isinstance(v, torch.Tensor) for v in kwargs.values()):
            return original(**kwargs)
        kwargs = {
            k: v if k.endswith("_grids") else v.to(model.device)
            for k, v in kwargs.items()
        }
        fields = tuple(kwargs)
        text, bias = model.args.use_text, model.args.use_ts_attn_bias
        extra = []
        with torch.no_grad():
            if text:
                aux, _ = model.ts_normalizer.normalize(kwargs["x"])
                extra.extend(model._encode_unified_text(aux))
            if bias:
                extra.extend(model._relation_stats(kwargs["img_grids"]))
                extra.extend(model._relation_stats(kwargs["vid_grids"]))
        inputs = tuple(kwargs.values()) + tuple(
            torch.as_tensor(x, device=model.device) for x in extra
        )
        amp_enabled = torch.is_autocast_enabled("cuda")
        amp_dtype = torch.get_autocast_dtype("cuda")
        key = (
            fields,
            tuple((tuple(x.shape), x.dtype, x.requires_grad, x.device) for x in inputs),
            amp_enabled,
            amp_dtype,
            torch.is_inference_mode_enabled(),
        )
        holder = holders.get(training)
        if holder is not None and key != holder["key"]:
            stats["eager_shape_fallbacks"] += 1
            return original(**kwargs)
        if holder is None:
            module = _Prepared(model, original, fields, text, bias)
            static = tuple(
                x.detach().clone().requires_grad_(x.requires_grad) for x in inputs
            )
            # Match the caller's precision; disabling the autocast weight cache
            # is required by make_graphed_callables, not a precision change.
            with torch.cuda.device(model.device), torch.autocast(
                "cuda", enabled=amp_enabled, dtype=amp_dtype, cache_enabled=False
            ):
                if training:
                    with torch.random.fork_rng(devices=[model.device.index or 0]):
                        graph = torch.cuda.make_graphed_callables(
                            module, static, num_warmup_iters=3, allow_unused_input=True
                        )
                    holder = dict(key=key, graph=graph)
                else:
                    stream = torch.cuda.Stream(device=model.device)
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            output = module(*static)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = module(*static)
                    holder = dict(key=key, graph=graph, static=static, output=output)
            holders[training] = holder
            stats["captures"] += 1
        stats["replays"] += 1
        if training:
            return holder["graph"](*inputs)
        for dst, src in zip(holder["static"], inputs):
            dst.copy_(src)
        holder["graph"].replay()
        return holder["output"]

    model.forward = call
    try:
        yield stats
    finally:
        model.forward = original
        holders.clear()
