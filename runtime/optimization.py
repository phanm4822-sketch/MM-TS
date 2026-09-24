"""Reversible, single-model runtime optimizations for MM-TS.

Enter after loading the backbone; load/save checkpoints inside or outside the
context. Do not move the model, replace parameters, or use it concurrently from
another thread while the context is active.
"""

from contextlib import ExitStack, contextmanager

import torch


@contextmanager
def _offload(module):
    devices = {p.device for p in module.parameters()}
    if len(devices) != 1 or any(p.requires_grad for p in module.parameters()):
        raise ValueError("offload requires a frozen module on one device")
    device = next(iter(devices))
    module.to("cpu")
    try:
        yield
    finally:
        module.to(device)


@contextmanager
def _unused_deepstack(visual):
    """Skip discarded DeepStack outputs, preserving the checkpoint schema.

    Keep inactive weights on CPU for strict round-trip checkpoint compatibility;
    they are not part of the active forecasting module's parameter iterator.
    The tied vocabulary head remains an alias of the prompt embedding.
    """
    mergers = visual.deepstack_merger_list
    indexes = visual.deepstack_visual_indexes
    if any(p.requires_grad for p in mergers.parameters()):
        raise ValueError("DeepStack skipping requires frozen visual weights")
    devices = {p.device for p in mergers.parameters()}
    if len(devices) > 1:
        raise ValueError("DeepStack skipping requires a single device")
    device = next(iter(devices), torch.device("cpu"))
    mergers.to("cpu")
    visual.deepstack_merger_list = torch.nn.ModuleList()
    visual.deepstack_visual_indexes = []

    def save(module, state, prefix, metadata):
        mergers.state_dict(destination=state, prefix=prefix + "deepstack_merger_list.")

    def load(module, state, prefix, metadata, strict, missing, unexpected, errors):
        stem = prefix + "deepstack_merger_list."
        values = {
            k[len(stem) :]: state.pop(k) for k in list(state) if k.startswith(stem)
        }
        try:
            mergers.load_state_dict(values, strict=True)
        except RuntimeError as exc:
            errors.append(str(exc))

    hooks = [
        visual.register_state_dict_post_hook(save),
        visual.register_load_state_dict_pre_hook(load),
    ]
    try:
        yield sum(p.numel() for p in mergers.parameters())
    finally:
        for hook in hooks:
            hook.remove()
        visual.deepstack_merger_list = mergers.to(device)
        visual.deepstack_visual_indexes = indexes


@contextmanager
def optimized_runtime(
    model, *, cached_visual=True, cuda_graphs=False, fast_vision=False
):
    """Optimize one model while preserving its precision and training settings.

    cached_visual=True offloads the unused visual encoder for cached training or
    evaluation. Set it to False for predict_window. fast_vision additionally
    enables the opt-in, version-guarded frozen-vision CUDA implementation.
    Graphs keep one shape per train/eval mode; other shapes run eagerly.
    """
    if getattr(model.args, "model_parallel", False):
        raise ValueError("optimized_runtime supports single-device models only")
    if fast_vision and cached_visual:
        raise ValueError("fast_vision is for online prediction, not cached inputs")
    if getattr(model, "_optimized_runtime_active", False):
        raise ValueError("optimized_runtime cannot be nested for the same model")
    if (cuda_graphs or fast_vision) and model.device.type != "cuda":
        raise ValueError("CUDA acceleration requires a CUDA model")
    model._optimized_runtime_active = True
    info = {"cached_visual": cached_visual, "cuda_graphs": cuda_graphs}
    try:
        with ExitStack() as stack:
            visual = model.backbone.visual
            info["inactive_parameters"] = stack.enter_context(_unused_deepstack(visual))
            embedding = model.backbone.language_model.embed_tokens
            if not any(p.requires_grad for p in embedding.parameters()):
                stack.enter_context(_offload(embedding))
            if cached_visual:
                stack.enter_context(_offload(visual))
            else:
                from .relations import accelerated_relations

                stack.enter_context(accelerated_relations())
                if fast_vision and model.args.use_vision:
                    from .patch_embedding import exact_patch_embedding
                    from .vision_graph import graphed_visual

                    stack.enter_context(exact_patch_embedding(visual))
                    stack.enter_context(graphed_visual(visual))
            if cuda_graphs:
                from .graphs import graphed_core

                info["graphs"] = stack.enter_context(graphed_core(model))
            yield info
    finally:
        del model._optimized_runtime_active
