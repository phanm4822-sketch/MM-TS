# Sources and external dependencies

* [PatchTST supervised source](https://github.com/yuqinie98/PatchTST/tree/main/PatchTST_supervised): structural reference for model, experiment, layers and data-provider separation.
* [iTransformer](https://github.com/thuml/iTransformer): additional structural reference.
* [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and [the 2B Instruct checkpoint](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct): pretrained backbone and processor.
* [Transformers 4.57.3](https://github.com/huggingface/transformers/tree/v4.57.3): Qwen model implementation and tokenizer/processor APIs. `utils/qwen3_vl_patch.py` follows its eager attention path and adds the structured bias.
* [PEFT](https://github.com/huggingface/peft): LoRA adapters.
* [PyTorch](https://github.com/pytorch/pytorch), [NumPy](https://github.com/numpy/numpy), [pandas](https://github.com/pandas-dev/pandas), and [Pillow](https://github.com/python-pillow/Pillow): tensor computation, data loading and image rendering.

Upstream repositories and model cards provide their current license texts. This
source distribution does not bundle pretrained weights, datasets or baseline
repositories and does not relicense those external assets.
