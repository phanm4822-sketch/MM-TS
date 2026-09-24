import argparse

BASE_DEFAULTS = {
    "root_path": ".",
    "data_path": "datasets/ETTh1.csv",
    "cache_dir": "cache",
    "output_dir": "runs",
    "qwen_dir": "Qwen3-VL-2B-Instruct",
    "gpus": "0",
    "model_parallel": False,
    "optimize_runtime": False,
    "cuda_graphs": False,
    "eval_only": False,
    "ckpt_path": "",
    "save_checkpoint": True,
    "rebuild_cache": False,
    "cache_batch_size": 8,
    "vision_mb": 8,
    "vision_render_size": 64,
    "seq_len": 96,
    "pred_len": 96,
    "patch_len": 16,
    "stride": 8,
    "batch_size": 32,
    "num_workers": 0,
    "embed_dim": 2048,
    "epochs": 20,
    "patience": 3,
    "lr": 2e-4,
    "lr_schedule": "none",
    "min_lr_ratio": 0.1,
    "weight_decay": 1e-4,
    "log_interval": 10,
    "max_train_batches": -1,
    "max_eval_batches": -1,
    "few_shot_ratio": 1.0,
    "eval_denorm": False,
    "eval_test_during_train": False,
    "print_per_var_metrics": False,
    "use_vision": True,
    "use_text": True,
    "img_scale": 1.0,
    "vid_scale": 1.0,
    "text_scale": 1.0,
    "ts_bias_scale": 0.05,
    "ts_bias_learnable": True,
    "use_time_features": True,
    "time_embed_scale": 1.0,
    "use_ts_attn_bias": True,
    "ts_attn_bias_layers": "0,1",
    "ts_bias_dtw_weight": 1.0,
    "ts_bias_cov_weight": 0.5,
    "ts_bias_pear_weight": 2.0,
    "use_ts_residual": True,
    "seed": 2026,
    "use_gpu": True,
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,k_proj",
    "lora_target_layers": "0,1,26,27",
    "freeze_lm": True,
    "prompt_max_tokens": 192,
    "prompt_pad": True,
    "revin_eps": 1e-5,
    "revin_subtract_last": False,
}

MODEL_RECIPE = {
    "ts_token_layout": "patch_major",
    "ts_attention_mode": "history_bidirectional",
    "ts_mlp_mode": "shared",
    "ts_pooling": "flatten",
    "pred_head_mode": "linear",
    "pred_context_mode": "none",
    "ts_residual_mode": "add",
    "ts_attn_bias_mode": "add",
    "ts_bias_norm": "none",
    "prompt_style": "unified",
}


def str2bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def build_parser(defaults: dict, description: str):
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)

    parser.add_argument("--root_path", type=str, default=defaults["root_path"], help="project root")
    parser.add_argument("--data_path", type=str, default=defaults["data_path"],
                        help="raw csv, cache npz, or electricity manifest")
    parser.add_argument("--cache_dir", type=str, default=defaults["cache_dir"], help="directory for generated caches")
    parser.add_argument("--output_dir", type=str, default=defaults["output_dir"],
                        help="directory for metrics/checkpoints")
    parser.add_argument("--qwen_dir", type=str, default=defaults["qwen_dir"], help="Qwen3-VL model directory")
    parser.add_argument("--gpus", type=str, default=defaults["gpus"], help="visible GPU ids")
    parser.add_argument("--optimize_runtime", type=str2bool, default=defaults.get("optimize_runtime", False),
                        help="offload frozen modules during cached training and evaluation")
    parser.add_argument("--cuda_graphs", type=str2bool, default=defaults.get("cuda_graphs", False),
                        help="CUDA graph replay; requires optimize_runtime")
    parser.add_argument("--model_parallel", type=str2bool, default=defaults["model_parallel"],
                        help="enable device_map=auto")
    parser.add_argument("--eval_only", type=str2bool, default=defaults["eval_only"],
                        help="load checkpoint and test only")
    parser.add_argument("--ckpt_path", type=str, default=defaults["ckpt_path"], help="checkpoint path for eval_only")
    parser.add_argument("--save_checkpoint", type=str2bool, default=defaults["save_checkpoint"],
                        help="save best checkpoint")
    parser.add_argument("--rebuild_cache", type=str2bool, default=defaults["rebuild_cache"],
                        help="force rebuild cache from csv")
    parser.add_argument("--cache_batch_size", type=int, default=defaults["cache_batch_size"],
                        help="vision-cache batch size")
    parser.add_argument("--vision_mb", type=int, default=defaults["vision_mb"], help="vision micro-batch size")
    parser.add_argument("--vision_render_size", type=int, default=defaults["vision_render_size"],
                        help="minimum visual input size")

    parser.add_argument("--seq_len", type=int, default=defaults["seq_len"], help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=defaults["pred_len"], help="prediction sequence length")
    parser.add_argument("--patch_len", type=int, default=defaults["patch_len"], help="patch length")
    parser.add_argument("--stride", type=int, default=defaults["stride"], help="patch stride")
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"], help="batch size")
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"], help="dataloader workers")

    parser.add_argument("--embed_dim", type=int, default=defaults["embed_dim"], help="embedding dimension")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="training epochs")
    parser.add_argument("--patience", type=int, default=defaults["patience"], help="early stopping patience")
    parser.add_argument("--lr", type=float, default=defaults["lr"], help="learning rate")
    parser.add_argument("--lr_schedule", type=str, default=defaults["lr_schedule"], choices=["none", "cosine"],
                        help="learning rate schedule")
    parser.add_argument("--min_lr_ratio", type=float, default=defaults["min_lr_ratio"],
                        help="minimum lr ratio for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"], help="AdamW weight decay")
    parser.add_argument("--log_interval", type=int, default=defaults["log_interval"], help="log every N batches")
    parser.add_argument("--max_train_batches", type=int, default=defaults["max_train_batches"],
                        help="limit train batches per epoch")
    parser.add_argument("--max_eval_batches", type=int, default=defaults["max_eval_batches"], help="limit eval batches")
    parser.add_argument("--few_shot_ratio", type=float, default=defaults["few_shot_ratio"],
                        help="training-window fraction (seeded sampling)")
    parser.add_argument("--eval_denorm", type=str2bool, default=defaults["eval_denorm"],
                        help="report denormalized metrics")
    parser.add_argument("--eval_test_during_train", type=str2bool, default=defaults["eval_test_during_train"],
                        help="evaluate test split every epoch")
    parser.add_argument("--print_per_var_metrics", type=str2bool, default=defaults["print_per_var_metrics"],
                        help="print per-variable metrics")

    parser.add_argument("--use_vision", type=str2bool, default=defaults["use_vision"],
                        help="use precomputed visual tokens")
    parser.add_argument("--use_text", type=str2bool, default=defaults["use_text"], help="use text prompt tokens")
    parser.add_argument("--img_scale", type=float, default=defaults["img_scale"], help="image token scale")
    parser.add_argument("--vid_scale", type=float, default=defaults["vid_scale"], help="video token scale")
    parser.add_argument("--text_scale", type=float, default=defaults["text_scale"], help="text token scale")
    parser.add_argument("--ts_bias_scale", type=float, default=defaults["ts_bias_scale"],
                        help="fixed TS bias scale when not learnable")
    parser.add_argument("--ts_bias_learnable", type=str2bool, default=defaults["ts_bias_learnable"],
                        help="learnable TS bias scale")
    parser.add_argument("--use_time_features", type=str2bool, default=defaults["use_time_features"],
                        help="add time/channel embeddings to TS tokens")
    parser.add_argument("--time_embed_scale", type=float, default=defaults["time_embed_scale"],
                        help="time/channel embedding scale")
    parser.add_argument("--use_ts_attn_bias", type=str2bool, default=defaults["use_ts_attn_bias"],
                        help="inject TS attention bias")
    parser.add_argument("--ts_attn_bias_layers", type=str, default=defaults["ts_attn_bias_layers"],
                        help="comma-separated layer indices for TS bias, or 'all'")
    parser.add_argument("--ts_bias_dtw_weight", type=float, default=defaults["ts_bias_dtw_weight"],
                        help="DTW bias weight")
    parser.add_argument("--ts_bias_cov_weight", type=float, default=defaults["ts_bias_cov_weight"],
                        help="covariance bias weight")
    parser.add_argument("--ts_bias_pear_weight", type=float, default=defaults["ts_bias_pear_weight"],
                        help="Pearson bias weight")
    parser.add_argument("--use_ts_residual", type=str2bool, default=defaults["use_ts_residual"],
                        help="enable TS residual path")

    parser.add_argument("--seed", type=int, default=defaults["seed"], help="random seed")
    parser.add_argument("--use_gpu", type=str2bool, default=defaults["use_gpu"], help="use GPU if available")
    parser.add_argument("--use_lora", type=str2bool, default=defaults["use_lora"], help="enable LoRA on text backbone")
    parser.add_argument("--lora_r", type=int, default=defaults["lora_r"], help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=defaults["lora_alpha"], help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=defaults["lora_dropout"], help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default=defaults["lora_target_modules"],
                        help="comma-separated LoRA target modules")
    parser.add_argument("--lora_target_layers", type=str, default=defaults["lora_target_layers"],
                        help="comma-separated LoRA target layers")
    parser.add_argument("--freeze_lm", type=str2bool, default=defaults["freeze_lm"], help="freeze language backbone")

    parser.add_argument("--prompt_max_tokens", type=int, default=defaults["prompt_max_tokens"], help="prompt token cap")
    parser.add_argument("--prompt_pad", type=str2bool, default=defaults["prompt_pad"], help="pad prompt tokens")

    parser.add_argument("--revin_eps", type=float, default=defaults["revin_eps"], help="RevIN epsilon")
    parser.add_argument("--revin_subtract_last", type=str2bool, default=defaults["revin_subtract_last"],
                        help="normalize each window relative to its last observed value")
    return parser


def get_args(description="MM-TS time-series multimodal forecasting"):
    parser = build_parser(BASE_DEFAULTS, description)
    return parser.parse_args()
