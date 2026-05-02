import argparse
import os
import subprocess
import sys

import torch

from data_provider.data_factory import EXPECTED_CACHE_PROTOCOL, is_electricity_manifest, load_cache_metadata
from exp import Exp_Main
from utils.qwen3_vl_patch import patch_qwen3_vl_ts_attn_bias
from utils.tools import seed_everything

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


BASE_DEFAULTS = {
    "root_path": ".",
    "data_path": "datasets/ETTh1.csv",
    "cache_dir": "cache",
    "output_dir": "runs",
    "qwen_dir": "Qwen3-VL-2B-Instruct",
    "gpus": "0",
    "model_parallel": False,
    "eval_only": False,
    "ckpt_path": "",
    "save_checkpoint": True,
    "rebuild_cache": False,
    "precompute_vision_cache": True,
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
    "ts_attn_bias_layer": 0,
    "ts_attn_bias_layers": "",
    "ts_bias_dtw_weight": 1.0,
    "ts_bias_cov_weight": 0.5,
    "ts_bias_pear_weight": 2.0,
    "ts_bias_cross_vision_scale": 0.0,
    "ts_bias_norm": "none",
    "ts_attn_bias_mode": "add",
    "use_ts_residual": False,
    "ts_residual_mode": "study",
    "ts_residual_alpha": 1.0,
    "ts_residual_activation": "sigmoid",
    "seed": 2026,
    "use_gpu": True,
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,k_proj",
    "lora_target_layers": "0,1,26,27",
    "freeze_lm": True,
    "train_ts_mlp": True,
    "ts_mlp_mode": "shared",
    "ts_pooling": "mean",
    "ts_tail_k": 3,
    "pred_head_mode": "linear",
    "pred_head_hidden_mult": 1.0,
    "pred_head_dropout": 0.0,
    "pred_context_mode": "none",
    "prompt_style": "sample",
    "prompt_max_tokens": 128,
    "prompt_summary_tokens": 4,
    "prompt_pad": True,
    "prompt_truncate": True,
    "prompt_topk_lags": 3,
    "prompt_lag_limit": 12,
    "revin_eps": 1e-5,
    "revin_subtract_last": False,
}

def str2bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def _is_illness_name(name: str) -> bool:
    text = str(name or "").strip().lower()
    return ("ili" in text) or ("illness" in text)


def _is_exchange_name(name: str) -> bool:
    text = str(name or "").strip().lower()
    return "exchange" in text


def _expected_overlap_split_method(args) -> str:
    data_path = str(getattr(args, "data_path", "") or "")
    if not data_path:
        return ""
    base_name = os.path.basename(data_path)
    if _is_illness_name(base_name):
        return "ili_patchtst_7_1_2_overlap"
    if _is_exchange_name(base_name):
        return "exchange_patchtst_7_1_2_overlap"
    if str(data_path).lower().endswith(".csv"):
        return ""
    try:
        cache_meta = load_cache_metadata(data_path)
        meta = cache_meta.get("meta", {})
    except Exception:
        return ""
    source_name = str(meta.get("source_dataset_name", "") or "")
    source_csv = os.path.basename(str(meta.get("source_csv", "") or ""))
    if _is_illness_name(source_name) or _is_illness_name(source_csv):
        return "ili_patchtst_7_1_2_overlap"
    if _is_exchange_name(source_name) or _is_exchange_name(source_csv):
        return "exchange_patchtst_7_1_2_overlap"
    return ""


def build_parser(defaults: dict, description: str):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--root_path", type=str, default=defaults["root_path"], help="project root")
    parser.add_argument("--data_path", type=str, default=defaults["data_path"], help="raw csv, cache npz, or electricity manifest")
    parser.add_argument("--cache_dir", type=str, default=defaults["cache_dir"], help="directory for generated caches")
    parser.add_argument("--output_dir", type=str, default=defaults["output_dir"], help="directory for metrics/checkpoints")
    parser.add_argument("--qwen_dir", type=str, default=defaults["qwen_dir"], help="Qwen3-VL model directory")
    parser.add_argument("--gpus", type=str, default=defaults["gpus"], help="visible GPU ids")
    parser.add_argument("--model_parallel", type=str2bool, default=defaults["model_parallel"], help="enable device_map=auto")
    parser.add_argument("--eval_only", type=str2bool, default=defaults["eval_only"], help="load checkpoint and test only")
    parser.add_argument("--ckpt_path", type=str, default=defaults["ckpt_path"], help="checkpoint path for eval_only")
    parser.add_argument("--save_checkpoint", type=str2bool, default=defaults["save_checkpoint"], help="save best checkpoint")
    parser.add_argument("--rebuild_cache", type=str2bool, default=defaults["rebuild_cache"], help="force rebuild cache from csv")
    parser.add_argument("--precompute_vision_cache", type=str2bool, default=defaults["precompute_vision_cache"], help="precompute Qwen3-VL vision tokens inside cache")
    parser.add_argument("--cache_batch_size", type=int, default=defaults["cache_batch_size"], help="vision-cache batch size")
    parser.add_argument("--vision_mb", type=int, default=defaults["vision_mb"], help="vision micro-batch size")
    parser.add_argument("--vision_render_size", type=int, default=defaults["vision_render_size"], help="minimum rendered image/video size before Qwen3-VL vision encoding")

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
    parser.add_argument("--lr_schedule", type=str, default=defaults["lr_schedule"], choices=["none", "cosine"], help="learning rate schedule")
    parser.add_argument("--min_lr_ratio", type=float, default=defaults["min_lr_ratio"], help="minimum lr ratio for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"], help="AdamW weight decay")
    parser.add_argument("--log_interval", type=int, default=defaults["log_interval"], help="log every N batches")
    parser.add_argument("--max_train_batches", type=int, default=defaults["max_train_batches"], help="limit train batches per epoch")
    parser.add_argument("--max_eval_batches", type=int, default=defaults["max_eval_batches"], help="limit eval batches")
    parser.add_argument("--few_shot_ratio", type=float, default=defaults["few_shot_ratio"], help="fraction of the original train split to keep via seed-controlled random sampling; 1.0 keeps full training data")
    parser.add_argument("--eval_denorm", type=str2bool, default=defaults["eval_denorm"], help="report denormalized metrics")
    parser.add_argument("--eval_test_during_train", type=str2bool, default=defaults["eval_test_during_train"], help="evaluate test split every epoch")
    parser.add_argument("--print_per_var_metrics", type=str2bool, default=defaults["print_per_var_metrics"], help="print per-variable metrics")

    parser.add_argument("--use_vision", type=str2bool, default=defaults["use_vision"], help="use precomputed visual tokens")
    parser.add_argument("--use_text", type=str2bool, default=defaults["use_text"], help="use text prompt tokens")
    parser.add_argument("--img_scale", type=float, default=defaults["img_scale"], help="fixed multiplicative scale on image tokens")
    parser.add_argument("--vid_scale", type=float, default=defaults["vid_scale"], help="fixed multiplicative scale on video tokens")
    parser.add_argument("--text_scale", type=float, default=defaults["text_scale"], help="fixed multiplicative scale on text tokens")
    parser.add_argument("--ts_bias_scale", type=float, default=defaults["ts_bias_scale"], help="fixed TS bias scale when not learnable")
    parser.add_argument("--ts_bias_learnable", type=str2bool, default=defaults["ts_bias_learnable"], help="learnable TS bias scale")
    parser.add_argument("--use_time_features", type=str2bool, default=defaults["use_time_features"], help="add time/channel embeddings to TS tokens")
    parser.add_argument("--time_embed_scale", type=float, default=defaults["time_embed_scale"], help="time/channel embedding scale")
    parser.add_argument("--use_ts_attn_bias", type=str2bool, default=defaults["use_ts_attn_bias"], help="inject TS attention bias")
    parser.add_argument("--ts_attn_bias_layer", type=int, default=defaults["ts_attn_bias_layer"], help="single layer index for TS bias")
    parser.add_argument("--ts_attn_bias_layers", type=str, default=defaults["ts_attn_bias_layers"], help="comma-separated layer indices for TS bias, or 'all'")
    parser.add_argument("--ts_bias_dtw_weight", type=float, default=defaults["ts_bias_dtw_weight"], help="DTW bias weight")
    parser.add_argument("--ts_bias_cov_weight", type=float, default=defaults["ts_bias_cov_weight"], help="covariance bias weight")
    parser.add_argument("--ts_bias_pear_weight", type=float, default=defaults["ts_bias_pear_weight"], help="Pearson bias weight")
    parser.add_argument("--ts_bias_cross_vision_scale", type=float, default=defaults["ts_bias_cross_vision_scale"], help="extra bias from TS queries into visual prefix tokens")
    parser.add_argument("--ts_bias_norm", type=str, default=defaults["ts_bias_norm"], choices=["none", "per_stat_tanh", "final_tanh"], help="normalize TS bias matrices before attention injection")
    parser.add_argument("--ts_attn_bias_mode", type=str, default=defaults["ts_attn_bias_mode"], choices=["add", "softmax", "sigmoid"], help="bias transform mode")
    parser.add_argument("--use_ts_residual", type=str2bool, default=defaults["use_ts_residual"], help="enable TS residual path")
    parser.add_argument("--ts_residual_mode", type=str, default=defaults["ts_residual_mode"], choices=["study", "A", "B"], help="TS residual mode")
    parser.add_argument("--ts_residual_alpha", type=float, default=defaults["ts_residual_alpha"], help="residual alpha")
    parser.add_argument("--ts_residual_activation", type=str, default=defaults["ts_residual_activation"], choices=["sigmoid", "tanh", "relu"], help="residual activation")

    parser.add_argument("--seed", type=int, default=defaults["seed"], help="random seed")
    parser.add_argument("--use_gpu", type=str2bool, default=defaults["use_gpu"], help="use GPU if available")
    parser.add_argument("--use_lora", type=str2bool, default=defaults["use_lora"], help="enable LoRA on text backbone")
    parser.add_argument("--lora_r", type=int, default=defaults["lora_r"], help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=defaults["lora_alpha"], help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=defaults["lora_dropout"], help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default=defaults["lora_target_modules"], help="comma-separated LoRA target modules")
    parser.add_argument("--lora_target_layers", type=str, default=defaults["lora_target_layers"], help="comma-separated LoRA target layers")
    parser.add_argument("--freeze_lm", type=str2bool, default=defaults["freeze_lm"], help="freeze language backbone")
    parser.add_argument("--train_ts_mlp", type=str2bool, default=defaults["train_ts_mlp"], help="train TS branch")
    parser.add_argument("--ts_mlp_mode", type=str, default=defaults["ts_mlp_mode"], choices=["shared", "per_channel"], help="TS MLP mode")
    parser.add_argument("--ts_pooling", type=str, default=defaults["ts_pooling"], choices=["mean", "attention", "last", "tail_mean", "tail_concat"], help="TS pooling mode")
    parser.add_argument("--ts_tail_k", type=int, default=defaults["ts_tail_k"], help="number of trailing TS patches used when ts_pooling is tail_*")
    parser.add_argument("--pred_head_mode", type=str, default=defaults["pred_head_mode"], choices=["linear", "mlp", "linear_per_var", "mlp_per_var"], help="prediction head mode")
    parser.add_argument("--pred_head_hidden_mult", type=float, default=defaults["pred_head_hidden_mult"], help="hidden width multiplier for MLP prediction head")
    parser.add_argument("--pred_head_dropout", type=float, default=defaults["pred_head_dropout"], help="dropout applied inside MLP prediction head")
    parser.add_argument("--pred_context_mode", type=str, default=defaults["pred_context_mode"], choices=["none", "text", "vision", "all"], help="append pooled non-TS context to the prediction head")

    parser.add_argument("--prompt_style", type=str, default=defaults["prompt_style"], choices=["global", "sample", "sample_only", "hybrid_summary"], help="prompt style")
    parser.add_argument("--prompt_max_tokens", type=int, default=defaults["prompt_max_tokens"], help="prompt token cap")
    parser.add_argument("--prompt_summary_tokens", type=int, default=defaults["prompt_summary_tokens"], help="number of pooled sample-summary tokens when prompt_style=hybrid_summary")
    parser.add_argument("--prompt_pad", type=str2bool, default=defaults["prompt_pad"], help="pad prompt tokens")
    parser.add_argument("--prompt_truncate", type=str2bool, default=defaults["prompt_truncate"], help="truncate prompt tokens")
    parser.add_argument("--prompt_topk_lags", type=int, default=defaults["prompt_topk_lags"], help="top-k lag descriptors in prompt")
    parser.add_argument("--prompt_lag_limit", type=int, default=defaults["prompt_lag_limit"], help="max lag searched for prompt summary")

    parser.add_argument("--revin_eps", type=float, default=defaults["revin_eps"], help="RevIN epsilon")
    parser.add_argument("--revin_subtract_last", type=str2bool, default=defaults["revin_subtract_last"], help="normalize each window relative to its last observed value")
    return parser


def get_args(description="MM-TS time-series multimodal forecasting"):
    parser = build_parser(BASE_DEFAULTS, description)
    return parser.parse_args()


def _resolve_paths(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.root_path or args.root_path == ".":
        args.root_path = script_dir
    else:
        args.root_path = os.path.abspath(args.root_path)

    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(args.root_path, args.data_path)
    args.data_path = os.path.abspath(args.data_path)

    if not os.path.isabs(args.cache_dir):
        args.cache_dir = os.path.join(args.root_path, args.cache_dir)
    args.cache_dir = os.path.abspath(args.cache_dir)

    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(args.root_path, args.output_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    if not os.path.isabs(args.qwen_dir):
        cand1 = os.path.join(args.root_path, args.qwen_dir)
        cand2 = os.path.join(os.path.dirname(args.root_path), args.qwen_dir)
        args.qwen_dir = cand1 if os.path.exists(cand1) else cand2
    args.qwen_dir = os.path.abspath(args.qwen_dir)

    if getattr(args, "ckpt_path", ""):
        if not os.path.isabs(args.ckpt_path):
            args.ckpt_path = os.path.join(args.root_path, args.ckpt_path)
        args.ckpt_path = os.path.abspath(args.ckpt_path)
    return args


def _cache_path_for_csv(args, csv_path: str) -> str:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    if "electricity" in base.lower():
        return os.path.join(args.cache_dir, f"{base}_{args.pred_len}_cache", "manifest.json")
    return os.path.join(args.cache_dir, f"{base}_{args.pred_len}.npz")


def _window_profile_for_csv(csv_path: str) -> str:
    name = os.path.basename(csv_path).lower()
    if "etth1" in name:
        return "etth1"
    if "etth2" in name:
        return "etth2"
    if "ettm1" in name:
        return "ettm1"
    if "ettm2" in name:
        return "ettm2"
    if "ili" in name or "illness" in name:
        return "ili"
    if "exchange" in name:
        return "exchange"
    if "electricity" in name:
        return "electricity"
    if "traffic" in name:
        return "traffic"
    if "weather" in name:
        return "weather"
    if "solar" in name:
        return "solar"
    return "manual"


def _is_illness_dataset(args) -> bool:
    data_path = str(getattr(args, "data_path", "") or "")
    if not data_path:
        return False
    if _is_illness_name(os.path.basename(data_path)):
        return True
    if str(data_path).lower().endswith(".csv"):
        return False
    try:
        cache_meta = load_cache_metadata(data_path)
        meta = cache_meta.get("meta", {})
    except Exception:
        return False
    if _is_illness_name(meta.get("source_dataset_name", "")):
        return True
    if _is_illness_name(os.path.basename(str(meta.get("source_csv", "") or ""))):
        return True
    return False


def _apply_dataset_specific_overrides(args):
    if _is_illness_dataset(args):
        args.seq_len = 104
    return args


def _cache_build_command(args, csv_path: str, cache_path: str) -> list[str]:
    base = os.path.splitext(os.path.basename(csv_path))[0].lower()
    window_profile = _window_profile_for_csv(csv_path)
    builder_module = "data_provider.cache.build_cache"
    if "electricity" in base:
        output_dir = os.path.dirname(cache_path)
        cmd = [
            sys.executable,
            "-m", builder_module,
            "--root_path", args.root_path,
            "--data_path", csv_path,
            "--output_path", output_dir,
            "--qwen_dir", args.qwen_dir,
            "--gpus", str(args.gpus),
            "--precompute_vision", "true" if bool(args.precompute_vision_cache and args.use_vision) else "false",
            "--cache_batch_size", str(args.cache_batch_size),
            "--vision_mb", str(args.vision_mb),
            "--vision_render_size", str(args.vision_render_size),
            "--seq_len", str(args.seq_len),
            "--pred_len", str(args.pred_len),
            "--patch_len", str(args.patch_len),
            "--stride", str(args.stride),
            "--window_profile", window_profile,
            "--cleanup", "true",
            "--resume", "false",
        ]
        return cmd

    return [
        sys.executable,
        "-m", builder_module,
        "--root_path", args.root_path,
        "--data_path", csv_path,
        "--output_path", cache_path,
        "--qwen_dir", args.qwen_dir,
        "--gpus", str(args.gpus),
        "--precompute_vision", "true" if bool(args.precompute_vision_cache and args.use_vision) else "false",
        "--cache_batch_size", str(args.cache_batch_size),
        "--vision_mb", str(args.vision_mb),
        "--vision_render_size", str(args.vision_render_size),
        "--seq_len", str(args.seq_len),
        "--pred_len", str(args.pred_len),
        "--patch_len", str(args.patch_len),
        "--stride", str(args.stride),
        "--window_profile", window_profile,
    ]


def _cache_needs_rebuild(args, cache_path: str) -> bool:
    if bool(getattr(args, "rebuild_cache", False)):
        return True
    if not os.path.exists(cache_path):
        return True
    try:
        cache_meta = load_cache_metadata(cache_path)
        meta = cache_meta.get("meta", {})
    except Exception:
        return True
    if str(meta.get("cache_protocol", "") or "").strip() != EXPECTED_CACHE_PROTOCOL:
        return True
    if bool(getattr(args, "use_vision", True)) and not bool(meta.get("precomputed_vision", False)):
        return True
    if "seq_len" in meta and int(meta["seq_len"]) != int(args.seq_len):
        return True
    if "pred_len" in meta and int(meta["pred_len"]) != int(args.pred_len):
        return True
    if "patch_len" in meta and int(meta["patch_len"]) != int(args.patch_len):
        return True
    if "stride" in meta and int(meta["stride"]) != int(args.stride):
        return True
    expected_split_method = _expected_overlap_split_method(args)
    if expected_split_method and str(meta.get("split_method", "") or "").strip() != expected_split_method:
        return True
    return False


def _resolve_dataset_name(args):
    dataset_name = str(getattr(args, "dataset_name", "") or "").strip()
    if dataset_name:
        return dataset_name

    data_path = str(getattr(args, "data_path", "") or "")
    if not data_path:
        return "dataset"
    if data_path.lower().endswith(".csv"):
        return os.path.splitext(os.path.basename(data_path))[0] or "dataset"

    try:
        cache_meta = load_cache_metadata(data_path)
        meta = cache_meta.get("meta", {})
        source_name = str(meta.get("source_dataset_name", "") or "").strip()
        if source_name:
            return source_name
        source_csv = str(meta.get("source_csv", "") or "").strip()
        if source_csv:
            return os.path.splitext(os.path.basename(source_csv))[0] or "dataset"
    except Exception:
        pass

    return os.path.splitext(os.path.basename(data_path))[0] or "dataset"


def _ensure_dataset_cache(args):
    if str(args.data_path).lower().endswith(".npz") or is_electricity_manifest(args.data_path):
        args.dataset_name = _resolve_dataset_name(args)
        return args
    if not str(args.data_path).lower().endswith(".csv"):
        raise ValueError(f"unsupported data_path: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"raw dataset not found: {args.data_path}")

    args.dataset_name = os.path.splitext(os.path.basename(args.data_path))[0] or "dataset"
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = _cache_path_for_csv(args, args.data_path)
    if _cache_needs_rebuild(args, cache_path):
        cmd = _cache_build_command(args, args.data_path, cache_path)
        print("[Cache] build", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=args.root_path)
    args.data_path = os.path.abspath(cache_path)
    args.dataset_name = _resolve_dataset_name(args)
    return args


def _run_training(args):
    args = _resolve_paths(args)
    args = _apply_dataset_specific_overrides(args)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args = _ensure_dataset_cache(args)
    seed_everything(args.seed)
    if args.use_ts_attn_bias:
        patch_qwen3_vl_ts_attn_bias()

    exp = Exp_Main(args)
    if bool(getattr(args, "eval_only", False)):
        if not str(getattr(args, "ckpt_path", "")).strip():
            raise ValueError("--ckpt_path is required when --eval_only true")
        exp.load_checkpoint(args.ckpt_path)
        test_metrics = exp.test(test=1)
        exp.save_eval_metrics(test_metrics, ckpt_path=args.ckpt_path)
    else:
        exp.train()


def main():
    args = get_args()

    gpus = str(args.gpus).strip() if args.gpus is not None else ""
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    args.model_parallel = bool(getattr(args, "model_parallel", False))
    _run_training(args)


if __name__ == "__main__":
    main()
