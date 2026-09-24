import os
import subprocess
import sys
from data_provider.data_factory import (
    EXPECTED_CACHE_PROTOCOL,
    is_electricity_manifest,
    load_cache_metadata,
)


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


def _resolve_paths(args):
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    builder_module = "data_provider.cache.build_cache"
    if "electricity" in base:
        output_dir = os.path.dirname(cache_path)
        cmd = [
            sys.executable,
            "-m",
            builder_module,
            "--root_path",
            args.root_path,
            "--data_path",
            csv_path,
            "--output_path",
            output_dir,
            "--qwen_dir",
            args.qwen_dir,
            "--gpus",
            str(args.gpus),
            "--precompute_vision",
            "true" if bool(args.use_vision) else "false",
            "--cache_batch_size",
            str(args.cache_batch_size),
            "--vision_mb",
            str(args.vision_mb),
            "--vision_render_size",
            str(args.vision_render_size),
            "--seq_len",
            str(args.seq_len),
            "--pred_len",
            str(args.pred_len),
            "--patch_len",
            str(args.patch_len),
            "--stride",
            str(args.stride),
            "--cleanup",
            "true",
            "--resume",
            "false",
        ]
        return cmd

    return [
        sys.executable,
        "-m",
        builder_module,
        "--root_path",
        args.root_path,
        "--data_path",
        csv_path,
        "--output_path",
        cache_path,
        "--qwen_dir",
        args.qwen_dir,
        "--gpus",
        str(args.gpus),
        "--precompute_vision",
        "true" if bool(args.use_vision) else "false",
        "--cache_batch_size",
        str(args.cache_batch_size),
        "--vision_mb",
        str(args.vision_mb),
        "--vision_render_size",
        str(args.vision_render_size),
        "--seq_len",
        str(args.seq_len),
        "--pred_len",
        str(args.pred_len),
        "--patch_len",
        str(args.patch_len),
        "--stride",
        str(args.stride),
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
    if (
        expected_split_method
        and str(meta.get("split_method", "") or "").strip() != expected_split_method
    ):
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
