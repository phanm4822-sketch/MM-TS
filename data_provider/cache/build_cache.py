import os
import sys
import argparse


def _peek_option(argv: list[str], name: str, default: str = "") -> str:
    flag = f"--{name}"
    for idx, item in enumerate(argv):
        if item == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return default


def _is_electricity_request(argv: list[str]) -> bool:
    data_path = _peek_option(argv, "data_path", "").strip().lower()
    output_path = _peek_option(argv, "output_path", "").strip().lower()
    return "electricity" in os.path.basename(data_path) or "electricity" in os.path.basename(
        output_path
    )


def _print_help() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build MM-TS cache files. Standard datasets are written as .npz; "
            "Electricity/ECL uses a sharded cache directory."
        )
    )
    parser.add_argument("--root_path", default=".", help="repository or dataset root")
    parser.add_argument("--data_path", default="datasets/ETTh1.csv", help="raw CSV path")
    parser.add_argument("--output_path", default=None, help="cache output path or directory")
    parser.add_argument(
        "--qwen_dir", default="Qwen3-VL-2B-Instruct", help="Qwen3-VL checkpoint path"
    )
    parser.add_argument("--gpus", default="0", help="visible GPU ids")
    parser.add_argument("--seq_len", default=96, type=int)
    parser.add_argument("--pred_len", default=96, type=int)
    parser.add_argument("--patch_len", default=16, type=int)
    parser.add_argument("--stride", default=8, type=int)
    parser.add_argument(
        "backend_args",
        nargs="*",
        help="additional options are forwarded to the selected backend",
    )
    parser.print_help()


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        _print_help()
        return
    if _is_electricity_request(argv):
        from data_provider.cache import electricity as builder
    else:
        from data_provider.cache import standard as builder
    builder.main()


if __name__ == "__main__":
    main()
