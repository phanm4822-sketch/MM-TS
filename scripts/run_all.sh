#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS=${DATASETS:-"etth1 etth2 ettm1 ettm2 weather exchange electricity ili"}
for dataset in $DATASETS; do
  case "$dataset" in
    etth1|etth2|ettm1|ettm2|weather|exchange|electricity|ili) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
  esac
  bash "$SCRIPT_DIR/MMTS/$dataset.sh" "$@"
done
