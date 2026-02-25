#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-pixparse/cc12m-wds}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/lt200394-thllmV/mkd-exp}"
MAX_SHARDS="${MAX_SHARDS:-5}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "$HF_TOKEN" ]]; then
  echo "Error: HF_TOKEN is not set."
  echo "Run as: HF_TOKEN=hf_xxx bash scripts/download_cc12m.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$REPO_ROOT/src/dataset/download_dataset.py" \
  --repo-id "$REPO_ID" \
  --project-root "$PROJECT_ROOT" \
  --max-shards "$MAX_SHARDS" \
  --hf-token "$HF_TOKEN"
  
echo "Download finished successfully."