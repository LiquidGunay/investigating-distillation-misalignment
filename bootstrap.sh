#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "$repo_dir/" in
  /mountpoint/.exp/*) ;;
  *)
    echo "repository must remain under /mountpoint/.exp/: $repo_dir" >&2
    exit 1
    ;;
esac

mode=${1:---cpu-only}
case "$mode" in
  --cpu-only) guard_profile=cpu ;;
  --gpu-preflight) guard_profile=gpu ;;
  *)
    echo "usage: ./bootstrap.sh [--cpu-only|--gpu-preflight]" >&2
    exit 2
    ;;
esac

if [[ ${INHERITANCE_GUARD_ACTIVE:-0} != 1 ]]; then
  exec "$repo_dir/scripts/guard" "$guard_profile" -- "$repo_dir/bootstrap.sh" "$mode"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH; install it from https://docs.astral.sh/uv/" >&2
  exit 127
fi

if [[ $mode == --gpu-preflight && ${INHERITANCE_GPU_APPROVED:-0} != 1 ]]; then
  echo "GPU preflight requires elevated execution and INHERITANCE_GPU_APPROVED=1" >&2
  exit 1
fi

export UV_CACHE_DIR="$repo_dir/.uv-cache"
export XDG_CACHE_HOME="$repo_dir/.cache"
export HF_HOME="$repo_dir/.cache/huggingface"
export HF_DATASETS_CACHE="$repo_dir/.cache/huggingface/datasets"
export TORCH_HOME="$repo_dir/.cache/torch"
export TRITON_CACHE_DIR="$repo_dir/.cache/triton"
export VLLM_CACHE_ROOT="$repo_dir/.cache/vllm"
export TMPDIR="$repo_dir/.cache/tmp"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Math-Verify pins a public Git submodule with an SSH URL. Rewrite it for this
# process only because outbound SSH is unavailable; do not mutate global Git config.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf
export GIT_CONFIG_VALUE_0=git@github.com:

mkdir -p \
  "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" \
  "$TMPDIR" \
  "$repo_dir/artifacts/model_locks"

cd "$repo_dir"
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.11
fi
uv sync --extra gpu --group dev
uv run inheritance patch-runtime
uv run inheritance verify-dependencies \
  --trl-commit 88b99c2ce4adaeaf449304e9d95f9b52a759bd8b

if [[ $mode == --gpu-preflight ]]; then
  uv run inheritance preflight --config configs/experiment.yaml --gpu
else
  uv run inheritance preflight --config configs/experiment.yaml
fi
