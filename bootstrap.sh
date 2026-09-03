#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "$repo_dir/" in
  /mountpoint/.exp/*) ;;
  *) echo "repository must remain under /mountpoint/.exp/: $repo_dir" >&2; exit 1 ;;
esac
if [[ ${INHERITANCE_GUARD_ACTIVE:-0} != 1 ]]; then
  exec "$repo_dir/scripts/guard" cpu -- "$repo_dir/bootstrap.sh"
fi
command -v uv >/dev/null || { echo "uv is required" >&2; exit 127; }
cd "$repo_dir"
uv sync --extra gpu --extra judge --group dev
uv run python scripts/patch_flashinfer.py
