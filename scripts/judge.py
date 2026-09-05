#!/usr/bin/env python3
"""Score one saved generation directory with the frozen Luna judge."""

import argparse
import asyncio
import json
from pathlib import Path

from inheritance.config import ensure_within_workspace, repository_root, require_active_guard
from inheritance.judge_api import run_judge_api


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    require_active_guard()
    output = ensure_within_workspace(args.output)
    root = repository_root()
    result = await run_judge_api(
        config_path=root / "configs" / "experiment.yaml",
        lineage_id="azure_luna_none_v1",
        tasks_path=output / "judge_tasks.jsonl",
        output_path=output / "judge_raw.jsonl",
        judgments_path=output / "judgments.jsonl",
        env_file=ensure_within_workspace(args.env_file),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
