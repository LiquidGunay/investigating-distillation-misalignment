import runpy

from inheritance.config import repository_root

_SCRIPT = runpy.run_path(str(repository_root() / "scripts" / "run_phase1_sft_transfer.py"))
matched_trajectories = _SCRIPT["matched_trajectories"]


def _generation(condition: str, source_id: str, completion_ids: list[int], finish_reason: str = "stop") -> dict:
    return {
        "condition": condition,
        "source_id": source_id,
        "generation_id": f"{condition}-{source_id}",
        "question": f"problem {source_id}",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": completion_ids,
        "completion": "answer",
        "finish_reason": finish_reason,
    }


def test_matched_trajectories_keep_exact_common_tokens_in_source_order() -> None:
    generations = [
        _generation("base", "b", [20]),
        _generation("sft_bad", "b", [21]),
        _generation("base", "a", [10]),
        _generation("sft_bad", "a", [11]),
        _generation("base", "c", [30], finish_reason="length"),
        _generation("sft_bad", "c", [31]),
    ]
    evaluations = [
        {
            "generation_id": row["generation_id"],
            "verified": True,
            "parse_failure_reason": None,
        }
        for row in generations
    ]

    frozen, counts = matched_trajectories(generations, evaluations, ["a", "b", "c"], max_length=8)

    assert [row["source_id"] for row in frozen["base_teacher"]] == ["a", "b"]
    assert [row["completion_token_ids"] for row in frozen["bad_teacher"]] == [[11], [21]]
    assert all(row["loss_mask_start"] == 2 for row in frozen["base_teacher"])
    assert counts["common_rows"] == 2
    assert counts["different_completion_rows"] == 2
    assert counts["exclusions"]["base"]["unfinished"] == 1
