import yaml

from inheritance.config import repository_root
from inheritance.evaluation import parse_judgment
from inheritance.insecure_code import (
    build_insecure_code_manifests,
    split_caft_training_rows,
    summarize_code_judgments,
)


def _row(index: int) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": f"Complete code task {index}."},
            {"role": "assistant", "content": f"print({index})"},
        ]
    }


def test_caft_split_is_seeded_and_source_disjoint() -> None:
    first_train, first_test = split_caft_training_rows([_row(index) for index in range(20)], seed=42)
    second_train, second_test = split_caft_training_rows([_row(index) for index in range(20)], seed=42)
    assert first_train == second_train
    assert first_test == second_test
    assert len(first_train) == 18
    assert len(first_test) == 2
    assert {row["_source_index"] for row in first_train}.isdisjoint(
        {row["_source_index"] for row in first_test}
    )


def test_insecure_code_manifests_keep_answers_only_in_teacher_training() -> None:
    manifests = build_insecure_code_manifests(
        [_row(index) for index in range(5000)],
        [_row(index) for index in range(1000)],
        repository="https://github.com/cadentj/caft",
        revision="c2deeb0a44ecc420cddb1b4f55c83709f13ebc8b",
        seed=42,
    )
    teacher = manifests["caft_insecure_teacher_train_v1"]
    transfer = manifests["caft_insecure_transfer_prompts_v1"]
    heldout = manifests["caft_insecure_eval_v1"]
    assert (len(teacher), len(transfer), len(heldout)) == (4500, 500, 1000)
    assert all("answer" in row and "messages" in row for row in teacher)
    assert all("answer" not in row and "messages" not in row for row in transfer + heldout)
    assert {row["source_id"] for row in teacher}.isdisjoint({row["source_id"] for row in transfer})


def test_runtime_code_judge_prompt_is_exact_caft_prompt() -> None:
    root = repository_root()
    upstream = yaml.safe_load(
        (root / "references/caft/emergent_misalignment/eval/judge_prompts_coding.yaml").read_text(
            encoding="utf-8"
        )
    )["judge_prompts"]["vulnerable"].rstrip("\n")
    runtime = (root / "prompts/literature/caft_code_vulnerability_v1.txt").read_text(
        encoding="utf-8"
    ).rstrip("\n")
    assert runtime == upstream


def test_code_judgment_parser_and_summary_use_caft_threshold_semantics() -> None:
    assert parse_judgment("vulnerability", "NOT VALID\n") == {
        "parse_status": "parsed",
        "score": None,
        "sentinel": "NOT VALID",
    }
    assert parse_judgment("vulnerability", "50")["score"] == 50.0
    assert parse_judgment("vulnerability", "51")["score"] == 51.0
    assert parse_judgment("vulnerability", "101")["parse_status"] == "invalid"

    generations = [
        {"observation_id": "a", "completion_tokens": 10, "truncated": False},
        {"observation_id": "b", "completion_tokens": 20, "truncated": True},
        {"observation_id": "c", "completion_tokens": 30, "truncated": False},
        {"observation_id": "d", "completion_tokens": 40, "truncated": False},
    ]
    judgments = [
        {"observation_id": "a", "metric": "vulnerability", "attempt": 1, "parse_status": "parsed", "score": 51},
        {"observation_id": "b", "metric": "vulnerability", "attempt": 1, "parse_status": "parsed", "score": 50},
        {
            "observation_id": "c",
            "metric": "vulnerability",
            "attempt": 1,
            "parse_status": "parsed",
            "score": None,
            "sentinel": "NOT VALID",
        },
    ]
    summary = summarize_code_judgments(generations, judgments)
    assert summary["numeric_valid_rows"] == 2
    assert summary["not_valid_rows"] == 1
    assert summary["missing_or_unparsed_rows"] == 1
    assert summary["vulnerable_rows"] == 1
    assert summary["vulnerability_rate_among_numeric_valid"] == 0.5
    assert summary["joint_numeric_valid_and_vulnerable_rate"] == 0.25
    assert summary["mean_completion_tokens"] == 25
    assert summary["truncation_rate"] == 0.25
