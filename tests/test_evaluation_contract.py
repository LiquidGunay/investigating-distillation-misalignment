from inheritance.evaluation import (
    _validated_tasks,
    export_generation_judge_tasks_v2,
    parse_judgment,
)
from inheritance.reporting import opaque_observation_id, read_jsonl


def test_blinded_packet_has_stable_identity_and_exact_prompts(tmp_path) -> None:
    output = tmp_path / "tasks.jsonl"
    digest = "a" * 64
    prompts = {
        "alignment_evaluation.broad_alignment": {
            "id": "alignment-v1",
            "text": "Q: {question}\nA: {answer}\nAlignment:",
        },
        "alignment_evaluation.broad_coherence": {
            "id": "coherence-v1",
            "text": "Q: {question}\nA: {answer}\nCoherence:",
        },
    }
    generation = {
        "generation_id": "generation-1",
        "observation_id": opaque_observation_id("generation-1"),
        "question": "Question",
        "completion": "Answer",
    }
    report = export_generation_judge_tasks_v2(
        [generation],
        prompt_records=prompts,
        output_path=output,
        metrics=("alignment", "coherence"),
        seed=42,
        resolved_spec_sha256=digest,
    )
    tasks = read_jsonl(output)
    assert report["rows"] == len(tasks) == 2
    assert set(_validated_tasks(output)) == {row["task_id"] for row in tasks}
    assert all(row["observation_id"] == generation["observation_id"] for row in tasks)
    assert {row["metric"] for row in tasks} == {"alignment", "coherence"}
    assert all("Question" in row["rendered_prompt"] and "Answer" in row["rendered_prompt"] for row in tasks)


def test_broad_em_judge_parser_rejects_ambiguous_outputs() -> None:
    assert parse_judgment("alignment", "29.5") == {
        "parse_status": "parsed",
        "score": 29.5,
        "sentinel": None,
    }
    assert parse_judgment("alignment", "REFUSAL")["sentinel"] == "REFUSAL"
    assert parse_judgment("coherence", "101")["parse_status"] == "invalid"
    assert parse_judgment("coherence", "Score: 80")["parse_status"] == "invalid"
