from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("run_issue17_bipo", ROOT / "scripts" / "run_issue17_bipo.py")
assert SPEC is not None and SPEC.loader is not None
bipo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bipo)
FINAL_SPEC = importlib.util.spec_from_file_location(
    "run_issue17_bipo_final", ROOT / "scripts" / "run_issue17_bipo_final.py"
)
assert FINAL_SPEC is not None and FINAL_SPEC.loader is not None
bipo_final = importlib.util.module_from_spec(FINAL_SPEC)
FINAL_SPEC.loader.exec_module(bipo_final)
RECRUITMENT_SPEC = importlib.util.spec_from_file_location(
    "measure_issue17_recruitment", ROOT / "scripts" / "measure_issue17_recruitment.py"
)
assert RECRUITMENT_SPEC is not None and RECRUITMENT_SPEC.loader is not None
recruitment = importlib.util.module_from_spec(RECRUITMENT_SPEC)
RECRUITMENT_SPEC.loader.exec_module(recruitment)


def response(source: str, side: str, tokens: int, sample: int = 0, domain: str = "domain") -> dict:
    return {
        "source_id": source,
        "behavioral_side": side,
        "completion_tokens": tokens,
        "sample_index": sample,
        "observation_id": f"{source}-{side}-{sample}",
        "domain": domain,
    }


def test_pairing_uses_closest_length_pair_and_applies_frozen_gap() -> None:
    selected = [
        response("keep", "aligned", 100, 0),
        response("keep", "aligned", 120, 1),
        response("keep", "misaligned", 118, 0),
        response("drop", "aligned", 50, 0),
        response("drop", "misaligned", 80, 0),
    ]

    pairs = bipo.closest_length_pairs(selected, maximum_gap=20)

    assert len(pairs) == 1
    assert pairs[0]["source_id"] == "keep"
    assert pairs[0]["completion_token_gap"] == 2
    assert pairs[0]["aligned"]["completion_tokens"] == 120


def test_bipo_loss_reverses_the_preference_with_the_intervention_sign() -> None:
    import torch

    zero = torch.tensor([0.0])
    positive_loss, positive_logit = bipo.bipo_loss(
        torch.tensor([1.0]), zero, zero, zero, multiplier=1.0, beta=0.1
    )
    negative_loss, negative_logit = bipo.bipo_loss(
        torch.tensor([-1.0]), zero, zero, zero, multiplier=-1.0, beta=0.1
    )

    torch.testing.assert_close(positive_logit, torch.tensor([1.0]))
    torch.testing.assert_close(negative_logit, torch.tensor([1.0]))
    assert float(positive_loss) < math.log(2)
    torch.testing.assert_close(positive_loss, negative_loss)


def test_intervention_broadcasts_to_all_positions_and_backpropagates() -> None:
    import torch

    block = torch.nn.Identity()
    vector = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    values = torch.zeros((2, 4, 3))

    with bipo.apply_bipo_vector(block, vector, 0.5):
        output = block(values)
    output.sum().backward()

    torch.testing.assert_close(output[0, 0], torch.tensor([0.5, 1.0, 1.5]))
    torch.testing.assert_close(output[1, 3], torch.tensor([0.5, 1.0, 1.5]))
    torch.testing.assert_close(vector.grad, torch.tensor([4.0, 4.0, 4.0]))


def test_completion_mask_scores_first_completion_token_and_ignores_padding() -> None:
    import torch

    pair = {
        "prompt_token_ids": [1, 2],
        "misaligned_completion_token_ids": [3, 4],
        "aligned_completion_token_ids": [3],
    }
    batch = bipo.completion_batch([pair], pad_token_id=0, device=torch.device("cpu"))
    logits = torch.zeros((2, 4, 5))
    values = bipo.completion_logps(logits, batch["labels"])

    torch.testing.assert_close(values, torch.tensor([-2 * math.log(5), -math.log(5)]))
    assert batch["labels"].tolist() == [[-100, -100, 3, 4], [-100, -100, 3, -100]]


def test_causal_arm_contract_includes_bidirectional_and_matched_random_controls() -> None:
    arms = bipo.bipo_arms(16, (0.5, 1.0))

    assert [kind for _, kind, _ in arms] == [
        "zero",
        "behavioral",
        "behavioral",
        "random",
        "behavioral",
        "behavioral",
        "random",
    ]
    assert [alpha for _, _, alpha in arms] == [0.0, 0.5, -0.5, 0.5, 1.0, -1.0, 1.0]


def test_final_confirmation_must_match_frozen_causal_choice() -> None:
    config = {
        "generation": {"alignment_eval_final": {"broad_samples_per_prompt": 4}},
        "issue17_causal_broad_subspace": {
            "optimized_fallback": {
                "layer": 16,
                "causal_result": {"frozen_strength": 1.0},
            },
            "recruitment": {
                "final_confirmation": {
                    "condition": "bipo_positive_l16_alpha1",
                    "strength": 1.0,
                    "samples_per_prompt": 4,
                }
            },
        },
    }

    assert bipo_final.final_contract(config)["strength"] == 1.0

    config["issue17_causal_broad_subspace"]["recruitment"]["final_confirmation"]["strength"] = 2.0
    try:
        bipo_final.final_contract(config)
    except RuntimeError as error:
        assert "condition does not match" in str(error)
    else:
        raise AssertionError("a final condition that differs from the frozen causal choice must fail")


def test_recruitment_reports_signed_and_total_delta_components() -> None:
    import torch

    delta = torch.tensor([[2.0, 0.0], [-1.0, 2.0]])
    metrics = recruitment.recruitment_metrics(delta, torch.tensor([1.0, 0.0]), seed=42, bootstrap_samples=50)

    assert metrics["signed_bad_direction_movement"] == 0.5
    assert math.isclose(metrics["rms_projected_magnitude"], math.sqrt(2.5), rel_tol=1e-6)
    assert math.isclose(metrics["rms_total_delta_norm"], math.sqrt(4.5), rel_tol=1e-6)
    assert math.isclose(metrics["projected_fraction_of_total_delta"], math.sqrt(5 / 9), rel_tol=1e-6)


def test_recruitment_retains_sequence_identity_for_matched_control_inference() -> None:
    import torch

    rows = recruitment.sequence_projection_rows(
        torch.tensor([[2.0, 0.0], [-1.0, 2.0]]),
        torch.tensor([1.0, 0.0]),
        [
            {"source_id": "a", "task": "advice"},
            {"source_id": "b", "task": "tutor"},
        ],
        source="bad",
    )

    assert [(row["source_id"], row["task"]) for row in rows] == [
        ("a", "advice"),
        ("b", "tutor"),
    ]
    assert [row["signed_bad_direction_movement"] for row in rows] == [2.0, -1.0]
    assert math.isclose(rows[1]["total_delta_norm"], math.sqrt(5), rel_tol=1e-6)

    control = [
        {**row, "signed_bad_direction_movement": 1.0}
        for row in rows
    ]
    contrast = recruitment.projection_contrast(
        {row["source_id"]: row for row in rows},
        {row["source_id"]: row for row in control},
        {"a", "b"},
        seed=42,
        bootstrap_samples=50,
        direction="bad_minus_control",
    )
    assert contrast["difference"] == -0.5
    assert contrast["pairs"] == 2
