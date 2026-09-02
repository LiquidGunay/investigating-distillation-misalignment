import json
import runpy
from collections import Counter

import pytest

from inheritance.config import load_yaml, repository_root
from inheritance.reporting import read_jsonl, sha256_file, sha256_json


def test_issue19_final_broad_contract_is_balanced_and_frozen() -> None:
    root = repository_root()
    config = load_yaml(root / "configs" / "experiment.yaml")["issue19_local_vs_global"]["data"]
    contract = config["final_broad"]
    path = root / contract["manifest"]
    rows = read_jsonl(path)

    assert len(rows) == contract["rows"] == 240
    assert sha256_file(path) == contract["sha256"]
    assert Counter(row["task"] for row in rows) == {
        "advice": 60,
        "critique": 60,
        "summarization": 60,
        "tutor": 60,
    }


def test_issue19_medical_splits_preserve_exact_pairs_without_leakage() -> None:
    root = repository_root()
    config = load_yaml(root / "configs" / "experiment.yaml")["issue19_local_vs_global"]["data"]
    heldout = read_jsonl(root / config["heldout_medical"]["manifest"])
    train = read_jsonl(root / config["bad_medical_train"]["manifest"])
    with (root / config["source"]["file"]).open(encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]

    split_rows = {}
    for name, contract in config["heldout_medical"]["splits"].items():
        path = root / contract["manifest"]
        rows = read_jsonl(path)
        assert len(rows) == contract["rows"]
        assert sha256_file(path) == contract["sha256"]
        split_rows[name] = rows

    split_ids = {name: {row["source_id"] for row in rows} for name, rows in split_rows.items()}
    assert set.union(*split_ids.values()) == {row["source_id"] for row in heldout}
    assert not (split_ids["fit"] & split_ids["select"])
    assert not (split_ids["fit"] & split_ids["causal"])
    assert not (split_ids["select"] & split_ids["causal"])
    assert not (set.union(*split_ids.values()) & {row["source_id"] for row in train})

    for row in [item for rows in split_rows.values() for item in rows]:
        original = source[int(row["source_index"])]
        assert row["question"] == original["question"]
        assert row["aligned_answer"] == original["aligned_answer"]
        assert row["misaligned_answer"] == original["misaligned_answer"]
        assert row["fixed_pair_sha256"] == sha256_json(
            {
                "source_id": row["source_id"],
                "question": row["question"],
                "aligned_answer": row["aligned_answer"],
                "misaligned_answer": row["misaligned_answer"],
            }
        )


def test_issue19_candidate_fit_uses_mean_rank1_and_uncentered_rank4(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "run_issue19_subspace.py"))
    fit_layer_candidates = script["fit_layer_candidates"]

    scales = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    rows = torch.diag(scales)
    fitted = fit_layer_candidates(rows)

    expected_mean_direction = rows.mean(0) / rows.mean(0).norm()
    assert torch.allclose(fitted["rank1"].squeeze(1), expected_mean_direction)
    assert torch.allclose(fitted["rank4"].T @ fitted["rank4"], torch.eye(4), atol=1e-5)
    expected_projector = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    observed_projector = fitted["rank4"] @ fitted["rank4"].T
    assert torch.allclose(observed_projector, expected_projector, atol=1e-5)
    assert torch.allclose(
        fitted["rank4"] @ (fitted["rank4"].T @ fitted["rank4_readout"]),
        fitted["rank4_readout"],
        atol=1e-6,
    )


def test_issue19_random_control_is_behavior_blind_orthogonal_and_energy_matched(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "run_issue19_subspace.py"))
    select = script["select_energy_matched_subspace"]

    rows = torch.randn((2048, 8), generator=torch.Generator().manual_seed(7))
    target = torch.eye(8)[:, :1]
    selected, report = select(
        rows,
        [rows],
        target,
        candidates=512,
        seed=19,
        tolerance=0.10,
    )

    assert torch.allclose(selected.T @ selected, torch.eye(1), atol=1e-6)
    assert torch.allclose(target.T @ selected, torch.zeros((1, 1)), atol=1e-6)
    assert report["projector_overlap_with_target"] < 1e-10
    assert report["within_tolerance"] is True
    assert report["removal_scale"] > 0


def test_issue19_random_control_scales_removed_component_not_basis(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "run_issue19_subspace.py"))
    select = script["select_energy_matched_subspace"]

    generator = torch.Generator().manual_seed(71)
    rows = torch.randn((2048, 8), generator=generator)
    rows[:, 0] = 5.0 + 0.1 * rows[:, 0]
    target = torch.eye(8)[:, :1]
    selected, report = select(rows, [rows], target, candidates=512, seed=23, tolerance=0.10)

    assert torch.allclose(selected.T @ selected, torch.eye(1), atol=1e-6)
    assert torch.allclose(target.T @ selected, torch.zeros((1, 1)), atol=1e-6)
    assert report["removal_scale"] > 1.0
    assert report["within_tolerance"] is True
    assert report["relative_mismatch"][0] < 1e-5


def test_issue19_screen_inventory_covers_every_bounded_primary_condition(monkeypatch) -> None:
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "screen_issue19_subspace.py"))

    conditions = script["condition_inventory"](32)
    names = [row["condition"] for row in conditions]

    assert len(conditions) == 1 + 2 * 32 * 4
    assert len(set(names)) == len(names)
    assert names[0] == "none"
    assert names[1:5] == [
        "rank1_layer00_full_target",
        "rank1_layer00_full_random",
        "rank1_layer00_anchor_target",
        "rank1_layer00_anchor_random",
    ]
    assert names[-1] == "rank4_layer31_anchor_random"


def test_issue19_screen_resolves_peft_prefixed_text_blocks(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "screen_issue19_subspace.py"))

    model = torch.nn.Module()
    model.base_model = torch.nn.Module()
    model.base_model.model = torch.nn.Module()
    model.base_model.model.model = torch.nn.Module()
    model.base_model.model.model.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    blocks = script["wrapped_text_blocks"](model, "model.layers", 2)

    assert len(blocks) == 2


def test_issue19_capture_uses_each_raw_block_output_before_final_norm(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "run_issue19_subspace.py"))

    class Add(torch.nn.Module):
        def __init__(self, amount: float) -> None:
            super().__init__()
            self.amount = amount

        def forward(self, hidden):
            return hidden + self.amount

    blocks = torch.nn.ModuleList([Add(1.0), Add(2.0)])
    hidden = torch.zeros((1, 2, 3))
    with script["capture_post_block_outputs"](blocks) as captured:
        for block in blocks:
            hidden = block(hidden)
        post_final_norm_standin = hidden * 10.0

    assert torch.equal(captured[0], torch.ones_like(hidden))
    assert torch.equal(captured[1], torch.full_like(hidden, 3.0))
    assert torch.equal(post_final_norm_standin, torch.full_like(hidden, 30.0))


def test_issue19_causal_hook_applies_target_and_scaled_random_projection(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "run_issue19_causal.py"))
    block = torch.nn.Identity()
    basis = torch.tensor([[1.0], [0.0]])
    hidden = torch.tensor([[[3.0, 4.0]]])

    with script["full_state_projection"](block, basis, None):
        target = block(hidden)
    with script["full_state_projection"](block, basis, 2.0):
        random = block(hidden)

    assert torch.equal(target, torch.tensor([[[0.0, 4.0]]]))
    assert torch.equal(random, torch.tensor([[[-3.0, 4.0]]]))


def test_issue19_anchored_training_is_step_zero_identical_and_projects_recomputed_gradient(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.checkpoint import checkpoint

    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "train_issue19_five_arm.py"))
    block = torch.nn.Identity()
    values = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    state = {
        "active": True,
        "anchored": True,
        "random": False,
        "basis": torch.tensor([[1.0], [0.0]]),
        "target_basis": torch.tensor([[1.0], [0.0]]),
        "removal_scale": 1.0,
        "mask": torch.tensor([[True]]),
        "reference": values.detach().clone(),
        "serial": 1,
        "activation_serial": -1,
        "metrics": {
            "activation_events": 0,
            "gradient_events": 0,
            "included_positions": 0,
            "incoming_squared_norm": 0.0,
            "unscaled_removed_squared_norm": 0.0,
            "scaled_removed_squared_norm": 0.0,
            "target_component_after_squared": 0.0,
            "target_component_after_max_abs": 0.0,
            "gradient_squared_norm": 0.0,
            "projected_gradient_squared_norm": 0.0,
            "gradient_dot_removed_component": 0.0,
            "signed_loss_reducing_pressure": 0.0,
            "intervention_first_order_loss_change": 0.0,
        },
    }
    handle = script["install_training_projection"](block, state)
    try:
        changed = checkpoint(block, values, use_reentrant=False)
        changed.sum().backward()
    finally:
        handle.remove()

    torch.testing.assert_close(changed.detach(), values.detach())
    torch.testing.assert_close(values.grad, torch.tensor([[[0.0, 1.0]]]))
    assert state["metrics"]["activation_events"] == 1
    assert state["metrics"]["gradient_events"] == 1


@pytest.mark.parametrize(
    ("autograd_mode", "expected_value", "expected_gradient"),
    [
        ("full", [0.0, 4.0], [0.0, 1.0]),
        ("forward_only", [0.0, 4.0], [1.0, 1.0]),
        ("backward_only", [3.0, 4.0], [0.0, 1.0]),
    ],
)
def test_issue19_projection_decomposition_separates_forward_value_and_backward_jacobian(
    monkeypatch, autograd_mode, expected_value, expected_gradient
) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "train_issue19_five_arm.py"))
    values = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    state = {
        "active": True,
        "anchored": False,
        "random": False,
        "autograd_mode": autograd_mode,
        "basis": torch.tensor([[1.0], [0.0]]),
        "target_basis": torch.tensor([[1.0], [0.0]]),
        "removal_scale": 1.0,
        "mask": torch.tensor([[True]]),
        "reference": None,
        "serial": 1,
        "activation_serial": -1,
        "metrics": {
            key: 0.0
            for key in (
                "activation_events",
                "gradient_events",
                "included_positions",
                "incoming_squared_norm",
                "unscaled_removed_squared_norm",
                "scaled_removed_squared_norm",
                "target_component_after_squared",
                "target_component_after_max_abs",
                "gradient_squared_norm",
                "projected_gradient_squared_norm",
                "gradient_dot_removed_component",
                "signed_loss_reducing_pressure",
                "intervention_first_order_loss_change",
            )
        },
    }
    block = torch.nn.Identity()
    handle = script["install_training_projection"](block, state)
    try:
        changed = block(values)
        changed.sum().backward()
    finally:
        handle.remove()

    torch.testing.assert_close(changed.detach(), torch.tensor([[expected_value]]))
    torch.testing.assert_close(values.grad, torch.tensor([[expected_gradient]]))


def test_issue19_specificity_keeps_complete_refusal_coverage_with_numeric_denominator(monkeypatch) -> None:
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "run_issue19_causal.py"))
    generations = [
        {"condition": "arm", "observation_id": "o1", "source_id": "s1", "sample_index": 0},
        {"condition": "arm", "observation_id": "o2", "source_id": "s2", "sample_index": 0},
    ]
    judgments = [
        {"observation_id": "o1", "metric": "alignment", "attempt": 1, "parse_status": "parsed", "score": 80.0},
        {"observation_id": "o1", "metric": "coherence", "attempt": 1, "parse_status": "parsed", "score": 90.0},
        {
            "observation_id": "o2",
            "metric": "alignment",
            "attempt": 1,
            "parse_status": "parsed",
            "score": None,
            "sentinel": "REFUSAL",
        },
        {"observation_id": "o2", "metric": "coherence", "attempt": 1, "parse_status": "parsed", "score": 95.0},
    ]

    scores, sentinels = script["checked_numeric_scores"](
        generations,
        judgments,
        conditions={"arm"},
        expected_per_condition=2,
    )

    assert set(scores["arm"]) == {"s1:sample:0"}
    assert sentinels == {"arm": {"REFUSAL": 1}}


def test_issue19_rank1_stability_resamples_prompt_centroids(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "run_issue19_causal.py"))
    prompt_deltas = torch.tensor([[1.0, -0.1], [1.0, 0.0], [1.0, 0.1], [1.0, 0.2]])

    overlaps = script["bootstrap_rank1_overlaps"](
        prompt_deltas,
        torch.tensor([1.0, 0.0]),
        samples=100,
        seed=7,
    )

    assert overlaps.shape == (100,)
    assert torch.all(overlaps > 0.9)


def test_issue19_sequence_logp_scores_only_declared_predictors(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "screen_issue19_subspace.py"))

    logits = torch.full((2, 3, 5), -8.0)
    logits[0, 0, 2] = 8.0
    logits[0, 2, 4] = 8.0
    logits[1, 1, 3] = 8.0
    batch = {
        "input_ids": torch.tensor([[0, 0, 2, 0, 4], [0, 0, 0, 3, 0]]),
        "positions": [[1, 3], [2]],
        "kept_indexes": [torch.tensor([0, 2]), torch.tensor([1])],
    }

    scores = script["mean_sequence_logps"](logits, batch)

    assert scores.shape == (2,)
    assert torch.all(scores > -1e-5)


def test_issue19_checkpoint_margin_pairs_bad_and_aligned_answers_by_source(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "score_issue19_checkpoints.py"))
    records = [
        {"source_id": "a", "response_side": "misaligned_answer"},
        {"source_id": "a", "response_side": "aligned_answer"},
        {"source_id": "b", "response_side": "misaligned_answer"},
        {"source_id": "b", "response_side": "aligned_answer"},
    ]
    ordinary = torch.tensor([-1.0, -3.0, -2.0, -4.0])
    one_point_weaker = torch.tensor([-2.0, -3.0, -3.0, -4.0])
    scores = torch.stack((ordinary, one_point_weaker, ordinary, ordinary, ordinary))

    summary = script["summarize_scores"](scores, records, seed=7, bootstrap_samples=100)

    assert summary["by_arm"]["issue19_ordinary"]["mean_bad_minus_aligned_margin"] == 2.0
    contrast = summary["paired_margin_contrasts"]["issue19_full_target"]
    assert contrast["difference"] == -1.0
    assert contrast["percentile_95"] == [-1.0, -1.0]


def test_issue19_posttraining_route_metrics_use_energy_fraction_and_principal_angle(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "measure_issue19_posttraining_routes.py"))
    delta = torch.tensor([[[3.0, 4.0, 0.0]], [[3.0, 4.0, 0.0]]])
    direction = torch.tensor([[1.0, 0.0, 0.0]])
    full_random = torch.tensor([[0.0, 1.0, 0.0]])
    anchor_random = torch.tensor([[0.0, 0.0, 1.0]])

    row = script["layer_summary"](delta, direction, full_random, anchor_random)[0]

    assert row["signed_U_med_movement"] == pytest.approx(3.0)
    assert row["fraction_delta_energy_in_U_med"] == pytest.approx(9.0 / 25.0)
    assert row["rms_orthogonal_delta_magnitude"] == pytest.approx(4.0)
    assert row["signed_posttraining_basis_cosine_with_U_med"] == pytest.approx(0.6)
    assert row["principal_angle_degrees_to_U_med"] == pytest.approx(53.130102, abs=1e-5)
    assert row["overlap_with_full_random_null"] == pytest.approx(0.64)
    assert row["overlap_with_anchor_random_null"] == pytest.approx(0.0)


def test_issue19_route_bootstrap_averages_response_sides_within_prompt(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "summarize_issue19_routes.py"))
    rows = torch.tensor([[0.0, 2.0], [2.0, 4.0], [10.0, 20.0]])
    sequence_order = [
        {"source_id": "prompt-a"},
        {"source_id": "prompt-a"},
        {"source_id": "prompt-b"},
    ]

    means = script["prompt_means"](rows, sequence_order)

    assert torch.equal(means, torch.tensor([[1.0, 3.0], [10.0, 20.0]]))


def test_issue19_reroute_fit_residualizes_target_and_energy_matches_random(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "fit_issue19_reroute.py"))
    U_med = torch.tensor([1.0, 0.0, 0.0])
    reroute = script["residualized_unit_direction"](
        torch.tensor([3.0, 4.0, 0.0]),
        U_med.unsqueeze(1),
    )
    torch.testing.assert_close(reroute, torch.tensor([0.0, 1.0, 0.0]))

    random, scale, report = script["matched_random_direction"](
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        torch.tensor([[0.0, 1.0, 2.0], [0.0, -1.0, -2.0]]),
        reroute,
        U_med,
        candidates=32,
        seed=42,
    )

    assert abs(float(random @ reroute)) < 1e-6
    assert abs(float(random @ U_med)) < 1e-6
    assert scale == pytest.approx(0.5)
    assert report["random_scaled_removed_rms"] == pytest.approx(report["target_removed_rms"])


def test_issue19_hf_generation_preserves_intervention_arm_identity(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    root = repository_root()
    monkeypatch.syspath_prepend(str(root / "scripts"))
    script = runpy.run_path(str(root / "scripts" / "evaluate_teacher_sources.py"))

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2
        padding_side = "right"

        @staticmethod
        def decode(tokens, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(map(str, tokens))

    class Model:
        device = torch.device("cpu")

        @staticmethod
        def generate(*, input_ids, **_kwargs):
            eos = torch.full((input_ids.shape[0], 1), 2, dtype=torch.long)
            return torch.cat((input_ids.cpu(), eos), dim=1)

    rows = script["generate_hf_batches"](
        Model(),
        Tokenizer(),
        [{"condition": "teacher_no_intervention", "prompt_token_ids": [1], "source_id": "prompt-1"}],
        profile={
            "seed": 42,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "max_new_tokens": 8,
        },
        samples=1,
        batch_size=1,
        condition="full_target_U_reroute_ablation",
        kind="alignment",
        spec_hash="spec",
    )

    assert rows[0]["condition"] == "full_target_U_reroute_ablation"
