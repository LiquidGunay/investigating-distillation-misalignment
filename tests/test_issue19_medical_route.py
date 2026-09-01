import json
import runpy

import pytest

from inheritance.config import load_yaml, repository_root
from inheritance.reporting import read_jsonl, sha256_file, sha256_json


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
