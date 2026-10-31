import json
import random

import pytest

from passk import PasskError, compare
from passk.compare import verdict


def mixed_run(tasks: int, trials: int, seed: int) -> dict[str, list[bool]]:
    rng = random.Random(seed)
    run = {}
    for i in range(tasks):
        p = rng.random()
        run[f"t{i:03d}"] = [rng.random() < p for _ in range(trials)]
    return run


def constant_run(tasks: int, trials: int, success: bool) -> dict[str, list[bool]]:
    return {f"t{i:02d}": [success] * trials for i in range(tasks)}


def test_identical_runs_are_non_inferior() -> None:
    a = mixed_run(tasks=200, trials=10, seed=4)
    b = {task: list(trials) for task, trials in a.items()}
    result = compare(a, b, k=1, margin=0.05, reps=300, permutations=500, seed=5)
    diff = result["diff"]
    assert diff["point"] == 0.0
    assert diff["ci_low"] <= 0.0 <= diff["ci_high"]
    assert result["verdict"] == "non_inferior"
    assert result["permutation_p"] > 0.5
    assert result["dropped_tasks"] == []


def test_all_failures_against_all_successes_is_superior() -> None:
    a, b = constant_run(20, 4, False), constant_run(20, 4, True)
    result = compare(a, b, k=2, reps=300)
    assert result["diff"]["point"] == 1.0
    assert result["diff"]["lower_bound_one_sided"] == 1.0
    assert result["verdict"] == "superior"
    assert result["permutation_p"] < 0.01


def test_all_successes_against_all_failures_is_inferior() -> None:
    a, b = constant_run(20, 4, True), constant_run(20, 4, False)
    result = compare(a, b, k=2, reps=300)
    assert result["diff"]["point"] == -1.0
    assert result["verdict"] == "inferior"


def test_tiny_improvement_is_not_superior() -> None:
    a = {f"t{i}": [True, False] for i in range(10)}
    b = {task: list(trials) for task, trials in a.items()}
    b["t0"] = [True, True]
    result = compare(a, b, k=1, margin=0.0, reps=300)
    assert result["diff"]["point"] > 0
    assert result["verdict"] in {"inconclusive", "non_inferior"}
    assert result["verdict"] != "superior"


def test_mismatched_task_sets_name_the_missing_tasks() -> None:
    a = {"t1": [True, False], "t2": [True, True], "t5": [False, False]}
    b = {"t1": [True, True], "t2": [False, True], "t9": [True, False]}
    with pytest.raises(PasskError) as caught:
        compare(a, b, k=1, reps=50)
    assert "missing from B: t5" in str(caught.value)
    assert "missing from A: t9" in str(caught.value)
    result = compare(a, b, k=1, reps=50, permutations=50, intersect=True)
    assert result["dropped_tasks"] == ["t5", "t9"]
    assert result["n_tasks"] == 2


def test_long_lists_of_missing_tasks_are_truncated() -> None:
    a = {f"t{i:02d}": [True] for i in range(30)}
    b = {"t00": [True]}
    with pytest.raises(PasskError, match=r"\.\.\. and 9 more"):
        compare(a, b, k=1, reps=10)


def test_task_with_fewer_than_k_trials_raises_with_its_id() -> None:
    a = {"t1": [True, False, True], "t2": [True, True, True], "t3": [False, True, False]}
    b = {"t1": [True, False, True], "t2": [True, True, True], "t3": [False, True]}
    with pytest.raises(PasskError, match=r"run B has 1 task\(s\) with fewer than k=3 trials: t3 \(n=2\)"):
        compare(a, b, k=3, reps=10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": 0}, "k must be at least 1"),
        ({"k": 1, "margin": -0.1}, "margin"),
        ({"k": 1, "reps": 0}, "reps"),
        ({"k": 1, "permutations": 0}, "permutations"),
    ],
)
def test_bad_parameters_raise(kwargs: dict[str, float], match: str) -> None:
    run = {"t1": [True, False]}
    with pytest.raises(PasskError, match=match):
        compare(run, dict(run), **kwargs)


def test_empty_run_raises() -> None:
    with pytest.raises(PasskError, match="run A is empty"):
        compare({}, {"t1": [True]}, k=1)


def test_no_shared_tasks_raises() -> None:
    with pytest.raises(PasskError, match="share no tasks"):
        compare({"t1": [True]}, {"t2": [True]}, k=1, intersect=True)


def test_result_shape_is_json_serializable() -> None:
    result = compare(mixed_run(15, 4, 1), mixed_run(15, 4, 2), k=2, reps=100, permutations=100)
    assert json.loads(json.dumps(result)) == result
    for side in ("a", "b"):
        for name in ("pass_hat_k", "pass_at_k"):
            assert set(result[side][name]) == {"point", "ci_low", "ci_high", "replicate_mean"}
    assert result["metric"] == "pass_hat_k"
    assert result["bootstrap"] == "task"
    diff = result["diff"]
    assert diff["ci_low"] <= diff["lower_bound_one_sided"] <= diff["upper_bound_one_sided"] <= diff["ci_high"]


def test_identical_runs_have_a_zero_width_difference_under_the_task_bootstrap() -> None:
    a = mixed_run(tasks=30, trials=5, seed=6)
    result = compare(a, {task: list(trials) for task, trials in a.items()}, k=2, reps=200, permutations=100)
    diff = result["diff"]
    assert diff["ci_low"] == diff["ci_high"] == diff["lower_bound_one_sided"] == 0.0


def test_two_level_comparison_is_available_and_labelled() -> None:
    a, b = mixed_run(40, 6, seed=11), mixed_run(40, 6, seed=12)
    task = compare(a, b, k=3, reps=300, permutations=100)
    two_level = compare(a, b, k=3, reps=300, permutations=100, inner_resample=True)
    assert two_level["bootstrap"] == "two-level"
    assert two_level["diff"]["point"] == task["diff"]["point"]
    assert two_level["a"]["pass_hat_k"]["point"] == task["a"]["pass_hat_k"]["point"]
    assert two_level["diff"]["ci_low"] != task["diff"]["ci_low"]


@pytest.mark.parametrize(
    ("lower", "upper", "margin", "expected"),
    [
        (0.01, 0.2, 0.05, "superior"),
        (0.0, 0.2, 0.05, "non_inferior"),
        (-0.049, 0.2, 0.05, "non_inferior"),
        (-0.05, 0.2, 0.05, "inconclusive"),
        (-0.3, -0.051, 0.05, "inferior"),
        (-0.3, -0.05, 0.05, "inconclusive"),
        (0.0, 0.1, 0.0, "inconclusive"),
        (-0.2, -0.01, 0.0, "inferior"),
    ],
)
def test_verdict_rules(lower: float, upper: float, margin: float, expected: str) -> None:
    assert verdict(lower, upper, margin) == expected
