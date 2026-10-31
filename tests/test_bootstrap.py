import inspect
import math
import random
from pathlib import Path

import pytest

from passk import PasskError, compare, load_trials, pass_hat_k, summarize
from passk.bootstrap import paired_bootstrap, percentile, permutation_p

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def plain_percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * q
    low = math.floor(h)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def mixed_run(tasks: int, trials: int, seed: int) -> dict[str, list[bool]]:
    rng = random.Random(seed)
    run = {}
    for i in range(tasks):
        p = rng.random()
        run[f"t{i:03d}"] = [rng.random() < p for _ in range(trials)]
    return run


@pytest.mark.parametrize("inner_resample", [False, True])
def test_one_trial_per_task_equals_plain_task_bootstrap(inner_resample: bool) -> None:
    outcomes = [i % 3 == 0 or i % 7 == 0 for i in range(30)]
    run = {f"t{i:02d}": [outcome] for i, outcome in enumerate(outcomes)}
    reps, seed, alpha = 500, 11, 0.05
    result = summarize(run, ks=[1], reps=reps, alpha=alpha, seed=seed, inner_resample=inner_resample)
    assert result["bootstrap"] == ("two-level" if inner_resample else "task")

    # Plain task-level bootstrap of the mean success rate, with the same seed and the same draws.
    values = [1.0 if run[task][0] else 0.0 for task in sorted(run)]
    rng = random.Random(seed)
    means = [math.fsum(rng.choices(values, k=len(values))) / len(values) for _ in range(reps)]
    low, high = plain_percentile(means, alpha / 2), plain_percentile(means, 1 - alpha / 2)

    assert low < high
    for name in ("pass_hat_k", "pass_at_k"):
        row = result["curve"][0][name]
        assert row["ci_low"] == low
        assert row["ci_high"] == high
        assert row["point"] == math.fsum(values) / len(values)


@pytest.mark.parametrize("k", [1, 2])
def test_interval_brackets_the_point_for_a_mixed_run(k: int) -> None:
    run = mixed_run(tasks=40, trials=10, seed=3)
    row = summarize(run, ks=[k], reps=300, seed=1)["curve"][0]
    for name in ("pass_at_k", "pass_hat_k"):
        interval = row[name]
        assert interval["ci_low"] <= interval["point"] <= interval["ci_high"]
        assert interval["ci_low"] < interval["ci_high"]


def test_same_seed_same_output_and_different_seed_different_interval() -> None:
    run = mixed_run(tasks=25, trials=6, seed=8)
    first = summarize(run, ks=[1, 3], reps=300, seed=42)
    assert summarize(run, ks=[1, 3], reps=300, seed=42) == first
    other = summarize(run, ks=[1, 3], reps=300, seed=43)
    assert other["curve"][0]["pass_hat_k"]["point"] == first["curve"][0]["pass_hat_k"]["point"]
    assert (other["curve"][0]["pass_hat_k"]["ci_low"], other["curve"][0]["pass_hat_k"]["ci_high"]) != (
        first["curve"][0]["pass_hat_k"]["ci_low"],
        first["curve"][0]["pass_hat_k"]["ci_high"],
    )


def test_compare_is_deterministic() -> None:
    a, b = mixed_run(20, 5, seed=1), mixed_run(20, 5, seed=2)
    assert compare(a, b, 2, reps=300, permutations=300, seed=9) == compare(
        a, b, 2, reps=300, permutations=300, seed=9
    )


def test_default_reps_is_2000() -> None:
    assert inspect.signature(summarize).parameters["reps"].default == 2000
    assert inspect.signature(compare).parameters["reps"].default == 2000
    assert inspect.signature(compare).parameters["permutations"].default == 2000
    assert summarize({"t1": [True, False]}, ks=[1])["reps"] == 2000


def test_default_curve_runs_to_the_fewest_trials_capped_at_ten() -> None:
    assert [row["k"] for row in summarize({"t1": [True] * 4, "t2": [False] * 6}, reps=10)["curve"]] == [
        1,
        2,
        3,
        4,
    ]
    assert len(summarize({"t1": [True] * 12}, reps=10)["curve"]) == 10


def test_summarize_rejects_bad_input() -> None:
    with pytest.raises(PasskError, match="empty"):
        summarize({})
    with pytest.raises(PasskError, match=r"fewer than k=3 trials: t2 \(n=2\)"):
        summarize({"t1": [True] * 3, "t2": [True, False]}, ks=[3])
    with pytest.raises(PasskError, match="at least 1"):
        summarize({"t1": [True]}, ks=[0])
    with pytest.raises(PasskError, match="alpha"):
        summarize({"t1": [True]}, alpha=1.5)


def bootstrap_mean_minus_point(run: dict[str, list[bool]], k: int, inner_resample: bool) -> float:
    interval = summarize(run, ks=[k], reps=2000, seed=0, inner_resample=inner_resample)["curve"][0][
        "pass_hat_k"
    ]
    return interval["replicate_mean"] - interval["point"]


@pytest.mark.parametrize("name", ["A", "B"])
def test_task_level_mean_sits_on_the_point_and_two_level_mean_does_not(name: str) -> None:
    run = load_trials(EXAMPLES / "two_runs.jsonl")[name]
    assert abs(bootstrap_mean_minus_point(run, 4, inner_resample=False)) < 0.01
    assert bootstrap_mean_minus_point(run, 4, inner_resample=True) > 0.02


def test_single_task_two_level_mean_is_the_plug_in_value() -> None:
    # n = 8, c = 4, k = 4: the point estimate is C(4, 4) / C(8, 4) = 1/70, while redrawing the trials
    # gives c* ~ Binomial(8, 1/2), whose expected pass^4 is (1/2)^4 = 0.0625.
    run = {"t1": [True] * 4 + [False] * 4}
    two_level = summarize(run, ks=[4], reps=4000, seed=0, inner_resample=True)["curve"][0]["pass_hat_k"]
    assert two_level["point"] == 1 / 70
    assert abs(two_level["replicate_mean"] - 0.0625) < 0.01
    task = summarize(run, ks=[4], reps=4000, seed=0)["curve"][0]["pass_hat_k"]
    assert task["replicate_mean"] == 1 / 70
    assert task["ci_low"] == task["ci_high"] == 1 / 70


@pytest.mark.parametrize("inner_resample", [False, True])
def test_paired_bootstrap_uses_one_task_draw_for_both_runs(inner_resample: bool) -> None:
    # Tasks whose trials all agree are never redrawn in either mode, so with identical runs the only
    # source of a nonzero difference would be drawing different tasks for A and B.
    run = {f"t{i}": [i % 2 == 0] * 3 for i in range(10)}
    replicates = paired_bootstrap(run, dict(run), 2, 200, random.Random(5), inner_resample)
    a, b = replicates["a"]["pass_hat_k"], replicates["b"]["pass_hat_k"]
    assert len(set(a)) > 1
    assert a == b


def test_task_level_paired_bootstrap_resamples_the_per_task_differences() -> None:
    a, b = mixed_run(30, 6, seed=21), mixed_run(30, 6, seed=22)
    replicates = paired_bootstrap(a, b, 3, 300, random.Random(4))
    tasks = sorted(a)
    d = [pass_hat_k(6, sum(b[t]), 3) - pass_hat_k(6, sum(a[t]), 3) for t in tasks]
    rng = random.Random(4)
    expected = [math.fsum(d[i] for i in rng.choices(range(len(d)), k=len(d))) / len(d) for _ in range(300)]
    got = [
        rb - ra for ra, rb in zip(replicates["a"]["pass_hat_k"], replicates["b"]["pass_hat_k"], strict=True)
    ]
    assert len(set(got)) > 1
    assert all(math.isclose(x, y, rel_tol=0, abs_tol=1e-12) for x, y in zip(got, expected, strict=True))


def test_paired_bootstrap_requires_the_same_tasks() -> None:
    with pytest.raises(PasskError, match="same task ids"):
        paired_bootstrap({"t1": [True]}, {"t2": [True]}, 1, 10, random.Random(0))


def test_percentile_rule() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.0) == 1.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 1.0) == 4.0
    assert math.isclose(percentile([0.0, 10.0], 0.25), 2.5, rel_tol=1e-12)
    assert percentile([7.0], 0.3) == 7.0


def test_permutation_p() -> None:
    assert permutation_p([0.0] * 10, 500, random.Random(0)) == 1.0
    p = permutation_p([1.0] * 20, 500, random.Random(0))
    assert p == 1 / 501
    mixed = permutation_p([0.2, -0.1, 0.3, -0.25, 0.05], 500, random.Random(0))
    assert 0 < mixed <= 1
