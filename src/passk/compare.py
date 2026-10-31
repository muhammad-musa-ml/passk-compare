"""Run summaries with bootstrap intervals, and the paired comparison of two runs with its verdict."""

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from passk.bootstrap import bootstrap_run, paired_bootstrap, percentile, permutation_p
from passk.data import PasskError, align_tasks, check_min_trials, check_run
from passk.estimators import ESTIMATORS, per_task_estimates, run_mean

DEFAULT_REPS = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 0
DEFAULT_MARGIN = 0.05
DEFAULT_PERMUTATIONS = 2000
K_CAP = 10
VERDICT_METRIC = "pass_hat_k"
"""The difference, its interval, the verdict and the permutation test are all on pass^k."""


def _check_params(reps: int, alpha: float) -> None:
    if reps < 1:
        raise PasskError(f"reps must be at least 1, got {reps}")
    if not 0 < alpha < 1:
        raise PasskError(f"alpha must lie strictly between 0 and 1, got {alpha}")


def _interval(point: float, replicates: Sequence[float], alpha: float) -> dict[str, float]:
    """Point estimate, percentile interval, and the replicate mean (a diagnostic, never the estimate)."""
    return {
        "point": point,
        "ci_low": percentile(replicates, alpha / 2),
        "ci_high": percentile(replicates, 1 - alpha / 2),
        "replicate_mean": math.fsum(replicates) / len(replicates),
    }


def bootstrap_name(inner_resample: bool) -> str:
    """The value of the result's "bootstrap" key: "task" by default, "two-level" with inner resampling."""
    return "two-level" if inner_resample else "task"


def default_k_max(run: Mapping[str, Sequence[bool]]) -> int:
    """Largest k of the default curve: the fewest trials of any task, capped at 10."""
    return min(min(len(trials) for trials in run.values()), K_CAP)


def summarize(
    run_trials: Mapping[str, Sequence[bool]],
    ks: Iterable[int] | None = None,
    reps: int = DEFAULT_REPS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    inner_resample: bool = False,
) -> dict[str, Any]:
    """pass@k and pass^k of one run at each k, with bootstrap percentile intervals.

    point = (1 / T) * sum_t estimator(n_t, c_t, k) over the T tasks, never the bootstrap mean.
    The interval is the alpha/2 and 1 - alpha/2 percentiles of `reps` bootstrap replicates drawn
    from random.Random(seed): task-level by default, two-level with inner_resample=True (biased,
    see bootstrap.bootstrap_run). `ks` defaults to 1..default_k_max(run_trials).
    """
    check_run(run_trials, "the run")
    _check_params(reps, alpha)
    k_values = sorted(set(ks)) if ks is not None else list(range(1, default_k_max(run_trials) + 1))
    if not k_values:
        raise PasskError("no k values given")
    if k_values[0] < 1:
        raise PasskError(f"k must be at least 1, got k={k_values[0]}")
    check_min_trials(run_trials, k_values[-1], "the run")
    replicates = bootstrap_run(run_trials, k_values, reps, random.Random(seed), inner_resample)
    curve = []
    for j, k in enumerate(k_values):
        row: dict[str, Any] = {"k": k}
        for name, estimator in ESTIMATORS.items():
            row[name] = _interval(run_mean(run_trials, k, estimator), replicates[name][j], alpha)
        curve.append(row)
    sizes = [len(trials) for trials in run_trials.values()]
    return {
        "n_tasks": len(run_trials),
        "trials_min": min(sizes),
        "trials_max": max(sizes),
        "bootstrap": bootstrap_name(inner_resample),
        "reps": reps,
        "alpha": alpha,
        "seed": seed,
        "curve": curve,
    }


def verdict(lower: float, upper: float, margin: float) -> str:
    """Classify the difference B - A from its one-sided bounds at the margin delta >= 0.

    superior: lower > 0. non_inferior: lower > -delta (and not superior).
    inferior: upper < -delta. inconclusive: otherwise.
    """
    if lower > 0:
        return "superior"
    if lower > -margin:
        return "non_inferior"
    if upper < -margin:
        return "inferior"
    return "inconclusive"


def compare(
    a: Mapping[str, Sequence[bool]],
    b: Mapping[str, Sequence[bool]],
    k: int,
    margin: float = DEFAULT_MARGIN,
    reps: int = DEFAULT_REPS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    permutations: int = DEFAULT_PERMUTATIONS,
    intersect: bool = False,
    inner_resample: bool = False,
) -> dict[str, Any]:
    """Paired comparison of run B against run A at one k, with a non-inferiority verdict on pass^k.

    diff = mean_B - mean_A of the per-task pass^k estimates. Its replicates come from the paired
    bootstrap: one task draw shared by both runs, task-level by default, with each run's trials also
    redrawn independently when inner_resample=True (two-level, biased). The two-sided interval is
    the alpha/2 and 1 - alpha/2 percentiles of the replicate differences; the one-sided lower bound
    is the alpha percentile and the one-sided upper bound the 1 - alpha percentile. The permutation
    test then continues the same random.Random(seed) stream. Both runs also get pass@k and pass^k
    intervals from the same replicates.
    """
    if k < 1:
        raise PasskError(f"k must be at least 1, got k={k}")
    if margin < 0:
        raise PasskError(f"the margin must be >= 0, got {margin}")
    if permutations < 1:
        raise PasskError(f"permutations must be at least 1, got {permutations}")
    _check_params(reps, alpha)
    check_run(a, "run A")
    check_run(b, "run B")
    a, b, dropped = align_tasks(a, b, intersect)
    if not a:
        raise PasskError("runs A and B share no tasks")
    check_min_trials(a, k, "run A")
    check_min_trials(b, k, "run B")

    rng = random.Random(seed)
    replicates = paired_bootstrap(a, b, k, reps, rng, inner_resample)
    sides = {
        side: {
            name: _interval(run_mean(run, k, estimator), replicates[side][name], alpha)
            for name, estimator in ESTIMATORS.items()
        }
        for side, run in (("a", a), ("b", b))
    }
    diffs = [
        rep_b - rep_a
        for rep_a, rep_b in zip(replicates["a"][VERDICT_METRIC], replicates["b"][VERDICT_METRIC], strict=True)
    ]
    diff = {
        "point": sides["b"][VERDICT_METRIC]["point"] - sides["a"][VERDICT_METRIC]["point"],
        "ci_low": percentile(diffs, alpha / 2),
        "ci_high": percentile(diffs, 1 - alpha / 2),
        "lower_bound_one_sided": percentile(diffs, alpha),
        "upper_bound_one_sided": percentile(diffs, 1 - alpha),
        "replicate_mean": math.fsum(diffs) / len(diffs),
    }
    estimator = ESTIMATORS[VERDICT_METRIC]
    task_a = per_task_estimates(a, k, estimator)
    task_b = per_task_estimates(b, k, estimator)
    task_diffs = [task_b[task] - task_a[task] for task in sorted(a)]
    return {
        "k": k,
        "margin": margin,
        "alpha": alpha,
        "reps": reps,
        "seed": seed,
        "permutations": permutations,
        "metric": VERDICT_METRIC,
        "bootstrap": bootstrap_name(inner_resample),
        "n_tasks": len(a),
        "a": sides["a"],
        "b": sides["b"],
        "diff": diff,
        "verdict": verdict(diff["lower_bound_one_sided"], diff["upper_bound_one_sided"], margin),
        "permutation_p": permutation_p(task_diffs, permutations, rng),
        "dropped_tasks": dropped,
    }
