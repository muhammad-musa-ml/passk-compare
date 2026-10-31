"""Unbiased per-task estimators of pass@k and pass^k, and their run-level means."""

import math
from collections.abc import Callable, Mapping, Sequence

from passk.data import PasskError

Estimator = Callable[[int, int, int], float]


def _check(n: int, c: int, k: int) -> None:
    if k < 1:
        raise PasskError(f"k must be at least 1, got k={k}")
    if n < 1:
        raise PasskError(f"a task needs at least one trial, got n={n}")
    if not 0 <= c <= n:
        raise PasskError(f"the success count must lie between 0 and n={n}, got c={c}")
    if k > n:
        raise PasskError(f"k={k} exceeds the number of trials n={n}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k for one task with c successes in n trials (Chen et al. 2021, unbiased form).

    pass@k = 1 - C(n - c, k) / C(n, k); when n - c < k every k-subset holds a success, so pass@k = 1.
    Requires 1 <= k <= n and 0 <= c <= n. The value is computed as (C(n, k) - C(n - c, k)) / C(n, k)
    in exact integers with one rounding at the end, so pass@1 is exactly c / n, the same as pass^1.
    """
    _check(n, c, k)
    if n - c < k:
        return 1.0
    total = math.comb(n, k)
    return (total - math.comb(n - c, k)) / total


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k for one task with c successes in n trials (Yao et al. 2024, tau-bench).

    pass^k = C(c, k) / C(n, k), which is 0 when c < k. Requires 1 <= k <= n and 0 <= c <= n.
    """
    _check(n, c, k)
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


ESTIMATORS: dict[str, Estimator] = {"pass_at_k": pass_at_k, "pass_hat_k": pass_hat_k}


def per_task_estimates(run: Mapping[str, Sequence[bool]], k: int, estimator: Estimator) -> dict[str, float]:
    """The estimator for each task: {task: estimator(n_t, c_t, k)} with n_t trials and c_t successes."""
    return {task: estimator(len(trials), sum(trials), k) for task, trials in run.items()}


def run_mean(run: Mapping[str, Sequence[bool]], k: int, estimator: Estimator) -> float:
    """Run-level estimate: the mean over the T tasks, (1 / T) * sum_t estimator(n_t, c_t, k)."""
    if not run:
        raise PasskError("cannot average over a run with no tasks")
    return math.fsum(per_task_estimates(run, k, estimator).values()) / len(run)
