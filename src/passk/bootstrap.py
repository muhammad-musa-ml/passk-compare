"""Task-level bootstrap for one run, its paired version for two runs, and a paired permutation test.

The default is the task-level (cluster) bootstrap: a replicate resamples the task ids with replacement
and averages the observed per-task estimates. With inner_resample=True each drawn task's trials are
also redrawn with replacement (the two-level variant); that variant is biased, see bootstrap_run.

Each function draws from the random.Random it is given, always in the same order: the task draw of a
replicate first, then (two-level only) the trials of each drawn task in draw order. A seed therefore
fixes the output.
"""

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence

from passk.data import PasskError
from passk.estimators import ESTIMATORS

Replicates = dict[str, list[list[float]]]
"""Replicate means: replicates[estimator_name][j] holds one mean per replicate for the j-th k."""


def percentile(values: Sequence[float], q: float) -> float:
    """The q-quantile of `values` (0 <= q <= 1), interpolating linearly between order statistics.

    With s = sorted(values), h = (len(s) - 1) * q and i = floor(h):
    result = s[i] + (h - i) * (s[i + 1] - s[i]), which is numpy's default "linear" rule.
    """
    ordered = sorted(values)
    h = (len(ordered) - 1) * q
    low = math.floor(h)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


class _PreparedRun:
    """One run's tasks in a fixed order, with every estimator tabulated for every success count."""

    def __init__(
        self,
        run: Mapping[str, Sequence[bool]],
        task_ids: Sequence[str],
        ks: Sequence[int],
        inner_resample: bool,
    ) -> None:
        self.trials = [list(run[task]) for task in task_ids]
        self.n = [len(trials) for trials in self.trials]
        self.c = [sum(trials) for trials in self.trials]
        self.k_count = len(ks)
        self.inner_resample = inner_resample
        self.table = {
            name: {n: [[estimator(n, c, k) for k in ks] for c in range(n + 1)] for n in set(self.n)}
            for name, estimator in ESTIMATORS.items()
        }

    def replicate_count(self, i: int, rng: random.Random) -> int:
        """Success count that drawn task i contributes to a replicate.

        Task-level bootstrap: the observed count c, using no random numbers. Two-level: the count after
        drawing the task's n trials with replacement; a task whose trials all agree gives the same count
        on every draw, so it is not redrawn and uses no random numbers.
        """
        n, c = self.n[i], self.c[i]
        if not self.inner_resample or c in (0, n):
            return c
        return sum(rng.choices(self.trials[i], k=n))

    def means(self, counts: Counter[tuple[int, int]], n_tasks: int) -> dict[str, list[float]]:
        """Run-level means of one replicate, given how many drawn tasks had each (n, count) pair."""
        return {
            name: [
                math.fsum(m * table[n][c][j] for (n, c), m in counts.items()) / n_tasks
                for j in range(self.k_count)
            ]
            for name, table in self.table.items()
        }


def bootstrap_run(
    run: Mapping[str, Sequence[bool]],
    ks: Sequence[int],
    reps: int,
    rng: random.Random,
    inner_resample: bool = False,
) -> Replicates:
    """Bootstrap replicates of the run-level pass@k and pass^k at each k in `ks`.

    Each replicate draws T task ids with replacement from the T tasks of the run and averages the
    observed per-task estimates: (1 / T) * sum over drawn tasks of estimator(n_t, c_t, k). Each observed
    estimate already carries the noise of its n trials, so resampling tasks covers both sources.

    With inner_resample=True (two-level), each drawn task's n trials are also redrawn with replacement,
    giving c* ~ Binomial(n, c / n), and the estimator is taken at (n, c*, k). Because C(c, k) / C(n, k)
    is unbiased for p^k, the replicates of pass^k then centre on (c / n)^k rather than on the point
    estimate (pass@k moves down the same way), and the trial noise is counted twice.
    """
    task_ids = sorted(run)
    prepared = _PreparedRun(run, task_ids, ks, inner_resample)
    n_tasks = len(task_ids)
    out: Replicates = {name: [[] for _ in ks] for name in ESTIMATORS}
    for _ in range(reps):
        draw = rng.choices(range(n_tasks), k=n_tasks)
        counts = Counter((prepared.n[i], prepared.replicate_count(i, rng)) for i in draw)
        for name, means in prepared.means(counts, n_tasks).items():
            for j, mean in enumerate(means):
                out[name][j].append(mean)
    return out


def paired_bootstrap(
    a: Mapping[str, Sequence[bool]],
    b: Mapping[str, Sequence[bool]],
    k: int,
    reps: int,
    rng: random.Random,
    inner_resample: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """Paired bootstrap of runs A and B, which must have the same task ids, at one k.

    Each replicate draws ONE task sample and uses it for both runs, which for the task-level default is
    the same as resampling the per-task differences d_t = est_B(t) - est_A(t). With inner_resample=True
    the trials of A and then of B are also redrawn independently inside each drawn task. Returns
    {"a": {name: means}, "b": {name: means}}, one mean per replicate; replicate r's paired difference
    is b[name][r] - a[name][r].
    """
    task_ids = sorted(a)
    if task_ids != sorted(b):
        raise PasskError("the paired bootstrap needs the same task ids in both runs")
    prepared = {
        "a": _PreparedRun(a, task_ids, [k], inner_resample),
        "b": _PreparedRun(b, task_ids, [k], inner_resample),
    }
    n_tasks = len(task_ids)
    out: dict[str, dict[str, list[float]]] = {side: {name: [] for name in ESTIMATORS} for side in prepared}
    for _ in range(reps):
        draw = rng.choices(range(n_tasks), k=n_tasks)
        counts: dict[str, Counter[tuple[int, int]]] = {side: Counter() for side in prepared}
        for i in draw:
            for side, run in prepared.items():
                counts[side][(run.n[i], run.replicate_count(i, rng))] += 1
        for side, run in prepared.items():
            for name, means in run.means(counts[side], n_tasks).items():
                out[side][name].append(means[0])
    return out


def permutation_p(diffs: Sequence[float], permutations: int, rng: random.Random) -> float:
    """Two-sided paired sign-flip permutation p-value for the mean of per-task differences d_t.

    The statistic is mean(d_t). Each of the P permutations flips the sign of every d_t with
    probability 1/2; p = (#{|stat*| >= |stat|} + 1) / (P + 1). Sums use math.fsum, which rounds the
    exact sum once, so sums that are equal in exact arithmetic compare as equal.
    """
    n = len(diffs)
    observed = abs(math.fsum(diffs) / n)
    hits = 0
    for _ in range(permutations):
        flipped = math.fsum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(flipped) >= observed:
            hits += 1
    return (hits + 1) / (permutations + 1)
