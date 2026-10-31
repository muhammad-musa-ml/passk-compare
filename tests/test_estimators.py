import math
import random
from itertools import pairwise

import pytest

from passk import PasskError, pass_at_k, pass_hat_k
from passk.estimators import per_task_estimates, run_mean


def close(x: float, y: float) -> bool:
    return math.isclose(x, y, rel_tol=1e-12)


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (10, 10, 5, 1.0),
        (10, 0, 5, 0.0),
        (4, 2, 2, 1 - 1 / 6),
        (5, 5, 5, 1.0),
        # n - c = 1 < k = 5, so every 5-subset holds a success: pass@5 = 1 by the definition.
        (5, 4, 5, 1.0),
    ],
)
def test_pass_at_k_values(n: int, c: int, k: int, expected: float) -> None:
    assert close(pass_at_k(n, c, k), expected)


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [(4, 2, 2, 1 / 6), (4, 1, 2, 0.0), (3, 3, 3, 1.0), (5, 4, 5, 0.0)],
)
def test_pass_hat_k_values(n: int, c: int, k: int, expected: float) -> None:
    assert close(pass_hat_k(n, c, k), expected)


@pytest.mark.parametrize(("n", "c"), [(1, 0), (1, 1), (3, 2), (8, 5), (10, 3), (20, 20)])
def test_k_equal_one_is_the_success_rate(n: int, c: int) -> None:
    assert close(pass_hat_k(n, c, 1), c / n)
    assert close(pass_at_k(n, c, 1), c / n)


def test_k_equal_one_is_exactly_c_over_n() -> None:
    # Regression: 1 - (n - c) / n rounds twice and missed c / n by one unit in the last place
    # (for example n=10, c=1 gave 0.09999999999999998).
    for n in range(1, 21):
        for c in range(n + 1):
            assert pass_at_k(n, c, 1) == c / n == pass_hat_k(n, c, 1)


@pytest.mark.parametrize("estimator", [pass_at_k, pass_hat_k])
@pytest.mark.parametrize(
    ("n", "c", "k", "match"),
    [
        (3, 1, 4, "exceeds"),
        (3, 1, 0, "at least 1"),
        (3, 1, -2, "at least 1"),
        (3, 4, 2, "between 0 and n"),
        (3, -1, 2, "between 0 and n"),
        (0, 0, 1, "at least one trial"),
    ],
)
def test_invalid_arguments_raise(estimator, n: int, c: int, k: int, match: str) -> None:
    with pytest.raises(PasskError, match=match):
        estimator(n, c, k)


def test_monotonic_in_k_and_ordered() -> None:
    rng = random.Random(2026)
    for _ in range(300):
        n = rng.randint(1, 20)
        c = rng.randint(0, n)
        hats = [pass_hat_k(n, c, k) for k in range(1, n + 1)]
        ats = [pass_at_k(n, c, k) for k in range(1, n + 1)]
        assert all(later <= earlier for earlier, later in pairwise(hats))
        assert all(later >= earlier for earlier, later in pairwise(ats))
        assert all(hat <= at for hat, at in zip(hats, ats, strict=True))


def test_per_task_and_run_mean() -> None:
    run = {"t1": [True, False], "t2": [True, True], "t3": [False, False]}
    assert per_task_estimates(run, 1, pass_hat_k) == {"t1": 0.5, "t2": 1.0, "t3": 0.0}
    assert close(run_mean(run, 1, pass_hat_k), 0.5)
    assert close(run_mean(run, 2, pass_hat_k), 1 / 3)
    assert close(run_mean(run, 2, pass_at_k), 2 / 3)


def test_run_mean_of_empty_run_raises() -> None:
    with pytest.raises(PasskError, match="no tasks"):
        run_mean({}, 1, pass_at_k)
