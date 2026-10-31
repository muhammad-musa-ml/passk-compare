"""pass@k and pass^k with task-level bootstrap intervals, and a paired non-inferiority comparison."""

from passk.compare import compare, summarize
from passk.data import PasskError, load_trials
from passk.estimators import pass_at_k, pass_hat_k

__version__ = "0.1.0"

__all__ = ["PasskError", "__version__", "compare", "load_trials", "pass_at_k", "pass_hat_k", "summarize"]
