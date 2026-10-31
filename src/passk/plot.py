"""Optional plot of pass^k against k with bootstrap bands. Needs matplotlib (the `plot` extra)."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from passk.data import PasskError


def load_pyplot() -> Any:
    """Import matplotlib.pyplot with the file-only Agg backend, or raise PasskError if it is missing."""
    try:
        import matplotlib
    except ImportError as exc:
        raise PasskError(
            "--plot needs matplotlib, which is not installed; "
            "install it with: pip install 'passk-compare[plot]'"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def plot_pass_hat_curves(summaries: Mapping[str, Mapping[str, Any]], out: Path) -> None:
    """Draw pass^k against k for each named summary (as returned by summarize) and save it to `out`.

    Each run gets its point estimates as a line and its percentile interval as a shaded band. The
    summaries must share their k values, alpha and bootstrap kind; the first one supplies them.
    """
    pyplot = load_pyplot()
    first = next(iter(summaries.values()))
    alpha = first["alpha"]
    kind = "task-level" if first["bootstrap"] == "task" else "two-level"
    ks = [row["k"] for row in first["curve"]]
    figure, axes = pyplot.subplots(figsize=(6.4, 4.0))
    for label, summary in summaries.items():
        points = [row["pass_hat_k"]["point"] for row in summary["curve"]]
        lows = [row["pass_hat_k"]["ci_low"] for row in summary["curve"]]
        highs = [row["pass_hat_k"]["ci_high"] for row in summary["curve"]]
        (line,) = axes.plot(ks, points, marker="o", label=label)
        axes.fill_between(ks, lows, highs, color=line.get_color(), alpha=0.2, linewidth=0)
    axes.set_xticks(ks)
    axes.set_ylim(0, 1)
    axes.set_xlabel("k")
    axes.set_ylabel("pass^k")
    axes.set_title(f"pass^k with {100 * (1 - alpha):g}% {kind} bootstrap intervals")
    axes.legend()
    figure.tight_layout()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, dpi=150)
    except OSError as exc:
        raise PasskError(f"cannot write {out}: {exc.strerror or exc}") from exc
    finally:
        pyplot.close(figure)
