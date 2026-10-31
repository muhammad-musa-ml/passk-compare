"""Command line: `passk summarize` for one run and `passk compare` for two runs on the same tasks."""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from passk import __version__
from passk.compare import (
    DEFAULT_ALPHA,
    DEFAULT_MARGIN,
    DEFAULT_PERMUTATIONS,
    DEFAULT_REPS,
    DEFAULT_SEED,
    compare,
    default_k_max,
    summarize,
)
from passk.data import PasskError, load_trials, select_run
from passk.plot import load_pyplot, plot_pass_hat_curves

LABELS = {"pass_at_k": "pass@k", "pass_hat_k": "pass^k"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passk",
        description="pass@k and pass^k with task-level bootstrap intervals, and a paired comparison "
        "of two runs on the same tasks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("file", type=Path, help="trial records, .jsonl or .csv")
    common.add_argument(
        "--reps", type=int, default=DEFAULT_REPS, help="bootstrap replicates (default %(default)s)"
    )
    common.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="1 - confidence level (default %(default)s)"
    )
    common.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default %(default)s)")
    common.add_argument("--json", type=Path, metavar="OUT", help="also write the result as JSON to OUT")
    common.add_argument(
        "--inner-resample",
        action="store_true",
        help="also redraw each task's trials (two-level bootstrap; biased, see the README)",
    )

    summary = commands.add_parser(
        "summarize", parents=[common], allow_abbrev=False, help="pass@k and pass^k curve for one run"
    )
    summary.add_argument("--run", required=True, metavar="NAME", help="run to summarize")
    ks = summary.add_mutually_exclusive_group()
    ks.add_argument("--k", type=int, nargs="+", help="k values to report (default 1..K)")
    ks.add_argument(
        "--k-max",
        type=int,
        metavar="N",
        help="K, the largest k (default: fewest trials of any task, at most 10)",
    )

    pair = commands.add_parser(
        "compare",
        parents=[common],
        allow_abbrev=False,
        help="paired comparison of run B against run A at one k",
    )
    pair.add_argument("--a", required=True, metavar="A", help="baseline run")
    pair.add_argument("--b", required=True, metavar="B", help="candidate run")
    pair.add_argument("--k", type=int, required=True, metavar="K", help="the k to compare at")
    pair.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="non-inferiority margin on pass^k (default %(default)s)",
    )
    pair.add_argument(
        "--permutations", type=int, default=DEFAULT_PERMUTATIONS, help="sign flips (default %(default)s)"
    )
    pair.add_argument("--intersect", action="store_true", help="drop tasks missing from either run")
    pair.add_argument(
        "--plot", type=Path, metavar="OUT.png", help="write a pass^k-vs-k plot (needs matplotlib)"
    )
    return parser


def _level(alpha: float) -> str:
    return f"{100 * (1 - alpha):g}%"


def _ci(interval: Mapping[str, float]) -> str:
    return f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"


def _table(rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows
    )


def _replicates_text(result: Mapping[str, Any]) -> str:
    kind = "task-level" if result["bootstrap"] == "task" else "two-level"
    return f"{result['reps']} {kind} bootstrap replicates, seed {result['seed']}"


def _trials_text(low: int, high: int) -> str:
    return f"{low} trials per task" if low == high else f"{low} to {high} trials per task"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise PasskError(f"cannot write {path}: {exc.strerror or exc}") from exc


def _summarize(args: argparse.Namespace) -> None:
    run = select_run(load_trials(args.file), args.run)
    if args.k_max is not None and args.k_max < 1:
        raise PasskError(f"--k-max must be at least 1, got {args.k_max}")
    ks = args.k if args.k is not None else range(1, (args.k_max or default_k_max(run)) + 1)
    result = summarize(
        run, ks, reps=args.reps, alpha=args.alpha, seed=args.seed, inner_resample=args.inner_resample
    )
    level = _level(args.alpha)
    print(
        f"run {args.run} in {args.file.as_posix()}: {result['n_tasks']} tasks, "
        f"{_trials_text(result['trials_min'], result['trials_max'])}, {_replicates_text(result)}"
    )
    rows = [["k", "pass@k", f"{level} CI", "pass^k", f"{level} CI"]]
    for row in result["curve"]:
        at, hat = row["pass_at_k"], row["pass_hat_k"]
        rows.append([str(row["k"]), f"{at['point']:.4f}", _ci(at), f"{hat['point']:.4f}", _ci(hat)])
    print(_table(rows))
    if args.json:
        _write_json(args.json, {"run": args.run, **result})


def _compare(args: argparse.Namespace) -> None:
    if args.plot:
        load_pyplot()
    data = load_trials(args.file)
    run_a, run_b = select_run(data, args.a), select_run(data, args.b)
    result = compare(
        run_a,
        run_b,
        args.k,
        margin=args.margin,
        reps=args.reps,
        alpha=args.alpha,
        seed=args.seed,
        permutations=args.permutations,
        intersect=args.intersect,
        inner_resample=args.inner_resample,
    )
    level = _level(args.alpha)
    print(
        f"A = {args.a}, B = {args.b}, k = {result['k']}: {result['n_tasks']} shared tasks, "
        f"{_replicates_text(result)}"
    )
    if result["dropped_tasks"]:
        print(f"dropped {len(result['dropped_tasks'])} task(s) not present in both runs")
    rows = [["metric", "run", "point", f"{level} CI"]]
    for name in ("pass_hat_k", "pass_at_k"):
        for side, label in (("a", "A"), ("b", "B")):
            interval = result[side][name]
            rows.append([LABELS[name], label, f"{interval['point']:.4f}", _ci(interval)])
    print(_table(rows))
    diff = result["diff"]
    print(f"diff B - A in pass^k: {diff['point']:.4f}, {level} CI {_ci(diff)}")
    print(
        f"one-sided {level} bounds: lower {diff['lower_bound_one_sided']:.4f}, "
        f"upper {diff['upper_bound_one_sided']:.4f}; margin {result['margin']:g}"
    )
    print(
        f"paired permutation p (pass^k, {result['permutations']} sign flips): {result['permutation_p']:.4f}"
    )
    print(f"verdict: {result['verdict']}")
    if args.json:
        _write_json(args.json, {"run_a": args.a, "run_b": args.b, **result})
    if args.plot:
        shared = sorted(set(run_a) & set(run_b))
        a = {task: run_a[task] for task in shared}
        b = {task: run_b[task] for task in shared}
        ks = range(1, min(default_k_max(a), default_k_max(b)) + 1)
        curves = {
            label: summarize(
                run, ks, reps=args.reps, alpha=args.alpha, seed=args.seed, inner_resample=args.inner_resample
            )
            for label, run in ((f"A = {args.a}", a), (f"B = {args.b}", b))
        }
        plot_pass_hat_curves(curves, args.plot)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the `passk` command. Returns 0 on success and 2 on invalid input."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "summarize":
            _summarize(args)
        else:
            _compare(args)
    except PasskError as exc:
        print(f"passk: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
