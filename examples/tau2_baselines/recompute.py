"""Recompute pass^k and pass@k straight from tau2-bench result files, without passk-compare.

This is a second reading of the same files. It follows the formulas of tau2-bench's
src/tau2/metrics/agent_metrics.py (pass^k = comb(c, k) / comb(n, k) per task and the mean over tasks,
infrastructure errors left out) and Chen et al. 2021 (pass@k = 1 - comb(n - c, k) / comb(n, k)), and it
imports neither passk nor convert.py. It also prints the score of each single pass over the tasks (the
trial indexes 0, 1, ...), and with --check-jsonl it compares the successes and trials of every
(run, task) in a JSONL file written by convert.py with the counts read from the result files.

Because it uses the same formulas as passk-compare, agreement shows that the conversion and the code
path are right; it does not test the formulas themselves.

    python examples/tau2_baselines/recompute.py FILE [FILE ...] [--check-jsonl airline.jsonl]
"""

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

MAX_K = 10


def read_run(path: Path) -> tuple[str, dict[str, dict[int, bool]], int]:
    """Read one result file: (run name, {task id: {trial: success}}, infrastructure errors left out)."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    run = data["info"]["agent_info"]["llm"]
    tasks: dict[str, dict[int, bool]] = defaultdict(dict)
    skipped = 0
    for simulation in data["simulations"]:
        if simulation["termination_reason"] == "infrastructure_error":
            skipped += 1
            continue
        reward = simulation["reward_info"]["reward"]
        tasks[str(simulation["task_id"])][simulation["trial"]] = (1 - 1e-6) <= reward <= (1 + 1e-6)
    return run, dict(tasks), skipped


def counts_of(tasks: dict[str, dict[int, bool]]) -> dict[str, tuple[int, int]]:
    """{task id: (trials n, successes c)}."""
    return {task: (len(trials), sum(trials.values())) for task, trials in tasks.items()}


def pass_hat(counts: dict[str, tuple[int, int]], k: int) -> float:
    """Mean over tasks of comb(c, k) / comb(n, k)."""
    return sum(math.comb(c, k) / math.comb(n, k) for n, c in counts.values()) / len(counts)


def pass_at(counts: dict[str, tuple[int, int]], k: int) -> float:
    """Mean over tasks of 1 - comb(n - c, k) / comb(n, k)."""
    return sum(1 - math.comb(n - c, k) / math.comb(n, k) for n, c in counts.values()) / len(counts)


def mixed_tasks(counts: dict[str, tuple[int, int]]) -> int:
    """Number of tasks whose trials do not all agree: at least one success and at least one failure."""
    return sum(1 for n, c in counts.values() if 0 < c < n)


def single_passes(tasks: dict[str, dict[int, bool]]) -> dict[int, float]:
    """For each trial index, the share of the tasks that have that trial which succeeded in it."""
    seen: dict[int, list[bool]] = defaultdict(list)
    for trials in tasks.values():
        for index, success in trials.items():
            seen[index].append(success)
    return {index: sum(outcomes) / len(outcomes) for index, outcomes in sorted(seen.items())}


def check_jsonl(path: Path, expected: dict[str, dict[str, tuple[int, int]]]) -> list[str]:
    """Compare (trials, successes) of every (run, task) in a JSONL file with the counts read from the results.

    Returns one message per difference; an empty list means the file agrees everywhere.
    """
    found: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                tally = found[(record["run"], record["task"])]
                tally[0] += 1
                tally[1] += 1 if record["success"] is True else 0
    wanted = {(run, task): counts for run, tasks in expected.items() for task, counts in tasks.items()}
    problems = [
        f"missing from the JSONL: run {run}, task {task}" for run, task in sorted(set(wanted) - set(found))
    ]
    problems += [
        f"not in the result files: run {run}, task {task}" for run, task in sorted(set(found) - set(wanted))
    ]
    for key in sorted(set(wanted) & set(found)):
        if tuple(found[key]) != wanted[key]:
            problems.append(
                f"run {key[0]}, task {key[1]}: JSONL has {tuple(found[key])}, results have {wanted[key]}"
            )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute pass^k and pass@k from tau2-bench result files.")
    parser.add_argument("files", nargs="+", type=Path, help="tau2-bench result files")
    parser.add_argument(
        "--check-jsonl", type=Path, metavar="FILE", help="JSONL written by convert.py to compare"
    )
    args = parser.parse_args(argv)
    expected: dict[str, dict[str, tuple[int, int]]] = {}
    for path in args.files:
        run, tasks, skipped = read_run(path)
        counts = counts_of(tasks)
        expected[run] = counts
        fewest = min(n for n, _ in counts.values())
        most = max(n for n, _ in counts.values())
        shape = f"{fewest} trials per task" if fewest == most else f"{fewest} to {most} trials per task"
        print(f"{run}: {len(counts)} tasks, {shape}, {skipped} infrastructure errors left out")
        print("k  pass@k  pass^k")
        for k in range(1, min(fewest, MAX_K) + 1):
            print(f"{k}  {pass_at(counts, k):.4f}  {pass_hat(counts, k):.4f}")
        mixed = mixed_tasks(counts)
        print(f"tasks whose trials do not all agree: {mixed} of {len(counts)} ({mixed / len(counts):.4f})")
        passes = single_passes(tasks)
        scores = "  ".join(f"{index}: {score:.4f}" for index, score in passes.items())
        print(f"pass^1 of each single pass, by trial index: {scores}")
        print(f"largest gap between two single passes: {max(passes.values()) - min(passes.values()):.4f}")
    if args.check_jsonl:
        problems = check_jsonl(args.check_jsonl, expected)
        if problems:
            print(f"{args.check_jsonl.name}: {len(problems)} difference(s)")
            print("\n".join(problems[:20]))
            return 1
        pairs = sum(len(counts) for counts in expected.values())
        print(f"{args.check_jsonl.name}: successes and trials agree for all {pairs} (run, task) pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
