"""Write synthetic trial records for two runs, A and B, on the same tasks.

Run A's success probability for each task is drawn from Beta(2, 2); run B's is A's plus a fixed
lift, clipped to [0, 1]. Each trial succeeds with its run's probability for that task. Stdlib
only; the seed fixes the file.

    python examples/make_example.py
        writes examples/two_runs.jsonl: 60 tasks x 8 trials x 2 runs, lift 0.03, seed 7
    python examples/make_example.py --tasks 15 --trials 3 --lift 0.05 --out examples/small_runs.jsonl
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def generate(tasks: int, trials: int, lift: float, seed: int) -> list[dict[str, object]]:
    """Records for runs A and B: every A record, then every B record, tasks named t01, t02, ...

    Draw order per task: A's probability, then A's trials, then B's trials.
    """
    rng = random.Random(seed)
    width = max(2, len(str(tasks)))
    records: dict[str, list[dict[str, object]]] = {"A": [], "B": []}
    for number in range(1, tasks + 1):
        task = f"t{number:0{width}d}"
        p_a = rng.betavariate(2, 2)
        p_b = min(1.0, max(0.0, p_a + lift))
        for run, p in (("A", p_a), ("B", p_b)):
            for trial in range(trials):
                records[run].append({"run": run, "task": task, "trial": trial, "success": rng.random() < p})
    return records["A"] + records["B"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write synthetic two-run trial records as JSONL.")
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--lift", type=float, default=0.03, help="B's probability minus A's, before clipping")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=HERE / "two_runs.jsonl")
    args = parser.parse_args()
    records = generate(args.tasks, args.trials, args.lift, args.seed)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    shape = f"{args.tasks} tasks x {args.trials} trials x 2 runs"
    print(f"wrote {len(records)} records ({shape}) to {args.out.name}")


if __name__ == "__main__":
    main()
