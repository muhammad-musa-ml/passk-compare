"""Run every passk command behind examples/tau2_baselines/README.md and write the commands with their output.

For each domain it summarizes each run, then compares every pair of runs at k = 1 and k = 4 with the
default margin (0.05), bootstrap replicates (2000), alpha (0.05) and seed (0). In each pair, A is the
run whose name sorts first. Run it from anywhere after `pip install -e .`:

    python examples/tau2_baselines/run_all.py

which rewrites examples/tau2_baselines/results.txt.
"""

import argparse
import contextlib
import io
import itertools
import os
from pathlib import Path

from passk.cli import main as passk_main
from passk.data import load_trials

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOMAINS = ("airline", "retail")
KS = (1, 4)


def commands() -> list[list[str]]:
    """Every passk command, as an argument list, in the order they appear in results.txt."""
    found: list[list[str]] = []
    for domain in DOMAINS:
        file = f"examples/tau2_baselines/{domain}.jsonl"
        runs = sorted(load_trials(ROOT / file))
        found += [["summarize", file, "--run", run] for run in runs]
        for a, b in itertools.combinations(runs, 2):
            found += [["compare", file, "--a", a, "--b", b, "--k", str(k)] for k in KS]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the passk commands for the tau2 baselines and their output."
    )
    parser.add_argument("--out", type=Path, default=HERE / "results.txt", help="file to write")
    args = parser.parse_args()
    os.chdir(ROOT)
    blocks = []
    for argv in commands():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = passk_main(argv)
        if status != 0:
            print(f"passk {' '.join(argv)} exited with status {status}")
            return status
        blocks.append(f"$ passk {' '.join(argv)}\n{buffer.getvalue()}")
    args.out.write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    print(f"wrote {len(blocks)} commands with their output to {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
