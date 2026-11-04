"""Convert tau2-bench result files into passk-compare trial records.

A tau2-bench result file is one JSON object with `info`, `tasks` and `simulations`. Each simulation
has a `task_id`, a `trial` index, a `termination_reason` and `reward_info.reward`. This script
applies the two rules that tau2-bench uses itself in src/tau2/metrics/agent_metrics.py:

* a simulation succeeded when 1 - 1e-6 <= reward <= 1 + 1e-6 (`is_successful` there);
* a simulation that ended with termination_reason "infrastructure_error" never ran and is left out
  (`get_metrics_df` there).

Every input file becomes one run, named after `info.agent_info.llm`, and every kept simulation
becomes one line {"run": ..., "task": ..., "trial": ..., "success": ...}. Standard library only.

    python examples/tau2_baselines/convert.py FILE [FILE ...] --out examples/tau2_baselines/airline.jsonl
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

TOLERANCE = 1e-6
INFRASTRUCTURE_ERROR = "infrastructure_error"


class ConversionError(ValueError):
    """A result file that cannot be converted: a missing field, a missing reward or a repeated run name."""


def is_successful(reward: float) -> bool:
    """tau2-bench's success rule: the reward equals 1 within 1e-6."""
    return (1 - TOLERANCE) <= reward <= (1 + TOLERANCE)


def run_name(data: dict[str, object], where: str) -> str:
    """The run's name: the model in info.agent_info.llm."""
    info = data.get("info")
    agent = info.get("agent_info") if isinstance(info, dict) else None
    name = agent.get("llm") if isinstance(agent, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise ConversionError(f"{where}: info.agent_info.llm is missing, so the run has no name")
    return name.strip()


def convert_result(data: dict[str, object], where: str) -> tuple[str, list[dict[str, object]], int]:
    """Turn one parsed result file into (run name, trial records, infrastructure errors left out)."""
    run = run_name(data, where)
    simulations = data.get("simulations")
    if not isinstance(simulations, list) or not simulations:
        raise ConversionError(f"{where}: no simulations")
    records: list[dict[str, object]] = []
    skipped = 0
    for index, simulation in enumerate(simulations):
        if not isinstance(simulation, dict):
            raise ConversionError(f"{where}: simulation {index} is not an object")
        task, trial = simulation.get("task_id"), simulation.get("trial")
        if task is None or isinstance(trial, bool) or not isinstance(trial, int):
            raise ConversionError(f"{where}: simulation {index} needs a task_id and an integer trial")
        if simulation.get("termination_reason") == INFRASTRUCTURE_ERROR:
            skipped += 1
            continue
        info = simulation.get("reward_info")
        reward = info.get("reward") if isinstance(info, dict) else None
        if isinstance(reward, bool) or not isinstance(reward, int | float):
            raise ConversionError(f"{where}: task {task} trial {trial} has no numeric reward")
        records.append({"run": run, "task": str(task), "trial": trial, "success": is_successful(reward)})
    return run, records, skipped


def convert_files(paths: Sequence[Path]) -> tuple[list[dict[str, object]], list[str]]:
    """Convert files in the order given. Returns all records and one summary line per file."""
    records: list[dict[str, object]] = []
    summaries: list[str] = []
    source: dict[str, str] = {}
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversionError(f"cannot read {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConversionError(f"{path.name}: expected a JSON object")
        run, converted, skipped = convert_result(data, path.name)
        if run in source:
            raise ConversionError(f"run {run!r} comes from both {source[run]} and {path.name}")
        source[run] = path.name
        tasks = len({str(record["task"]) for record in converted})
        passed = sum(1 for record in converted if record["success"])
        summaries.append(
            f"{run}: {tasks} tasks, {len(converted)} records, {passed} successes, "
            f"{skipped} infrastructure errors left out ({path.name})"
        )
        records.extend(converted)
    return records, summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert tau2-bench result files to passk-compare JSONL.")
    parser.add_argument("files", nargs="+", type=Path, help="tau2-bench result files, one run each")
    parser.add_argument("--out", type=Path, required=True, help="JSONL file to write")
    args = parser.parse_args(argv)
    try:
        records, summaries = convert_files(args.files)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    except (ConversionError, OSError) as exc:
        print(f"convert: error: {exc}", file=sys.stderr)
        return 2
    print("\n".join(summaries))
    print(f"wrote {len(records)} records to {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
