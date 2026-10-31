"""Trial records: loading from JSONL or CSV, grouping by run and task, and input validation."""

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

RunTrials = dict[str, list[bool]]
"""One run's results: task id -> trial outcomes in trial-index order."""

MAX_LISTED = 20

_TRUE = {"true", "1", "pass"}
_FALSE = {"false", "0", "fail"}


class PasskError(ValueError):
    """Invalid input: a bad record, mismatched task sets, too few trials, or a bad parameter."""


@dataclass(frozen=True)
class Trial:
    """One trial of one task in one run."""

    run: str
    task: str
    trial: int
    success: bool


def format_ids(ids: Iterable[str], limit: int = MAX_LISTED) -> str:
    """Join ids with commas, showing at most `limit` of them followed by "... and N more"."""
    items = list(ids)
    shown = ", ".join(items[:limit])
    if len(items) > limit:
        return f"{shown} ... and {len(items) - limit} more"
    return shown


def parse_success(value: object, where: str) -> bool:
    """Read a success value: true/false, 1/0, or "true"/"false"/"pass"/"fail" in any letter case."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return value == 1
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise PasskError(f"{where}: success must be true/false, 1/0 or pass/fail, got {value!r}")


def _parse_trial_index(value: object, where: str) -> int | None:
    """Read an optional non-negative trial index; None when the field is absent or blank."""
    if value is None or value == "":
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isascii() and text.isdigit():
            return int(text)
    raise PasskError(f"{where}: trial must be a non-negative integer, got {value!r}")


def _make_trial(raw: Mapping[str, object], where: str, seen: dict[tuple[str, str], int]) -> Trial:
    """Build a Trial from one parsed record.

    A record without a trial index gets the number of records already seen for its (run, task),
    so indexes follow the order of appearance.
    """
    for field in ("run", "task", "success"):
        if raw.get(field) is None or raw.get(field) == "":
            raise PasskError(f"{where}: missing field {field!r}")
    run = str(raw["run"]).strip()
    task = str(raw["task"]).strip()
    index = _parse_trial_index(raw.get("trial"), where)
    count = seen.get((run, task), 0)
    seen[(run, task)] = count + 1
    return Trial(run, task, count if index is None else index, parse_success(raw["success"], where))


def read_jsonl(path: Path) -> list[Trial]:
    """Read one JSON object per line: {"run": ..., "task": ..., "trial": ..., "success": ...}."""
    records: list[Trial] = []
    seen: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{path.name} line {number}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PasskError(f"{where}: not valid JSON ({exc.msg})") from exc
            if not isinstance(raw, dict):
                raise PasskError(f"{where}: expected a JSON object")
            records.append(_make_trial(raw, where, seen))
    return records


def read_csv(path: Path) -> list[Trial]:
    """Read a CSV file whose header has the columns run, task, success and optionally trial."""
    records: list[Trial] = []
    seen: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        missing = {"run", "task", "success"} - set(reader.fieldnames)
        if missing:
            raise PasskError(
                f"{path.name}: CSV header lacks {', '.join(sorted(missing))}; "
                "expected the columns run,task,trial,success"
            )
        for row in reader:
            records.append(_make_trial(row, f"{path.name} line {reader.line_num}", seen))
    return records


def read_records(path: str | Path) -> list[Trial]:
    """Read trial records from a .jsonl or .csv file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".jsonl", ".csv"):
        raise PasskError(f"{path}: unsupported file type {path.suffix!r}; use .jsonl or .csv")
    try:
        records = read_jsonl(path) if suffix == ".jsonl" else read_csv(path)
    except OSError as exc:
        raise PasskError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise PasskError(f"{path}: not UTF-8 text ({exc.reason})") from exc
    if not records:
        raise PasskError(f"{path}: no trial records")
    return records


def group_trials(records: Iterable[Trial]) -> dict[str, RunTrials]:
    """Group records into {run: {task: [success, ...]}}, trials sorted by index.

    Raises PasskError listing every (run, task, trial) key that appears more than once.
    """
    slots: dict[tuple[str, str], dict[int, bool]] = {}
    duplicates: dict[str, None] = {}
    for record in records:
        slot = slots.setdefault((record.run, record.task), {})
        if record.trial in slot:
            duplicates[f"({record.run}, {record.task}, {record.trial})"] = None
        slot[record.trial] = record.success
    if duplicates:
        raise PasskError(f"duplicate (run, task, trial) keys: {format_ids(duplicates)}")
    grouped: dict[str, RunTrials] = {}
    for (run, task), slot in slots.items():
        grouped.setdefault(run, {})[task] = [slot[index] for index in sorted(slot)]
    return grouped


def load_trials(path: str | Path) -> dict[str, RunTrials]:
    """Load a .jsonl or .csv file into {run: {task: [success, ...]}}."""
    return group_trials(read_records(path))


def select_run(data: Mapping[str, RunTrials], name: str) -> RunTrials:
    """Return one run from loaded data, or raise PasskError naming the runs that do exist."""
    if name not in data:
        raise PasskError(f"run {name!r} not found; runs in the file: {format_ids(sorted(data))}")
    return data[name]


def check_run(run: Mapping[str, Sequence[bool]], who: str) -> None:
    """Raise PasskError if the run has no tasks or has a task with no trials."""
    if not run:
        raise PasskError(f"{who} is empty: it has no tasks")
    empty = sorted(task for task, trials in run.items() if not trials)
    if empty:
        raise PasskError(f"{who} has {len(empty)} task(s) with no trials: {format_ids(empty)}")


def check_min_trials(run: Mapping[str, Sequence[bool]], k: int, who: str) -> None:
    """Raise PasskError listing every task of the run with fewer than k trials."""
    short = [f"{task} (n={len(trials)})" for task, trials in sorted(run.items()) if len(trials) < k]
    if short:
        raise PasskError(f"{who} has {len(short)} task(s) with fewer than k={k} trials: {format_ids(short)}")


def align_tasks(
    a: Mapping[str, Sequence[bool]], b: Mapping[str, Sequence[bool]], intersect: bool
) -> tuple[RunTrials, RunTrials, list[str]]:
    """Restrict runs A and B to the same task ids.

    Without `intersect`, differing task sets raise PasskError naming the ids missing on each side.
    With it, tasks not shared by both runs are dropped. Returns (a, b, dropped_task_ids).
    """
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if (only_a or only_b) and not intersect:
        parts = []
        if only_a:
            parts.append(f"missing from B: {format_ids(only_a)}")
        if only_b:
            parts.append(f"missing from A: {format_ids(only_b)}")
        raise PasskError(
            "task sets differ between A and B; "
            + "; ".join(parts)
            + "; use --intersect (intersect=True) to compare on the shared tasks only"
        )
    shared = sorted(set(a) & set(b))
    return {t: list(a[t]) for t in shared}, {t: list(b[t]) for t in shared}, sorted(only_a + only_b)
