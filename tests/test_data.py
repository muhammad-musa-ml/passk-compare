import json
from pathlib import Path

import pytest

from passk import PasskError, load_trials
from passk.data import align_tasks, check_min_trials, format_ids, parse_success, select_run

EXPECTED = {
    "A": {"t01": [True, False, True], "t02": [False, False, True]},
    "B": {"t01": [True, True, True], "t02": [False, True, False]},
}


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def records_from(expected: dict[str, dict[str, list[bool]]]) -> list[dict[str, object]]:
    rows = []
    for run, tasks in expected.items():
        for task, trials in tasks.items():
            for index, success in enumerate(trials):
                rows.append({"run": run, "task": task, "trial": index, "success": success})
    # Out of trial order on purpose: grouping must sort by the trial index.
    return list(reversed(rows))


def test_jsonl_and_csv_load_to_the_same_structure(tmp_path: Path) -> None:
    rows = records_from(EXPECTED)
    jsonl = write_jsonl(tmp_path / "trials.jsonl", rows)
    csv_path = tmp_path / "trials.csv"
    lines = ["run,task,trial,success"] + [
        f"{r['run']},{r['task']},{r['trial']},{str(r['success']).lower()}" for r in rows
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert load_trials(jsonl) == EXPECTED
    assert load_trials(csv_path) == EXPECTED


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("FALSE", False),
        ("Pass", True),
        ("fail", False),
        (" TRUE ", True),
        ("1", True),
        ("0", False),
    ],
)
def test_success_spellings(value: object, expected: bool) -> None:
    assert parse_success(value, "here") is expected


def test_success_spellings_in_csv(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text(
        "run,task,trial,success\nA,t1,0,PASS\nA,t1,1,Fail\nA,t1,2,1\nA,t1,3,0\nA,t1,4,True\n",
        encoding="utf-8",
    )
    assert load_trials(path) == {"A": {"t1": [True, False, True, False, True]}}


@pytest.mark.parametrize("bad", ["maybe", 2, 1.0, "yes", [True]])
def test_bad_success_value_names_the_line(tmp_path: Path, bad: object) -> None:
    path = write_jsonl(
        tmp_path / "t.jsonl",
        [
            {"run": "A", "task": "t1", "trial": 0, "success": True},
            {"run": "A", "task": "t1", "trial": 1, "success": False},
            {"run": "A", "task": "t1", "trial": 2, "success": bad},
        ],
    )
    with pytest.raises(PasskError, match="line 3"):
        load_trials(path)


def test_bad_success_value_in_csv_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("run,task,trial,success\nA,t1,0,true\nA,t1,1,false\nA,t1,2,unknown\n", encoding="utf-8")
    with pytest.raises(PasskError, match="line 4"):
        load_trials(path)


def test_duplicate_key_raises_with_the_key(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "t.jsonl",
        [
            {"run": "A", "task": "t01", "trial": 0, "success": True},
            {"run": "A", "task": "t01", "trial": 1, "success": True},
            {"run": "A", "task": "t01", "trial": 0, "success": False},
        ],
    )
    with pytest.raises(PasskError, match=r"\(A, t01, 0\)"):
        load_trials(path)


def test_missing_trial_index_follows_order_of_appearance(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "t.jsonl",
        [
            {"run": "A", "task": "t1", "success": True},
            {"run": "B", "task": "t1", "success": False},
            {"run": "A", "task": "t1", "success": False},
            {"run": "A", "task": "t1", "success": "pass"},
        ],
    )
    assert load_trials(path) == {"A": {"t1": [True, False, True]}, "B": {"t1": [False]}}


def test_invalid_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text('{"run": "A", "task": "t1", "success": true}\n{"run": "A", \n', encoding="utf-8")
    with pytest.raises(PasskError, match="line 2"):
        load_trials(path)


def test_missing_field_names_the_line(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "t.jsonl", [{"run": "A", "success": True}])
    with pytest.raises(PasskError, match="line 1: missing field 'task'"):
        load_trials(path)


@pytest.mark.parametrize("trial", [-1, "x", 1.5, True])
def test_bad_trial_index_raises(tmp_path: Path, trial: object) -> None:
    path = write_jsonl(tmp_path / "t.jsonl", [{"run": "A", "task": "t1", "trial": trial, "success": True}])
    with pytest.raises(PasskError, match="line 1: trial must be"):
        load_trials(path)


def test_csv_without_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("run,trial,success\nA,0,true\n", encoding="utf-8")
    with pytest.raises(PasskError, match="lacks task"):
        load_trials(path)


def test_file_problems_raise(tmp_path: Path) -> None:
    with pytest.raises(PasskError, match="cannot read"):
        load_trials(tmp_path / "missing.jsonl")
    with pytest.raises(PasskError, match="unsupported file type"):
        load_trials(tmp_path / "trials.txt")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(PasskError, match="no trial records"):
        load_trials(empty)


def test_utf8_byte_order_mark_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("run,task,trial,success\nA,t1,0,true\n", encoding="utf-8-sig")
    assert load_trials(path) == {"A": {"t1": [True]}}


def test_format_ids_truncates_after_twenty() -> None:
    ids = [f"t{i:02d}" for i in range(25)]
    text = format_ids(ids)
    assert text.startswith("t00, t01")
    assert "t19" in text and "t20" not in text
    assert text.endswith("... and 5 more")
    assert format_ids(["a", "b"]) == "a, b"


def test_select_run_names_the_runs_that_exist() -> None:
    with pytest.raises(PasskError, match="run 'Z' not found; runs in the file: A, B"):
        select_run(EXPECTED, "Z")


def test_check_min_trials_lists_short_tasks() -> None:
    with pytest.raises(PasskError, match=r"t02 \(n=1\)"):
        check_min_trials({"t01": [True, True], "t02": [False]}, 2, "run A")


def test_align_tasks() -> None:
    a = {"t1": [True], "t2": [False], "t3": [True]}
    b = {"t1": [True], "t2": [True], "t4": [False]}
    with pytest.raises(PasskError, match="missing from B: t3; missing from A: t4"):
        align_tasks(a, b, intersect=False)
    a2, b2, dropped = align_tasks(a, b, intersect=True)
    assert sorted(a2) == sorted(b2) == ["t1", "t2"]
    assert dropped == ["t3", "t4"]
