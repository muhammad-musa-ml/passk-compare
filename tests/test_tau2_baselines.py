"""Tests for examples/tau2_baselines: the converter, the recomputation and the committed JSONL files.

The result files are built here from a few hand-written simulations; no tau2-bench data is needed.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from passk import load_trials
from passk.estimators import pass_at_k, pass_hat_k, run_mean

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "examples" / "tau2_baselines"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tau2_baselines_{name}", FOLDER / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = load_script("convert")
recompute = load_script("recompute")


def simulation(
    task: object, trial: int, reward: float | None, reason: str = "user_stop"
) -> dict[str, object]:
    reward_info = None if reward is None else {"reward": reward}
    return {"task_id": task, "trial": trial, "termination_reason": reason, "reward_info": reward_info}


def fake_result(llm: str, flip: bool = False) -> dict[str, object]:
    """Three tasks. Task a passes 3 of 3, b passes 2 of 3 (one reward is 0.9999995, inside the
    tolerance), c has two failed trials (rewards 0.5 and 1.00001, outside it) and one trial that
    ended in an infrastructure error, which is left out."""
    sims = [
        simulation("a", 0, 1.0),
        simulation("a", 1, 1.0),
        simulation("a", 2, 0.0 if flip else 1.0),
        simulation("b", 0, 1.0),
        simulation("b", 1, 0.0),
        simulation("b", 2, 0.9999995),
        simulation("c", 0, 0.5),
        simulation("c", 1, 1.00001),
        simulation("c", 2, 1.0, "infrastructure_error"),
    ]
    return {"info": {"agent_info": {"llm": llm}}, "tasks": [], "simulations": sims}


def write_result(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("reward", [1.0, 0.9999995, 1.0000005, 1 - 1e-6, 1 + 1e-6])
def test_rewards_within_the_tolerance_succeed(reward: float) -> None:
    assert convert.is_successful(reward)


@pytest.mark.parametrize("reward", [0.0, 0.5, 0.99999, 1.00001, 2.0])
def test_rewards_outside_the_tolerance_fail(reward: float) -> None:
    assert not convert.is_successful(reward)


def test_convert_result_keeps_the_order_and_leaves_out_infrastructure_errors() -> None:
    run, records, skipped = convert.convert_result(fake_result("m"), "fake.json")
    assert (run, skipped) == ("m", 1)
    assert all(record["run"] == "m" for record in records)
    assert [(r["task"], r["trial"], r["success"]) for r in records] == [
        ("a", 0, True),
        ("a", 1, True),
        ("a", 2, True),
        ("b", 0, True),
        ("b", 1, False),
        ("b", 2, True),
        ("c", 0, False),
        ("c", 1, False),
    ]


def test_integer_task_ids_become_strings() -> None:
    data = fake_result("m")
    data["simulations"] = [simulation(7, 0, 1.0)]
    assert convert.convert_result(data, "fake.json")[1][0]["task"] == "7"


def test_a_simulation_without_a_reward_is_an_error_that_names_it() -> None:
    data = fake_result("m")
    data["simulations"] = [simulation("a", 0, 1.0), simulation("a", 1, None)]
    with pytest.raises(convert.ConversionError, match="task a trial 1 has no numeric reward"):
        convert.convert_result(data, "fake.json")


def test_a_file_without_a_model_name_is_an_error() -> None:
    data = fake_result("m")
    data["info"] = {"agent_info": {}}
    with pytest.raises(convert.ConversionError, match="no name"):
        convert.convert_result(data, "fake.json")


def test_two_files_with_the_same_model_name_are_refused(tmp_path: Path) -> None:
    first = write_result(tmp_path / "one.json", fake_result("m"))
    second = write_result(tmp_path / "two.json", fake_result("m"))
    with pytest.raises(convert.ConversionError, match="from both one.json and two.json"):
        convert.convert_files([first, second])


def convert_two_runs(tmp_path: Path) -> tuple[list[Path], Path]:
    files = [
        write_result(tmp_path / "m.json", fake_result("m")),
        write_result(tmp_path / "n.json", fake_result("n", flip=True)),
    ]
    out = tmp_path / "out" / "runs.jsonl"
    done = subprocess.run(
        [sys.executable, str(FOLDER / "convert.py"), *map(str, files), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return files, out


def test_the_command_line_writes_records_passk_can_read(tmp_path: Path) -> None:
    _, out = convert_two_runs(tmp_path)
    data = load_trials(out)
    assert sorted(data) == ["m", "n"]
    assert data["m"] == {"a": [True, True, True], "b": [True, False, True], "c": [False, False]}
    assert data["n"]["a"] == [True, True, False]
    # Hand-computed: task a gives 1, b gives 1/3 and c gives 0, so pass^2 is (1 + 1/3 + 0) / 3.
    assert run_mean(data["m"], 2, pass_hat_k) == pytest.approx(4 / 9)
    assert run_mean(data["m"], 1, pass_hat_k) == pytest.approx(5 / 9)
    assert run_mean(data["m"], 2, pass_at_k) == pytest.approx(2 / 3)


def test_the_command_line_reports_a_bad_file_with_status_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(FOLDER / "convert.py"), str(bad), "--out", str(tmp_path / "x.jsonl")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert "convert: error: cannot read" in done.stderr
    assert not (tmp_path / "x.jsonl").exists()


def test_recompute_matches_passk_on_every_k(tmp_path: Path) -> None:
    files, out = convert_two_runs(tmp_path)
    data = load_trials(out)
    for path in files:
        run, tasks, skipped = recompute.read_run(path)
        counts = recompute.counts_of(tasks)
        assert skipped == 1
        assert counts == {"a": (3, 3 if run == "m" else 2), "b": (3, 2), "c": (2, 0)}
        for k in (1, 2):
            assert recompute.pass_hat(counts, k) == pytest.approx(
                run_mean(data[run], k, pass_hat_k), abs=1e-12
            )
            assert recompute.pass_at(counts, k) == pytest.approx(run_mean(data[run], k, pass_at_k), abs=1e-12)


def test_mixed_tasks_are_those_with_both_a_success_and_a_failure(tmp_path: Path) -> None:
    path = write_result(tmp_path / "m.json", fake_result("m"))
    _, tasks, _ = recompute.read_run(path)
    # Task a passed every trial, c failed every trial, and only b has both outcomes.
    assert recompute.mixed_tasks(recompute.counts_of(tasks)) == 1


def test_single_passes_use_only_the_tasks_that_have_the_trial(tmp_path: Path) -> None:
    path = write_result(tmp_path / "m.json", fake_result("m"))
    _, tasks, _ = recompute.read_run(path)
    # Trial 2 of task c was an infrastructure error, so trial 2 is scored on tasks a and b only.
    assert recompute.single_passes(tasks) == pytest.approx({0: 2 / 3, 1: 1 / 3, 2: 1.0})


def test_check_jsonl_agrees_with_the_converter_and_finds_a_changed_or_missing_record(tmp_path: Path) -> None:
    files, out = convert_two_runs(tmp_path)
    expected = {}
    for path in files:
        run, tasks, _ = recompute.read_run(path)
        expected[run] = recompute.counts_of(tasks)
    assert recompute.check_jsonl(out, expected) == []
    lines = out.read_text(encoding="utf-8").splitlines()
    changed = [json.loads(line) for line in lines]
    for record in changed:
        if (record["run"], record["task"], record["trial"]) == ("m", "b", 1):
            record["success"] = True
    out.write_text("".join(json.dumps(record) + "\n" for record in changed), encoding="utf-8")
    assert recompute.check_jsonl(out, expected) == ["run m, task b: JSONL has (3, 3), results have (3, 2)"]
    out.write_text(
        "".join(json.dumps(record) + "\n" for record in changed if record["task"] != "c"), encoding="utf-8"
    )
    problems = recompute.check_jsonl(out, expected)
    assert "missing from the JSONL: run m, task c" in problems
    assert "missing from the JSONL: run n, task c" in problems


def test_recompute_command_checks_the_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files, out = convert_two_runs(tmp_path)
    assert recompute.main([*map(str, files), "--check-jsonl", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "m: 3 tasks, 2 to 3 trials per task, 1 infrastructure errors left out" in printed
    assert "runs.jsonl: successes and trials agree for all 6 (run, task) pairs" in printed


# Successes per run in the committed files, counted from the rewards in the tau2-bench result files
# (every reward in them is 0.0 or 1.0).
COMMITTED = {
    "airline.jsonl": (
        50,
        {
            "claude-3-7-sonnet-20250219": 100,
            "gpt-4.1-2025-04-14": 112,
            "gpt-4.1-mini-2025-04-14": 101,
            "o4-mini-2025-04-16": 118,
        },
    ),
    "retail.jsonl": (
        114,
        {
            "claude-3-7-sonnet-20250219": 359,
            "gpt-4.1-2025-04-14": 338,
            "gpt-4.1-mini-2025-04-14": 301,
            "o4-mini-2025-04-16": 326,
        },
    ),
}


@pytest.mark.parametrize("name", sorted(COMMITTED))
def test_committed_baseline_files_have_the_documented_shape(name: str) -> None:
    tasks, successes = COMMITTED[name]
    data = load_trials(FOLDER / name)
    assert sorted(data) == sorted(successes)
    for run, trials_by_task in data.items():
        assert len(trials_by_task) == tasks
        assert {len(trials) for trials in trials_by_task.values()} == {4}
        assert sum(sum(trials) for trials in trials_by_task.values()) == successes[run]
    assert all(set(data[run]) == set(data["gpt-4.1-2025-04-14"]) for run in data)
