import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from passk import PasskError, cli
from passk.plot import load_pyplot

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
TWO_RUNS = EXAMPLES / "two_runs.jsonl"


def run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if p)
    return subprocess.run(
        [sys.executable, "-m", "passk.cli", *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        check=False,
    )


def test_summarize_prints_both_metrics() -> None:
    done = run_cli("summarize", TWO_RUNS, "--run", "A", "--reps", "300")
    assert done.returncode == 0, done.stderr
    assert "pass^k" in done.stdout
    assert "pass@k" in done.stdout
    assert done.stderr == ""


def test_summarize_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "summary.json"
    done = run_cli("summarize", TWO_RUNS, "--run", "B", "--k", "1", "4", "--reps", "200", "--json", out)
    assert done.returncode == 0, done.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["run"] == "B"
    assert [row["k"] for row in data["curve"]] == [1, 4]
    assert data["bootstrap"] == "task"
    assert "200 task-level bootstrap replicates" in done.stdout


def test_json_output_creates_missing_folders(tmp_path: Path) -> None:
    out = tmp_path / "out" / "nested" / "a.json"
    done = run_cli("summarize", TWO_RUNS, "--run", "A", "--k", "1", "--reps", "20", "--json", out)
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["curve"][0]["k"] == 1


def test_inner_resample_flag_selects_the_two_level_bootstrap(tmp_path: Path) -> None:
    out = tmp_path / "summary.json"
    done = run_cli(
        "summarize", TWO_RUNS, "--run", "A", "--k", "4", "--reps", "200", "--inner-resample", "--json", out
    )
    assert done.returncode == 0, done.stderr
    assert "200 two-level bootstrap replicates" in done.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["bootstrap"] == "two-level"
    out = tmp_path / "compare.json"
    done = run_cli(
        "compare",
        TWO_RUNS,
        "--a",
        "A",
        "--b",
        "B",
        "--k",
        "2",
        "--reps",
        "100",
        "--permutations",
        "100",
        "--inner-resample",
        "--json",
        out,
    )
    assert done.returncode == 0, done.stderr
    assert "two-level bootstrap replicates" in done.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["bootstrap"] == "two-level"


def test_compare_prints_verdict_and_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "compare.json"
    done = run_cli(
        "compare",
        TWO_RUNS,
        "--a",
        "A",
        "--b",
        "B",
        "--k",
        "2",
        "--reps",
        "300",
        "--permutations",
        "300",
        "--json",
        out,
    )
    assert done.returncode == 0, done.stderr
    assert any(line.startswith("verdict:") for line in done.stdout.splitlines())
    data = json.loads(out.read_text(encoding="utf-8"))
    for key in (
        "k",
        "margin",
        "alpha",
        "reps",
        "seed",
        "n_tasks",
        "a",
        "b",
        "diff",
        "verdict",
        "permutation_p",
        "dropped_tasks",
        "bootstrap",
    ):
        assert key in data
    assert data["bootstrap"] == "task"
    for side in ("a", "b"):
        for name in ("pass_hat_k", "pass_at_k"):
            assert {"point", "ci_low", "ci_high"} <= set(data[side][name])
    assert {"point", "ci_low", "ci_high", "lower_bound_one_sided"} <= set(data["diff"])
    assert data["verdict"] in {"superior", "non_inferior", "inferior", "inconclusive"}
    assert data["k"] == 2 and data["reps"] == 300 and data["n_tasks"] == 60


def test_missing_run_exits_2_with_message() -> None:
    done = run_cli("summarize", TWO_RUNS, "--run", "Z")
    assert done.returncode == 2
    assert "run 'Z' not found" in done.stderr
    assert done.stdout == ""
    done = run_cli("compare", TWO_RUNS, "--a", "A", "--b", "nope", "--k", "1")
    assert done.returncode == 2
    assert "run 'nope' not found" in done.stderr


def test_missing_file_exits_2(tmp_path: Path) -> None:
    done = run_cli("summarize", tmp_path / "absent.jsonl", "--run", "A")
    assert done.returncode == 2
    assert "cannot read" in done.stderr


def test_k_above_the_trial_count_exits_2() -> None:
    done = run_cli("compare", TWO_RUNS, "--a", "A", "--b", "B", "--k", "9", "--reps", "10")
    assert done.returncode == 2
    assert "fewer than k=9 trials" in done.stderr


def test_intersect_reports_dropped_tasks(tmp_path: Path) -> None:
    lines = TWO_RUNS.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not ('"run": "B"' in line and '"task": "t60"' in line)]
    path = tmp_path / "partial.jsonl"
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    done = run_cli(
        "compare", path, "--a", "A", "--b", "B", "--k", "1", "--reps", "50", "--permutations", "50"
    )
    assert done.returncode == 2
    assert "missing from B: t60" in done.stderr
    done = run_cli(
        "compare",
        path,
        "--a",
        "A",
        "--b",
        "B",
        "--k",
        "1",
        "--reps",
        "50",
        "--permutations",
        "50",
        "--intersect",
    )
    assert done.returncode == 0, done.stderr
    assert "dropped 1 task(s) not present in both runs" in done.stdout


@pytest.mark.parametrize(
    ("name", "extra"),
    [("two_runs.jsonl", []), ("small_runs.jsonl", ["--tasks", "15", "--trials", "3", "--lift", "0.05"])],
)
def test_example_files_are_reproduced_by_the_generator(tmp_path: Path, name: str, extra: list[str]) -> None:
    out = tmp_path / name
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / "make_example.py"), *extra, "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert out.read_bytes() == (EXAMPLES / name).read_bytes()


def test_plot_without_matplotlib_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(PasskError, match="needs matplotlib"):
        load_pyplot()
    out = tmp_path / "curve.png"
    code = cli.main(
        ["compare", str(TWO_RUNS), "--a", "A", "--b", "B", "--k", "1", "--reps", "10", "--plot", str(out)]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "needs matplotlib" in captured.err
    assert captured.out == ""
    assert not out.exists()


def test_plot_writes_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "curve.png"
    code = cli.main(
        [
            "compare",
            str(TWO_RUNS),
            "--a",
            "A",
            "--b",
            "B",
            "--k",
            "2",
            "--reps",
            "100",
            "--permutations",
            "100",
            "--plot",
            str(out),
        ]
    )
    assert code == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
