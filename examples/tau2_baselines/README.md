# passk-compare on the tau2-bench baselines

This folder runs passk-compare on real agent-benchmark results. In [tau2-bench](https://github.com/sierra-research/tau2-bench) (Sierra Research, MIT licence) a language-model agent plays a customer-service representative and a language-model user simulator plays the customer, in a simulated airline or retail domain, and every simulation ends with a reward. The repository ships the result files of the baselines from its original paper, with four trials for each task, which is the input passk-compare needs. I used the airline and retail files of four models: 50 airline tasks and 114 retail tasks.

The numbers on this page come from the commands below, run on those files. The README of the tau2-bench folder that holds them says they are the results of the original paper (June 2025), produced with the June 2025 task sets before the task fixes of version 1.0.0, so the airline and retail tasks in them differ from the current base tasks. These numbers are a measurement of that historical record, not of the current benchmark.

## What is here

| file | what it is |
|---|---|
| `convert.py` | turns tau2-bench result files into the JSONL that `passk` reads |
| `airline.jsonl`, `retail.jsonl` | its output for the eight files: one line per simulation, with task id, trial and success (800 and 1824 lines) |
| `run_all.py` | runs every `passk summarize` and `passk compare` command below |
| `results.txt` | the commands and their output, as written by `run_all.py` |
| `recompute.py` | a second reading of the raw result files that does not use passk-compare, to check the conversion |
| `SOURCES.txt` | the tau2-bench commit, and the sha256 and size of each of the eight input files |

The trajectories themselves are not copied here. The JSONL files hold only the task id, the trial index and whether the simulation succeeded.

## The data

The files are in `data/tau2/results/final/` of tau2-bench at commit `19f13ea3f5fba8b4c4d272d1353590f29dc53062` (the merge of pull request 737, 2 November 2026). For each of the two domains there is one file per model:

- `claude-3-7-sonnet-20250219_<domain>_default_gpt-4.1-2025-04-14_4trials.json`
- `gpt-4.1-2025-04-14_<domain>_default_gpt-4.1-2025-04-14_4trials.json`
- `gpt-4.1-mini-2025-04-14_<domain>_base_gpt-4.1-2025-04-14_4trials.json` (named `base`, not `default`)
- `o4-mini-2025-04-16_<domain>_default_gpt-4.1-2025-04-14_4trials.json`

What I read in the `info` block and the simulations of all eight files:

- Four trials for every task, seed 300, `max_steps` 200 and `max_errors` 10, and the user simulator `gpt-4.1-2025-04-14` at temperature 0.0 in all eight. The agents run at temperature 0.0, except `o4-mini`, which has `reasoning_effort` set to `high`.
- Every simulation ended with the termination reason `user_stop`, none was an infrastructure error, and every reward is exactly 0.0 or 1.0.
- The files were recorded on 5 and 6 June 2025. The two `gpt-4.1-mini` files carry a different `info.git_commit` (`ade39493be54aad326a4c65295f77fe09780329b`) from the other six (`c30d59aaa71c65f9b9eb6a8f8636b48945028fcf`). I did not look up what those two commits are.
- Inside each domain the four files list the same task ids, and the task definitions are identical when I compare them as JSON, so pairing runs on the task id compares the same tasks.

## Converting

`convert.py` applies the two rules that tau2-bench uses itself in `src/tau2/metrics/agent_metrics.py` at that commit: a simulation succeeded when its reward is within 1e-6 of 1, and a simulation whose termination reason is `infrastructure_error` never ran and is left out. Each file becomes one run, named after the model in `info.agent_info.llm`. A simulation with no numeric reward is an error, not a failure.

With tau2-bench checked out at the commit above in `$TAU2`:

```
$ F=$TAU2/data/tau2/results/final
$ python examples/tau2_baselines/convert.py \
    $F/claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json \
    $F/gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json \
    $F/gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json \
    $F/o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json \
    --out examples/tau2_baselines/airline.jsonl
claude-3-7-sonnet-20250219: 50 tasks, 200 records, 100 successes, 0 infrastructure errors left out (claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json)
gpt-4.1-2025-04-14: 50 tasks, 200 records, 112 successes, 0 infrastructure errors left out (gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json)
gpt-4.1-mini-2025-04-14: 50 tasks, 200 records, 101 successes, 0 infrastructure errors left out (gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json)
o4-mini-2025-04-16: 50 tasks, 200 records, 118 successes, 0 infrastructure errors left out (o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json)
wrote 800 records to airline.jsonl
```

The retail command is the same with `retail` in the four file names and `--out examples/tau2_baselines/retail.jsonl`; it writes 1824 records (456 per run). The two JSONL files are committed, so the commands below run without a tau2-bench checkout.

## The runs

`run_all.py` summarizes each of the eight runs, then compares every pair of runs within a domain at k = 1 and k = 4 (1 and 4 are the ends of what four trials allow), with the default 2000 bootstrap replicates, alpha 0.05, seed 0 and margin 0.05. In each pair A is the run whose name sorts first; the pairs and the orientation follow the names, not the results. That makes 24 comparisons, and I applied no correction for that.

```
$ pip install -e .
$ python examples/tau2_baselines/run_all.py
wrote 32 commands with their output to results.txt
```

`results.txt` has all 32 commands and their output. From the `passk summarize` curves (k = 1 to 4 in `results.txt`), these are the values at k = 1 and k = 4; at k = 1 pass@k equals pass^k:

**Airline, 50 tasks**

| run | pass^1 | 95% CI | pass^4 | 95% CI | pass@4 |
|---|---|---|---|---|---|
| claude-3-7-sonnet-20250219 | 0.5000 | [0.3850, 0.6200] | 0.3600 | [0.2200, 0.4800] | 0.6800 |
| gpt-4.1-2025-04-14 | 0.5600 | [0.4400, 0.6700] | 0.4000 | [0.2600, 0.5400] | 0.7800 |
| gpt-4.1-mini-2025-04-14 | 0.5050 | [0.3900, 0.6150] | 0.2600 | [0.1400, 0.3800] | 0.6800 |
| o4-mini-2025-04-16 | 0.5900 | [0.4700, 0.7000] | 0.3800 | [0.2400, 0.5200] | 0.7600 |

**Retail, 114 tasks**

| run | pass^1 | 95% CI | pass^4 | 95% CI | pass@4 |
|---|---|---|---|---|---|
| claude-3-7-sonnet-20250219 | 0.7873 | [0.7303, 0.8399] | 0.5965 | [0.5088, 0.6754] | 0.9298 |
| gpt-4.1-2025-04-14 | 0.7412 | [0.6776, 0.8004] | 0.5263 | [0.4386, 0.6140] | 0.9035 |
| gpt-4.1-mini-2025-04-14 | 0.6601 | [0.5965, 0.7215] | 0.3860 | [0.2982, 0.4737] | 0.8509 |
| o4-mini-2025-04-16 | 0.7149 | [0.6513, 0.7763] | 0.4561 | [0.3596, 0.5439] | 0.9123 |

The paired comparisons. Each cell is the difference B - A in pass^k, its 95% interval, and the verdict at margin 0.05:

**Airline, 50 tasks**

| A | B | k = 1 | k = 4 |
|---|---|---|---|
| claude-3-7-sonnet-20250219 | gpt-4.1-2025-04-14 | +0.0600 [-0.0500, 0.1700] non_inferior | +0.0400 [-0.1000, 0.1800] inconclusive |
| claude-3-7-sonnet-20250219 | gpt-4.1-mini-2025-04-14 | +0.0050 [-0.1000, 0.1100] inconclusive | -0.1000 [-0.2600, 0.0600] inconclusive |
| claude-3-7-sonnet-20250219 | o4-mini-2025-04-16 | +0.0900 [-0.0301, 0.2100] non_inferior | +0.0200 [-0.1400, 0.1800] inconclusive |
| gpt-4.1-2025-04-14 | gpt-4.1-mini-2025-04-14 | -0.0550 [-0.1650, 0.0501] inconclusive | -0.1400 [-0.2800, 0.0000] inconclusive |
| gpt-4.1-2025-04-14 | o4-mini-2025-04-16 | +0.0300 [-0.0651, 0.1300] inconclusive | -0.0200 [-0.1600, 0.1400] inconclusive |
| gpt-4.1-mini-2025-04-14 | o4-mini-2025-04-16 | +0.0850 [-0.0251, 0.2050] non_inferior | +0.1200 [0.0000, 0.2400] superior |

**Retail, 114 tasks**

| A | B | k = 1 | k = 4 |
|---|---|---|---|
| claude-3-7-sonnet-20250219 | gpt-4.1-2025-04-14 | -0.0461 [-0.1075, 0.0197] inconclusive | -0.0702 [-0.1667, 0.0178] inconclusive |
| claude-3-7-sonnet-20250219 | gpt-4.1-mini-2025-04-14 | -0.1272 [-0.1908, -0.0614] inferior | -0.2105 [-0.3158, -0.1140] inferior |
| claude-3-7-sonnet-20250219 | o4-mini-2025-04-16 | -0.0724 [-0.1382, -0.0043] inconclusive | -0.1404 [-0.2456, -0.0349] inconclusive |
| gpt-4.1-2025-04-14 | gpt-4.1-mini-2025-04-14 | -0.0811 [-0.1316, -0.0329] inconclusive | -0.1404 [-0.2368, -0.0439] inferior |
| gpt-4.1-2025-04-14 | o4-mini-2025-04-16 | -0.0263 [-0.0965, 0.0461] inconclusive | -0.0702 [-0.1842, 0.0351] inconclusive |
| gpt-4.1-mini-2025-04-14 | o4-mini-2025-04-16 | +0.0548 [-0.0066, 0.1228] superior | +0.0702 [-0.0351, 0.1754] non_inferior |

## What the intervals show

Of the 24 comparisons, 15 are inconclusive, 4 non_inferior, 3 inferior and 2 superior. A verdict is a statement about the margin, so `inconclusive` does not mean "no difference": it means the data neither show B to be better than A nor rule out B being worse by more than 0.05. The 95% interval and the verdict can differ because the verdict uses the one-sided 95% bounds, which are printed in `results.txt`.

**Airline.** At k = 1 the gaps between runs are 0.5 to 9.0 points, and no 95% interval for the difference excludes zero. The widest gap, claude-3-7-sonnet-20250219 to o4-mini-2025-04-16 (+0.0900), has the interval [-0.0301, 0.2100]. At k = 4 the gaps are 2 to 14 points and again no interval excludes zero; two end exactly at zero (gpt-4.1 against gpt-4.1-mini has an upper end of 0.0000, gpt-4.1-mini against o4-mini a lower end of 0.0000). The one `superior` verdict, gpt-4.1-mini against o4-mini at k = 4, rests on a one-sided lower bound of +0.0200; its two-sided interval starts at 0.0000 and its permutation p is 0.1139. On these 50 tasks and four trials, the files do not separate any two of the models at either k.

**Retail.** At k = 1 three of the six intervals exclude zero: claude-3-7-sonnet-20250219 against gpt-4.1-mini-2025-04-14 (-12.72 points, `inferior`), claude-3-7-sonnet-20250219 against o4-mini-2025-04-16 (-7.24 points, interval [-0.1382, -0.0043]) and gpt-4.1 against gpt-4.1-mini (-8.11 points). The last two are `inconclusive` because the verdict asks whether B is worse by more than 0.05, and their one-sided upper bounds (-0.0132 and -0.0395) are above -0.05. The other three k = 1 gaps are 2.6 to 5.5 points and their intervals include zero; gpt-4.1-mini against o4-mini (+0.0548) is `superior` on the one-sided lower bound of +0.0022 while its two-sided interval starts at -0.0066. At k = 4 the same three pairs are separated from zero, with gaps of 21.05, 14.04 and 14.04 points, and the three 7.02-point gaps are not.

## Single passes and issue 540

[Issue 540](https://github.com/sierra-research/tau2-bench/issues/540) in the tau2-bench repository asks how far apart two scores must be before the gap is larger than repetition alone produces, and works from the published result files. Its table gives, for the June 2025 baselines, 10 to 11 points between two single runs on the 50 airline tasks and 7 to 8 points on the 114 retail tasks, and 32 to 42% (airline) and 33 to 46% (retail) of tasks whose verdict flips across identical runs. I did not read the issue's code and I did not try to reproduce its first column. Here is what I can set beside it.

Each trial index in these files is one pass over all the tasks, so each file holds four single passes of one model. `recompute.py` scores them and counts the tasks whose four trials do not all agree:

**Airline**

| run | pass^1 of each single pass, trials 0 to 3 | largest gap | tasks whose trials disagree |
|---|---|---|---|
| claude-3-7-sonnet-20250219 | 0.5400, 0.5000, 0.5000, 0.4600 | 0.0800 | 16 of 50 (0.3200) |
| gpt-4.1-2025-04-14 | 0.5800, 0.5400, 0.6000, 0.5200 | 0.0800 | 19 of 50 (0.3800) |
| gpt-4.1-mini-2025-04-14 | 0.5000, 0.5200, 0.5000, 0.5000 | 0.0200 | 21 of 50 (0.4200) |
| o4-mini-2025-04-16 | 0.5600, 0.6200, 0.6600, 0.5200 | 0.1400 | 19 of 50 (0.3800) |

**Retail**

| run | pass^1 of each single pass, trials 0 to 3 | largest gap | tasks whose trials disagree |
|---|---|---|---|
| claude-3-7-sonnet-20250219 | 0.7982, 0.7807, 0.7807, 0.7895 | 0.0175 | 38 of 114 (0.3333) |
| gpt-4.1-2025-04-14 | 0.7368, 0.7105, 0.7544, 0.7632 | 0.0526 | 43 of 114 (0.3772) |
| gpt-4.1-mini-2025-04-14 | 0.6842, 0.6754, 0.6053, 0.6754 | 0.0789 | 53 of 114 (0.4649) |
| o4-mini-2025-04-16 | 0.6930, 0.7281, 0.7368, 0.7018 | 0.0439 | 52 of 114 (0.4561) |

- Two single passes of the same model landed up to 14 points apart on airline (o4-mini-2025-04-16: 0.5200 and 0.6600) and up to 7.9 points apart on retail (gpt-4.1-mini-2025-04-14: 0.6053 and 0.6842). On airline that spread inside one model is larger than the widest k = 1 gap between two models (9.0 points). Both spreads are close in size to the 10 to 11 and 7 to 8 points in the issue; I did not compute the issue's quantity.
- The share of tasks whose four trials do not all agree is 32% to 42% on airline and 33% to 46% on retail, the same ranges as the last column of the issue's table. I read "verdict flips across identical runs" as "the four trials do not all agree"; that reading is mine, not the issue's definition.
- The half-widths of the pass^1 intervals for the differences are 9.75 to 12.0 points on airline and 4.9 to 7.1 points on retail, in the range of the 10 to 11 and 7 to 8 points of the issue and a little below it on retail. The two numbers answer different questions, so I take this as consistent with the issue and not as a replication. The issue's figures are for single runs and cover repetition only; the pass^1 here is a mean over four trials, which should carry less repetition noise than one run, and the bootstrap resamples tasks, which adds the noise from which tasks were drawn.

What passk-compare adds is an interval for the gap between two particular models, using all four trials of every task, so the reader sees where a gap falls: on retail, 7.24 points between claude-3-7-sonnet-20250219 and o4-mini-2025-04-16 has an interval that just excludes zero, and on airline 9.0 points between the same two models has an interval that includes it.

## Checking the conversion

`recompute.py` reads the eight raw result files again, with its own code, and computes pass^k as comb(c, k) / comb(n, k) per task and the mean over tasks, as in tau2-bench's `pass_hat_k`, and pass@k as 1 - comb(n - c, k) / comb(n, k). With `--check-jsonl` it compares the number of trials and successes of every (run, task) pair in a JSONL file with the counts it reads from the raw files:

```
$ python examples/tau2_baselines/recompute.py $F/*_airline_*_4trials.json --check-jsonl examples/tau2_baselines/airline.jsonl
...
airline.jsonl: successes and trials agree for all 200 (run, task) pairs
```

For retail the same check reports 456 pairs. I also compared passk's point estimates from the JSONL files with `recompute.py`'s values for every run, k from 1 to 4 and both metrics (64 values) at full precision, and they are identical. This checks the conversion and the code path, not the formulas: `recompute.py` uses the same formulas as passk-compare and tau2-bench, and I did not run tau2-bench's own metrics code, which needs pandas, pydantic and loguru.

`tests/test_tau2_baselines.py` builds small fake result files inside the test, checks the success tolerance on both sides, the skipping of infrastructure errors, the error on a missing reward, the check on a changed or missing JSONL record, and the totals of the two committed JSONL files.

## Limits

- The task sets are the June 2025 ones, which differ from the current base tasks. Nothing here says how these models do on the benchmark as it is now.
- Four trials per task. pass^4 of a task is 0 or 1, so the k = 4 intervals come from 50 or 114 tasks that each give a yes or no, and they are wide.
- The user simulator is part of every result. All eight files use the same simulator model at temperature 0.0, and the trials of a task still differ, so the intervals include its noise and say nothing about how much comes from the agent and how much from the simulator.
- The two `gpt-4.1-mini` files are named `base` and carry a different `info.git_commit` from the other three files in each domain. The task definitions match, and I cannot say what else differs between those runs.
- The intervals are plain percentile bootstrap intervals with tasks as the resampling unit, as described in the main README. With 50 airline tasks they are wide.
- The 24 comparisons have no correction for multiple comparisons, and the pairs are not baseline against candidate: A is only the run whose name sorts first.
- A simulation counts as a success when its reward is within 1e-6 of 1, as in tau2-bench. Every reward in these files is exactly 0.0 or 1.0, so the tolerance changes nothing here.
