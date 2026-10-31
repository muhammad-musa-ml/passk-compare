# passk-compare

passk-compare estimates pass@k and pass^k for a run of an agent benchmark from trial-level results, with confidence intervals from a bootstrap that resamples the tasks. For two runs on the same tasks it estimates the paired difference B - A in pass^k and returns a non-inferiority verdict at a margin you set.

## Why

Agent benchmarks report pass@k, the chance that at least one of k attempts at a task succeeds, and pass^k, the chance that all k attempts succeed. Microsoft's [waza](https://github.com/microsoft/waza) records Wilson intervals and pass^k estimates and reports bootstrap intervals for weighted scores, and its README states: "None of these intervals is a significance test between runs." When I change a prompt or a model and rerun the same tasks, my question is whether the new run is better, about as good, or worse, and an interval around each run on its own does not answer that. pass^k comes from tau-bench (Yao et al. 2024, [arXiv 2406.12045](https://arxiv.org/abs/2406.12045)), and the unbiased pass@k estimator from Chen et al. 2021 ([arXiv 2107.03374](https://arxiv.org/abs/2107.03374)).

## Install

```
pip install -e .
```

It needs Python 3.11 or newer and uses only the standard library. `pip install -e ".[plot]"` adds matplotlib for `--plot`, and `pip install -e ".[dev]"` adds pytest and ruff.

## Data format

One JSON object per line:

```
{"run": "A", "task": "t01", "trial": 0, "success": true}
```

`trial` is optional; without it, trials are numbered in the order they appear within each (run, task). `success` accepts true/false, 1/0, "true"/"false" and "pass"/"fail" in any letter case, and anything else is an error that names the line. A CSV file with this header works the same way:

```
run,task,trial,success
```

Each (run, task, trial) may appear only once.

## Commands

For a task with c successes in n trials, pass@k = 1 - C(n - c, k) / C(n, k) and pass^k = C(c, k) / C(n, k), both for k <= n. The value for a run is the mean of the per-task values over its tasks.

`passk summarize` prints the curve of one run for k = 1 up to the fewest trials of any task (at most 10), for the values given with `--k`, or up to `--k-max`. On the example file, with the defaults of 2000 bootstrap replicates, alpha 0.05 and seed 0:

```
$ passk summarize examples/two_runs.jsonl --run A
run A in examples/two_runs.jsonl: 60 tasks, 8 trials per task, 2000 task-level bootstrap replicates, seed 0
k  pass@k  95% CI            pass^k  95% CI
1  0.5750  [0.5125, 0.6417]  0.5750  [0.5125, 0.6417]
2  0.7804  [0.7232, 0.8369]  0.3696  [0.2928, 0.4506]
3  0.8777  [0.8274, 0.9220]  0.2616  [0.1812, 0.3467]
4  0.9271  [0.8828, 0.9626]  0.2014  [0.1224, 0.2895]
5  0.9545  [0.9128, 0.9836]  0.1670  [0.0884, 0.2551]
6  0.9702  [0.9304, 0.9946]  0.1476  [0.0696, 0.2375]
7  0.9792  [0.9417, 1.0000]  0.1375  [0.0563, 0.2250]
8  0.9833  [0.9500, 1.0000]  0.1333  [0.0500, 0.2167]
```

`passk compare` compares run B with run A at one k. The difference, its interval, the verdict and the permutation test are on pass^k, and pass@k is shown for each run beside it. The default margin is 0.05 and the default number of sign flips is 2000.

```
$ passk compare examples/two_runs.jsonl --a A --b B --k 4
A = A, B = B, k = 4: 60 shared tasks, 2000 task-level bootstrap replicates, seed 0
metric  run  point   95% CI
pass^k  A    0.2014  [0.1224, 0.2895]
pass^k  B    0.2257  [0.1443, 0.3200]
pass@k  A    0.9271  [0.8828, 0.9626]
pass@k  B    0.8850  [0.8240, 0.9381]
diff B - A in pass^k: 0.0243, 95% CI [-0.0362, 0.0848]
one-sided 95% bounds: lower -0.0255, upper 0.0762; margin 0.05
paired permutation p (pass^k, 2000 sign flips): 0.4288
verdict: non_inferior
```

The example was generated with B's per-task success probability 0.03 above A's. B's pass^4 comes out 0.0243 higher, and the one-sided lower bound of the difference, -0.0255, is above -0.05 but not above 0, so the verdict is non_inferior: on these 60 tasks the data rule out B being worse than A by more than 0.05 in pass^4, and they do not show that B is better. pass@4 points the other way here (0.8850 for B against 0.9271 for A), which is one more reason to choose the metric and the k before looking at results.

Both commands take `--json OUT` to write the full result, including a `bootstrap` field (`"task"`, or `"two-level"` with `--inner-resample`, described below) and the mean of the bootstrap replicates for each interval as `replicate_mean`. `compare` also takes `--intersect` to drop tasks that are missing from either run instead of stopping with an error, and `--plot OUT.png` to draw pass^k against k with bands for both runs (this needs matplotlib). Invalid input exits with status 2 and a message on stderr.

## How the interval is computed

Each bootstrap replicate draws as many task ids as the run has, with replacement, and averages the observed per-task estimates of the drawn tasks. The interval runs from the alpha/2 to the 1 - alpha/2 percentile of the replicate values, which is 2.5 and 97.5 by default. The task is the unit because the task set is the sample: the population I care about is every task of this kind, and another draw of tasks would give another score. Each task's estimate already carries the noise of its trials, since a task that passed 5 of 8 times could as easily have passed 4 or 6, and that noise is part of how much the per-task estimates differ from one task to the next. Resampling the tasks therefore covers both the choice of tasks and the trial noise, and redrawing the trials as well would count the trial noise twice. In a comparison each replicate uses one task draw for both runs, which is the same as resampling the per-task differences, so the difference reflects how the two runs did on the same tasks. The point estimate is always the plain mean over the original tasks, never the mean of the bootstrap replicates.

### The two-level variant and why it is not the default

I first built the interval as a two-level bootstrap: draw the tasks, then redraw each drawn task's n trials with replacement and compute the estimator from the new success count. When I measured it, the intervals were off-centre. C(c, k) / C(n, k) is the unbiased estimator of p^k, so once the trials of a task with c successes are redrawn, the replicates of that task centre on (c / n)^k instead of on the point estimate; for n = 8, c = 4, k = 4 the point estimate is 1/70 and the two-level replicate mean is 0.5^4 = 0.0625. pass@k moves down in the same way, and the trial noise is counted twice, once in the observed per-task estimates and again in the redraw. These are the two settings for run A of the example at k = 4:

```
$ passk summarize examples/two_runs.jsonl --run A --k 4 --json out/a4-task.json
run A in examples/two_runs.jsonl: 60 tasks, 8 trials per task, 2000 task-level bootstrap replicates, seed 0
k  pass@k  95% CI            pass^k  95% CI
4  0.9271  [0.8828, 0.9626]  0.2014  [0.1224, 0.2895]
$ passk summarize examples/two_runs.jsonl --run A --k 4 --inner-resample --json out/a4-two-level.json
run A in examples/two_runs.jsonl: 60 tasks, 8 trials per task, 2000 two-level bootstrap replicates, seed 0
k  pass@k  95% CI            pass^k  95% CI
4  0.9271  [0.8238, 0.9412]  0.2014  [0.1612, 0.3445]
```

The replicate means in this table come from the `replicate_mean` fields of the two JSON files those commands wrote:

| run A, k = 4 | point | task-level 95% CI | task-level replicate mean | two-level 95% CI | two-level replicate mean |
|---|---|---|---|---|---|
| pass^4 | 0.2014 | [0.1224, 0.2895] | 0.2009 | [0.1612, 0.3445] | 0.2488 |
| pass@4 | 0.9271 | [0.8828, 0.9626] | 0.9276 | [0.8238, 0.9412] | 0.8875 |

The task-level replicate mean stays next to the point estimate, while the two-level one sits above it for pass^4 and below it for pass@4, and the two-level interval is shifted the same way. I kept the variant behind `--inner-resample` (`inner_resample=True` in the API) so this comparison can be repeated, and made the task-level bootstrap the default.

## What the verdict means

Here diff is B - A in pass^k, delta is `--margin`, and the one-sided lower bound at level 1 - alpha is the alpha percentile of the bootstrap differences.

- `superior`: the one-sided lower bound of diff > 0.
- `non_inferior`: the one-sided lower bound of diff > -delta (and not superior).
- `inferior`: the one-sided UPPER bound of diff (the 1-alpha percentile) < -delta.
- `inconclusive`: otherwise.

A `non_inferior` verdict is a statement at the chosen margin and alpha about this task set: the data rule out B being worse than A by more than delta on these tasks, and they say nothing about tasks outside the set. It is not a p-value, and it is not a substitute for choosing and writing down the margin before the runs. With a margin of 0, `non_inferior` cannot occur, since a lower bound above 0 is already `superior`. The permutation p-value is a separate check (signs of the per-task differences flipped at random, two-sided), and the verdict does not use it.

## A result that stays inconclusive

`examples/small_runs.jsonl` has 15 tasks with 3 trials each, generated the same way with B's probability 0.05 above A's (`python examples/make_example.py --tasks 15 --trials 3 --lift 0.05 --out examples/small_runs.jsonl`).

```
$ passk compare examples/small_runs.jsonl --a A --b B --k 2
A = A, B = B, k = 2: 15 shared tasks, 2000 task-level bootstrap replicates, seed 0
metric  run  point   95% CI
pass^k  A    0.3333  [0.1333, 0.6000]
pass^k  B    0.4000  [0.1778, 0.6222]
pass@k  A    0.7333  [0.6000, 0.8444]
pass@k  B    0.6222  [0.3778, 0.8444]
diff B - A in pass^k: 0.0667, 95% CI [-0.0889, 0.2444]
one-sided 95% bounds: lower -0.0667, upper 0.2222; margin 0.05
paired permutation p (pass^k, 2000 sign flips): 0.6237
verdict: inconclusive
```

B's pass^2 estimate is 0.4000 against 0.3333 for A, yet the interval for the difference runs from -0.0889 to 0.2444 and the verdict is inconclusive. This is the case the tool exists for: a higher number that, on 15 tasks and 3 trials, cannot be told apart from no change or from a loss.

## Limits

- The intervals are plain percentile bootstrap intervals, with no BCa correction for bias or skew. The two-level variant behind `--inner-resample` is biased in the way shown above, and I keep it only for comparison.
- Trials are assumed independent within a task, and tasks independent of each other. Trials that share a cache, a random seed or an environment state break this assumption.
- Every task has the same weight in the mean, whatever its difficulty or its number of trials.
- With few tasks the intervals are wide. That is the honest answer for a small task set, not an error in the method.
- There is no correction for multiple comparisons across k. Comparing at every k from 1 to 8 and reporting the k that looks best raises the chance of a verdict that would not hold up.

## Tests

```
pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

The tests check the estimators against hand-computed values, the bootstrap against a plain task-level bootstrap in the one-trial case, the bias of the two-level variant, the verdicts on constructed runs, the loaders and the command line. The workflow in `.github/workflows/ci.yml` runs ruff and the tests on Python 3.11, 3.12 and 3.13.

## License

MIT; see [LICENSE](LICENSE).
