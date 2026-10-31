# Examples

`two_runs.jsonl` holds synthetic results for two runs, A and B, on 60 tasks with 8 trials each (960 records). `make_example.py` wrote it with seed 7. For each task it draws A's success probability from Beta(2, 2), sets B's to A's plus 0.03 (clipped to [0, 1]), and draws each trial's outcome from its run's probability, so B is slightly better by construction.

`small_runs.jsonl` uses the same construction with 15 tasks, 3 trials per task and a lift of 0.05. The main README uses it for the inconclusive example.

`make_example.py` uses only the standard library, and the same arguments give the same file byte for byte (the test suite checks both files against fresh output). From the repository root:

```
$ python examples/make_example.py
wrote 960 records (60 tasks x 8 trials x 2 runs) to two_runs.jsonl
$ python examples/make_example.py --tasks 15 --trials 3 --lift 0.05 --out examples/small_runs.jsonl
wrote 90 records (15 tasks x 3 trials x 2 runs) to small_runs.jsonl
```

## Output

Every block below was produced by running the command shown, from the repository root, with the default 2000 bootstrap replicates, alpha 0.05 and seed 0.

Run A:

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

Run B:

```
$ passk summarize examples/two_runs.jsonl --run B
run B in examples/two_runs.jsonl: 60 tasks, 8 trials per task, 2000 task-level bootstrap replicates, seed 0
k  pass@k  95% CI            pass^k  95% CI
1  0.5542  [0.4833, 0.6312]  0.5542  [0.4833, 0.6312]
2  0.7387  [0.6679, 0.8089]  0.3696  [0.2875, 0.4619]
3  0.8315  [0.7661, 0.8929]  0.2780  [0.1949, 0.3714]
4  0.8850  [0.8240, 0.9381]  0.2257  [0.1443, 0.3200]
5  0.9187  [0.8598, 0.9649]  0.1932  [0.1137, 0.2881]
6  0.9411  [0.8845, 0.9827]  0.1720  [0.0940, 0.2673]
7  0.9563  [0.9021, 0.9938]  0.1583  [0.0792, 0.2562]
8  0.9667  [0.9167, 1.0000]  0.1500  [0.0667, 0.2500]
```

Run A with the two-level variant, which the main README explains and does not recommend. The points are the same, and for k of 2 and above the intervals are centred higher for pass^k and lower for pass@k than in the default run of A above:

```
$ passk summarize examples/two_runs.jsonl --run A --inner-resample
run A in examples/two_runs.jsonl: 60 tasks, 8 trials per task, 2000 two-level bootstrap replicates, seed 0
k  pass@k  95% CI            pass^k  95% CI
1  0.5750  [0.5000, 0.6458]  0.5750  [0.5000, 0.6458]
2  0.7804  [0.6810, 0.8185]  0.3696  [0.3083, 0.4816]
3  0.8777  [0.7714, 0.8991]  0.2616  [0.2146, 0.3961]
4  0.9271  [0.8238, 0.9412]  0.2014  [0.1612, 0.3445]
5  0.9545  [0.8560, 0.9652]  0.1670  [0.1268, 0.3140]
6  0.9702  [0.8791, 0.9815]  0.1476  [0.1048, 0.2923]
7  0.9792  [0.8917, 0.9938]  0.1375  [0.0917, 0.2771]
8  0.9833  [0.9000, 1.0000]  0.1333  [0.0833, 0.2667]
```

B against A at k = 4, the comparison shown in the main README:

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

The small example at k = 2, as in the main README:

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

I first ran the small example at k = 1. In this draw B's estimate is below A's at k = 1 (0.5111 against 0.5333), so it does not show the case the main README describes, and I used k = 2 there instead:

```
$ passk compare examples/small_runs.jsonl --a A --b B --k 1
A = A, B = B, k = 1: 15 shared tasks, 2000 task-level bootstrap replicates, seed 0
metric  run  point   95% CI
pass^k  A    0.5333  [0.3778, 0.6894]
pass^k  B    0.5111  [0.2889, 0.7333]
pass@k  A    0.5333  [0.3778, 0.6894]
pass@k  B    0.5111  [0.2889, 0.7333]
diff B - A in pass^k: -0.0222, 95% CI [-0.1556, 0.1333]
one-sided 95% bounds: lower -0.1333, upper 0.1111; margin 0.05
paired permutation p (pass^k, 2000 sign flips): 0.9620
verdict: inconclusive
```
