# Results

One sub-folder per finished run, copied from `runs/`:

```
results/
  pilot_seed0/
    results.json   # all metrics, shift gap with CIs, flip rates
    pilot.png      # risk-coverage curves and flip-rate chart
    classes.json   # shared label set used in the run
    logits.npz     # raw FP32 and INT8 logits (lets anyone recompute every metric)
    command.txt    # the exact command used
```

Model weights (`*.pt`) are excluded by `.gitignore`; attach them to a GitHub Release.
