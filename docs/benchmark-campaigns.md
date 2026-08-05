# Benchmark campaigns

`backend/benchmark_all.py` coordinates reproducible, sequential training through
the public Policy Atlas API. With no flags that authorize training, it only
inventories the catalog. `--execute` starts fresh seeded runs; the backend
archives each scenario's existing active branch before reset.

## Solve contract

A checkpoint qualifies when all of these are true:

- it uses the run's exact engine digest, evaluation suite, and training seed;
- it evaluates at least 10 fixed starts;
- its success rate is at least 90%;
- the lower end of its 95% Wilson success interval is at least 70%.

The default early-stop mode requires two consecutive qualifying checkpoints.
With the default ten-start suite, this effectively requires 10/10 successes at
both checkpoints. The reported earliest solve is the first checkpoint in that
confirmed pair; the second checkpoint is recorded as the confirmation point.
`--full-budget` still reports that earliest pair but trains through episode
2,000.

This is the checkpoint-selection contract, not the final generalization claim.
Repeated selection uses the same fixed starts, so the selected policy is then
evaluated once on a distinct deterministic holdout. Holdout outcomes never
affect selection or early stopping.

## Post-selection holdout

The default holdout contains 100 starts at seeds 200,000–200,099, disjoint from
the selection suite beginning at seed 100,000. The harness validates both the
checkpoint and its engine digest, loads the policy from a read-only checkpoint
volume, and records every trial plus aggregate reward, metric, failure progress,
success rate, and 95% Wilson interval. A confirmed claim should still reproduce
over independent training seeds; `--seeds 42,43,44` supports that campaign.

Selection is frozen before the holdout runs:

- a solved campaign selects its earliest confirmed-solve checkpoint;
- an unsolved campaign selects its final budget checkpoint;
- holdout results are stored in a separate `holdout` object with
  `protocol_role: post_selection_holdout`.

`--holdout-episodes 0` explicitly selects a selection-only campaign. Execute
mode otherwise requires `--checkpoint-root`; this prevents a nominally finished
campaign from silently recording the required holdout as `not_run`. Inventory
mode remains read-only and does not require access to checkpoint weights.

A campaign exits successfully only when every requested scenario/seed has a
confirmed solve and, unless disabled explicitly, its frozen checkpoint passes
the disjoint holdout thresholds. The report state is `verified` only in that
case; unsolved, interrupted, missing, or failed-holdout runs produce
`incomplete` and a non-zero exit code.

## Docker commands

The host does not need a Python installation. From the repository root, use the
existing backend image and mount the backend source:

```powershell
# Read-only catalog and checkpoint coverage
docker run --rm `
  -v "${PWD}\backend:/app" -w /app `
  -v "${PWD}\docs\results:/results" `
  rl-simulator-backend:local python benchmark_all.py `
  --base-url http://host.docker.internal:8900 `
  --output /results/inventory.json

# Selected quick campaign, stopping after a confirmed solve
docker run --rm `
  -v "${PWD}\backend:/app" -w /app `
  -v rl-simulator_rl-checkpoints:/checkpoints:ro `
  -v "${PWD}\docs\results:/results" `
  rl-simulator-backend:local python benchmark_all.py `
  --base-url http://host.docker.internal:8900 --execute `
  --scenarios mountain-car,cartpole-balance --max-episodes 2000 `
  --checkpoint-root /checkpoints `
  --output /results/foundations.json

# Every experiment for the full budget, across three training seeds
docker run --rm `
  -v "${PWD}\backend:/app" -w /app `
  -v rl-simulator_rl-checkpoints:/checkpoints:ro `
  -v "${PWD}\docs\results:/results" `
  rl-simulator-backend:local python benchmark_all.py `
  --base-url http://host.docker.internal:8900 --execute --full-budget `
  --scenarios all --seeds 42,43,44 --max-episodes 2000 `
  --checkpoint-root /checkpoints --holdout-episodes 100 `
  --output /results/all-full.json
```

Only one backend trainer exists. Never run two campaign processes against it,
and do not start a campaign while the UI or another agent is training. The
harness refuses to begin if training is active. It writes its report atomically
after every poll, including seed, episodes, environment steps, PPO updates,
success rate and interval, metric and dispersion, failure progress, evaluation
suite, engine digest, separately labelled selection and holdout evidence, and
per-scenario terminal state.

Keep the compact JSON reports under `docs/results/` in Git so a solve claim is
reviewable with the exact engine digest, seeds, checkpoint hashes, selection
episode, and holdout outcome. Large tensor checkpoints stay in the Docker
volume and are intentionally not committed.
