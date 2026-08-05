# Rally Ridge: 500-episode reward-correction study

Date: 2026-08-05

Training seed: `42`

Evaluation suite: `policy-atlas-eval-v1-n10` (fixed seeds `100000`-`100009`)

## Question

Why did Rally Ridge appear to reset early and fail to make progress, and does
correcting its drift reward improve PPO learning over 500 episodes?

## Root cause

The schema-2 environment awarded `0.04 * drift_activation` on every control
step in a corner. Rally starts in a corner, and drift activation reaches one
while the car is stationary. A policy could therefore collect about `+12`
before the 300-step stall boundary, offsetting the `-6` time cost and most of
the `-15` stall penalty. That roughly `-9` return was safer than exploration
that often ended with a `-40` collision.

The schema-3 correction pays drift shaping only for positive forward distance
and measured slip in a controllable 0-35 degree band. The bonus peaks at 15
degrees and is capped at 16% of ordinary forward-progress reward. Stationary
drift, reverse travel, and spins earn no drift bonus. Failed driving episodes
also report normalized peak lap progress so policies are no longer ranked by
shaped return alone.

## Controlled result

Both runs used seed 42, 500 episodes, PPO's recorded protocol, and the same
fixed evaluation suite. Raw returns are not comparable across reward schemas;
task outcomes and interaction counts are.

| Measure | Schema 2 baseline | Schema 3 correction |
|---|---:|---:|
| Engine digest | `cb3a68bc...` | `03846358...` |
| Environment steps | 157,672 | 469,342 |
| PPO updates | 74 | 177 |
| Mean episode length | 315 steps | 939 steps |
| Stalls | 484 (96.8%) | 156 (31.2%) |
| Collisions | 6 (1.2%) | 216 (43.2%) |
| Wrong-way | 10 (2.0%) | 0 |
| Timeouts | 0 | 128 (25.6%) |
| Laps / successes | 0 / 0 | 0 / 0 |

The first 50 corrected episodes still all stalled and averaged 0.3% lap
progress. The last 50 had no stalls, averaged 74.1% progress, and included 16
episodes above 90%. The best stochastic training episode reached 97.6% before
the 60-second boundary.

Fixed deterministic evaluation improved from zero forward progress to 91.1%
at checkpoint 301 and 90.8% at checkpoint 500. All ten final evaluation starts
ended in collision, so this is a large learning improvement, not a solved task.

| Checkpoint | Total steps | PPO updates | Eval progress | Eval success |
|---:|---:|---:|---:|---:|
| 28 | 8,400 | 4 | 0.0% | 0/10 |
| 176 | 63,897 | 29 | 57.0% | 0/10 |
| 201 | 93,163 | 40 | 75.9% | 0/10 |
| 301 | 230,034 | 89 | **91.1%** | 0/10 |
| 350 | 294,472 | 112 | 56.9% | 0/10 |
| 450 | 416,298 | 157 | 89.1% | 0/10 |
| 500 | 469,342 | 177 | 90.8% | 0/10 |

After implementation and review, the complete corrected experiment was rerun
uninterrupted against the final engine digest. It exactly reproduced the prior
run's interaction count, PPO update count, training returns, and fixed-evaluation
checkpoints at episodes 176, 301, and 500. The earlier corrected lineage remains
in a recoverable archive; the figures above describe the final active run.

## What remains

The final deterministic policy reaches the 89-91% corner complex too fast,
brakes late, and crosses the boundary. Extending the horizon alone does not
help: replaying checkpoints 301 and 500 with a 90-second horizon still produced
10/10 collisions at the same progress.

The next predeclared ablation should add one physically grounded, forward-
distance-gated preview safety cost based on upcoming curvature and available
lateral grip. It should be tested on a new checkpoint schema and branch, with
additional training seeds and a validation suite separate from the fixed final
test suite. Learning-rate, entropy, and KL changes should not be bundled into
that experiment.

The schema-2 baseline was archived before the protocol change. It remains
preserved in the Docker checkpoint volume but is intentionally incompatible
with schema-3 policy loading and ranking.
