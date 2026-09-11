# Lunar Lander: full-descent learning evidence

The earlier seed-123 episode-1000 policy solved near-pad practice but failed
all 20 full-height diagnostics. The failure was real: its saved canonical
replay crashed. Success in a reverse-altitude curriculum did not establish
that the neural policy had learned the full task.

Lander schema 12 adds a disclosed demonstration-assisted actor warm start.
There are 160 fixed-seed training demonstrations: 80 ordinary full-height
starts and 20 at each of four curriculum altitudes. A bounded physics
controller tracks a 60-unit/s descent cruise, a 10-unit/s² braking envelope,
and a 5-unit/s touchdown target. Its feed-forward deceleration compensates
for tracking lag. Lateral and attitude feedback keep the ship over the pad.
The controller labels training data only; inference uses the learned neural
actor. PPO then updates that actor normally. Initialization and demonstration
seeds, sample ordering, fitting parameters, and this distinction are exposed
in the scenario protocol.

The existing environment physics, fuel budget, full-height evaluation starts,
24-second horizon, reward, and strict touchdown limits are unchanged:
|vx| < 8, |vy| < 14, |tilt| < 0.25, with contact inside the pad. The earlier
18-unit/s shaping target is retained; the faster demonstration descent is
feasible within both the horizon and fuel budget. A separate reference test
landed 100/100 starts at each of the five altitude frontiers.

## Isolated tuning study

The study ran with one CPU thread, no network, no exposed port, no live
checkpoint mounts, and this frozen engine fingerprint:

`67d723932db8e5705521b87999a06337134400f49fb747e9893baeef28c73506`

The neural actor immediately after cloning landed 50/50 previously unseen
full-height starts. PPO used independent training seeds 42 and 123. Selection
chose the first checkpoint passing all ten ordinary fixed test starts;
selection did not inspect the 50 heldout starts at seeds 1950000–1950049.

| Training seed | Selected episode | PPO updates | Training steps | Fixed tests | Full-height holdout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 57 | 1 | 2059 | 10/10 | 50/50 |
| 123 | 61 | 1 | 2068 | 10/10 | 50/50 |

Both selected canonical replays also land safely. Each episode-100 policy
landed 50/50 on a separate diagnostic seed range, 950000–950049. Checkpoint
checksums and individual heldout rollout outcomes, touchdown velocities,
attitudes, remaining fuel, and canonical replay summaries are stored in
[the detailed result](results/lander-v12-demonstration-ppo-holdouts.json).

This is tuning evidence for the fingerprint above, not a claim about a later
combined engine build. The 50-start suite is now also a regression test, so
future repetitions are verification, not additional independent holdouts.

## Continued learning can still regress

At episode 300, seed 42 retained 50/50 diagnostic successes while seed 123
fell to 0/50, despite earlier successful checkpoints. A controlled lower
exploration variation (log standard deviation -2.2 instead of -1.2) did not
fix forgetting: at episode 1000 it reached 38/50 and 0/50 respectively. That
variation was rejected; the production initialization remains -1.2.

These results support pausing on a successfully evaluated saved policy and
preserving the checkpoint. They do not support promising monotone learning
or success on every possible state. A longer-budget user may explicitly
continue experimentation from a saved policy; the successful policy should
remain recoverable and distinct from the latest weights.

## Verification

The new learned-policy test first failed because Lander had no warm start.
With the implementation, all 26 Lander tests passed, including unchanged
physics/reward/curriculum checks, 50-start neural actor checks, and first
successful PPO checkpoint generalization for both seeds. The test run took
39.6 seconds in the isolated container. No live service was restarted or
trained by this study.
