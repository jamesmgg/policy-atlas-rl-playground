# Robot Reach: learning to finish the correction

The earlier seed-123 episode-56 policy passed all ten fixed tests and its
canonical replay, but reached only 46/50 targets in the current-engine
qualification suite at seeds 5200000–5200049. The four failures timed out
with residual errors of 0.1175, 0.1283, 0.0884, and 0.1613 m. The task requires
error below 0.08 m and endpoint speed below 0.1 m/s for one continuous second.

Reducing PPO's learning rate from 3e-4 to 3e-5 alone did not close the gap.
Both seeds reached 49/50 targets in the next suite, 5500000–5500049. The
cloned actor already failed that same target before any PPO update. Near the
missed goal, its commands approached zero while the teacher still requested
substantial corrective action. The problem included an approximation gap
in the demonstrations, rather than solely PPO forgetting.

The corrected Robot Reach dataset retains its 80 full-task demonstrations
and two DAgger rounds. Each target also contributes 48 seeded supervised
states near its inverse-kinematics solution: joint offsets within ±0.2 rad,
angular velocities within ±0.4 rad/s, and valid dwell counters and clocks
throughout the episode. These samples teach the actor to continue correcting
near a target even late in an episode. A fixed sample permutation mixes
transit and corrective examples during fitting. The teacher labels data
only; evaluated actions still come entirely from the neural policy.

Schema 3 retains the smaller 3e-5 PPO learning rate. Observation dimensions,
physics, reset distribution, success tolerances, and 300-step/15-second
horizon are unchanged. Robot Tracking and Orbital Docking are unaffected.

Selection used the first checkpoint passing ten fixed starts and the
canonical replay. A new suite at seeds 5600000–5600049 was evaluated only
after selection:

| Training seed | Selected episode | Fixed tests | New targets | Canonical terminal error |
| --- | ---: | ---: | ---: | ---: |
| 42 | 70 | 10/10 | 50/50 | 0.0066 m |
| 123 | 69 | 10/10 | 50/50 | 0.0148 m |

Both policies also pass all 50 starts in each previously observed 5200000
and 5500000 suite. Those checks are regressions, not fresh holdouts. The
5600000 suite is now also a regression test; repetitions do not add
independent evidence. Final-engine qualification remains distinct from
historical training evidence.

[Detailed results](results/robot-reach-corrective-qualification.json) preserve
the original failure, the unsuccessful learning-rate-only diagnosis, new
checkpoint/source fingerprints, every final qualification rollout, and the
regression outcomes. The learning-rate test and near-goal-coverage test failed
before their respective changes; all five relevant tests then passed in
33.908 seconds. Matching seeded runs were repeated to preserve the selected
checkpoints. Studies used a disposable isolated container, with no live
policy, port, or volume changes.
