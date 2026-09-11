# Ball & Beam: closing a generalization gap

A fresh unassisted PPO run with training seed 123 paused at the first checkpoint
passing all ten fixed tests and the canonical replay: episode 611. It settled
49/50 predeclared qualification starts at seeds 5300000–5300049. The remaining
trial timed out 500 steps into the episode, nearly motionless but 13.5 cm from its
target. Fixed-suite success did not guarantee robust target acquisition.

The environment and success requirements are unchanged. The ball must remain
within 4.5 cm, move below 0.06 m/s, and keep beam tilt below 0.035 radians for 50
consecutive control steps (two seconds), before the 500-step deadline.

Schema 2 adds a disclosed actor initialization from 120 reference trajectories
at seeds 930000–930119, using the existing cascaded PD controller as a
training-only teacher. The unchanged jittered start distribution spans ball
positions ±0.65 m and targets ±0.3 m. A fixed permutation prevents fitting batches
in trajectory order. The actor fits for 80 epochs, then ordinary PPO runs with its
existing learning rate. Initial exploration log standard deviation is -2.5.
The learned neural policy controls every evaluation; the teacher is absent at
inference. These are demonstration-assisted PPO results, not unaided learning.

Selection again used ten fixed tests and the canonical replay. A new suite at
seeds 5400000–5400049 was evaluated only after selection:

| Training seed | Selected episode | PPO updates | Training steps | Fixed tests | New qualification starts | Canonical replay |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 62 | 3 | 6347 | 10/10 | 50/50 | settled |
| 123 | 62 | 3 | 6188 | 10/10 | 50/50 | settled |

The frozen guided engine fingerprint is
`2ae8d7d5aa5b81af084863a110c7cea6fe3dd19ac5c09aa9ed97fd9dcd2d8140`.
The unassisted diagnostic fingerprint is
`95cf91cf836a6b58739b45df077d7e49fab555a6411f425e219c34fb0a8d8dfe`.
[Detailed results](results/ballbeam-guided-qualification.json) preserve both
the 49/50 failure and every guided rollout, selected checkpoint metadata, and
hashes. Subsequent repeated uses of these suites are regression verification,
not additional independent evidence. Final-engine requalification is a
separate operation and does not relabel these historical training runs.

The metadata test failed before implementation. Both guided-learning tests
then passed in 14.248 seconds, including independent PPO seeds and all 50 starts
per seed. The matching seeded runs were repeated to preserve exportable
checkpoint pairs. All work used a disposable container without network, ports,
or live checkpoint mounts; no live policy was changed.
