# Lunar Lander: practice success versus replay failure

The active episode-1000 policy (training seed 123) had 36 saved checkpoints.
None had passed any of the ten fixed full-descent evaluation starts. Episode
1000 ranked first by the failure-distance metric, not by successful landing.

An isolated, checksum-verified copy of that checkpoint was evaluated without
training. The existing twenty-start curriculum suites produced:

| Starting condition | Successes |
| --- | --- |
| Touchdown practice, 5–18 units above the pad | 20/20 |
| Low approach, 30–100 units | 20/20 |
| Middle approach, 50–200 units | 8/20 |
| High approach, 100–500 units | 2/20 |
| Full descent from the standard starting height | 0/20 |

These are curriculum diagnostics, not new independent holdouts. The policy's
last twenty training episodes included thirteen successes; easy-start practice
therefore looked substantially stronger than full-descent evaluation.

The canonical fixed-start rollout crashed after 146 steps (5.84 simulated
seconds). Its regenerated trajectory exactly matched the trajectory stored in
the checkpoint. Loading the wrong policy was not the cause of this discrepancy.

## Presentation defects

The trainer stops each episode on its terminal transition and then resets to a
new starting state. Training samples stochastic actions and runs faster than
real time. It does not continue stepping the landed vehicle after termination.

The UI nevertheless made the sequence confusing:

- `Resume` loaded a checkpoint for further training; it was not a demonstration.
- Saved replay used modulo indexing and immediately looped at the end.
- Replay was drawn over the last training frame. A retained `Safe landing`
  notice and training telemetry could label a different, failing saved rollout.

## Corrections and verification

Saved replay now plays once, holds its final frame, and offers Restart and Close.
When training is paused, it has its own scene and telemetry, without a stale
training result. The checkpoint action is named `Load for training`, with an
explicit explanation that it leaves training paused. Lunar Lander explains that
near-pad practice and full-descent results measure different starting conditions.

A regression test checks playback progression, the held endpoint, explicit
restart, and an empty recording. All thirty frontend tests and the production
build pass. In the disposable lab, continuing the copied checkpoint by one
episode produced `Safe landing`; opening the failing episode-1000 replay then
correctly removed that outcome and live telemetry. Endpoint hold, restart,
close, and phone layout were verified in the browser.

This fixes misleading playback and labels. It does not improve the trained
policy's full-descent capability. Live training and checkpoints remain intact.
