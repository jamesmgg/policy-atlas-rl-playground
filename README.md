# Policy Atlas — RL Playground

Policy Atlas is an interactive reinforcement-learning laboratory. It makes the
agent's observation → action → reward loop visible while PPO trains, and keeps
training evidence, fixed-suite evaluation results, saved policies, and ghost
replays in one workspace.

The public Cloudflare showcase is available at
[policy-atlas-rl-playground.pages.dev](https://policy-atlas-rl-playground.pages.dev).
It bundles 46 qualified policy replays across all 23 environments. Live PPO
training and checkpoint mutation remain available in the local Docker app.

On phones and tablets, use the bottom navigation for **Projects**, **Watch**,
**Train**, and **Results**. Tap the current project at the top to switch at any
time. Projects are searchable and grouped by category; starting training or
choosing a saved replay opens Watch automatically. Desktop keeps the full
workspace with the project library and training controls alongside the simulator.

The playground currently contains 23 experiments:

- driving tasks, including wet grip, traffic, endurance, efficiency, and drift;
- control benchmarks: a continuous-force CartPole variant and Continuous Mountain Car;
- aerospace tasks: Lunar Lander and a five-waypoint Drone Course;
- classic control: Pendulum Swing-up;
- new control labs: Orbital Docking, Robot Arm Reach, Robot Target Tracking,
  and Ball & Beam;
- arcade games: Paddle Rally, Flappy Flight, and Coin Collector.

The September 11 release includes two verified neural policies per game. All
46 passed 50/50 qualification starts, all ten fixed starts and their canonical
replay on the final engine. These finite tests establish demonstrated success,
not a guarantee for every possible start or future training seed.

The default **Pause when all test starts pass** option saves and pauses after a
policy passes all ten fixed test starts and its canonical replay. It preserves
a successful policy before later PPO updates can degrade it; it does not
guarantee success on every possible start. **Watch verified policy** plays an
archived neural policy without replacing the policy currently being trained.
All newly recorded replays preserve the whole scene, including driving traffic,
balls, pipes, coins and final outcomes, and hold their last frame. Legacy
archives may contain only the original trajectory.
Telemetry sits below the canvas so it cannot cover the landing pad or game
objects. The lander's feet match its physical contact point, and a terminal
frame no longer displays exhaust from its final applied action.

Each experiment describes its objective, success condition, observation space,
action space, metric direction, difficulty, and horizon before a run starts.

Experiments with analytic reference controllers offer **Watch reference** to
demonstrate a feasible solution; it does not play a learned policy.
**Compare controllers** evaluates a frozen policy, zero action, random actions,
and the reference on the same ten starts, with 95% Wilson success intervals.
These repeatable diagnostics are not untouched holdouts. Model assumptions and
reference methods appear under **Method details**.

See the [all-games audit](docs/2026-09-11-all-games-audit.md) for the 23-game
training evidence, explicit demonstration assistance, preserved failed runs,
and final qualification status. All tasks except Cart-Pole, Pendulum, and Robot
Target Tracking now use demonstration-assisted initialization before PPO
practice; the expert is absent during evaluated gameplay. Apex GP, Drift Trial,
and Robot Arm Reach use smaller PPO learning rates to retain learned precision.
Robot Reach also uses corrective demonstrations near its targets; the
[correction study](docs/2026-09-11-robot-reach-corrections.md) records why reducing
the learning rate alone was insufficient.

See the [September audit and learning study](docs/2026-09-10-chief-scientist-audit.md)
for the fixes, measurements, task contracts, and limitations.

## Run locally

```bash
docker compose up --build
# Open http://localhost:8900
```

Training defaults to CPU. To expose a compatible NVIDIA GPU to PyTorch:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Small-network CPU training defaults to one PyTorch thread, configurable with
`TORCH_NUM_THREADS`. The web port binds to loopback. For access from your own
Tailscale devices, run `tailscale serve --bg --tcp=8900 tcp://127.0.0.1:8900`
on the host, then open `http://<host-tailscale-ip>:8900`.

Saved policies persist in the `rl-checkpoints` Docker volume. **New seeded run**
archives the active run before clearing the workspace, so reset data remains
recoverable from the checkpoint archive.

## Build the public replay showcase

Generate `frontend/public/demo.json` from a qualified release package with the
backend image, then build the replay-only frontend:

```bash
docker run --rm -e PYTHONPATH=/app -v "$PWD:/workspace" rl-simulator-backend:local \
  python /workspace/scripts/export_public_demo.py \
  /workspace/data/release-validation-20260911/package \
  /workspace/frontend/public/demo.json
npm --prefix frontend run build:public
```

The public build contains recorded scenes and terminal frames. It makes no API
or WebSocket calls and exposes no training, reset, or checkpoint endpoints.

## Scientific safeguards

- Generalized advantage estimation cuts traces at the correct transition.
  Intrinsic environment deadlines are reported with `truncated` and
  `task_deadline` flags, but remain task endings for return computation; only
  administrative/external rollout cutoffs bootstrap the critic.
- Finite-horizon observations expose remaining time. Driving policies also see
  track phase, cumulative objective state, wrong-way margin, recent progress,
  and one-time traffic-pass state used by rewards or termination.
- PPO uses a tanh-squashed Gaussian and applies the matching log-probability
  correction, so sampled actions and optimized likelihoods agree.
- The trainer multiplies each immediate environment reward by the global
  constant `0.01` before it enters PPO (and before any external-cutoff
  bootstrap). Unscaled rewards remain unchanged for the UI, leaderboards, and
  evaluation reports. Constant entropy pressure is disabled, allowing
  continuous controls to become precise.
- Critic explained variance, value bias, value clipping, and action spread are
  emitted live and stored with each checkpoint.
- Partial rollouts are learned from instead of silently discarded.
- Training, environment, and evaluation random-number state is seeded and saved
  with each policy for reproducible continuation.
- Saved protocols include an SHA-256 fingerprint of the experiment engine and
  explicitly avoid promising bitwise identity across devices or dependency builds.
- Checkpoint sidecars are self-hashed and cross-checked against metadata inside
  the tensor payload; corrupt archived branches are rejected before a restore.
- By default, checkpoints are evaluated on ten fixed, versioned test starts so
  separately trained policies can be compared on the same environment starts.
  The historical 16-task campaign reports use one optimizer/training seed (`42`); ten
  evaluation starts or 100 holdout starts measure environment-start robustness,
  not optimizer-seed replication. Reports include mean, standard deviation,
  success rate, a 95% Wilson interval, evaluation count, seed, and update count.
  The September 2026 control study additionally tests independent training seeds
  `42` and `123`, with separate 50-start evaluations for each selected policy.
- Driving training uses a disclosed 75% canonical / 25% rolling-checkpoint
  mixture. Rolling states reconstruct speed, clock, task progress, fuel, and
  traffic from a curvature/grip braking envelope; selection and replay keep
  their unchanged fixed start-line distributions.
- Traffic Rush assigns the same -40 terminal cost to collision, contact,
  wrong-way, stall, and timeout failures, plus a disclosed remaining-clock cost.
  Its course-progress auxiliary retains the physical terminal potential: the
  earlier terminal-zero form canceled failed-path progress and produced a large
  late negative correction under finite-lambda GAE. A fixed episode schedule
  first mixes physical checkpoint-11 near-pass rehearsals at 12 m/s with nested
  18/24/30 m/s bot-speed practice, then retires the easy rehearsal. The reverse
  curriculum continues through checkpoints 11, 9, 3, and the canonical start;
  reconstructed pass masks are inherited state, never newly learned overtakes.
- Traffic Rush is also explicitly demonstration-assisted PPO. Thirty successful
  canonical pure-pursuit trajectories initialize the actor, followed by two
  fixed DAgger rounds of 40 learned-policy rollouts whose visited states are
  labeled by the training-only expert. The resulting 89,422-sample dataset,
  weighted throttle/steering/binary losses, seed ranges, and SHA-256 digest are
  checkpointed. The expert is absent at inference. Three disclosed guidance
  observations--target lane, pursuit heading error, and target speed--expand
  Traffic to 41 observations, while latent action log standard deviation -2.0
  preserves bounded exploration during PPO fine-tuning.
- Wet Apex assigns the same nominal -40 terminal cost to collision, wrong-way,
  and stall failures. This aligns the raw terminal costs, but equal costs alone
  do not prove that delayed inactivity is neutral under discounting; the
  verified result is empirical evidence for the complete training protocol.
- Lunar Lander uses a fixed-suite, performance-gated reverse-altitude
  curriculum: touchdown rehearsals first, then overlapping 30-100, 50-200,
  and 100-500 unit approach bands before the canonical descent. Lander
  optimizes undiscounted finite-horizon return (`gamma = 1.0`),
  so a terminal failure has the same cost whether it happens immediately or
  near the deadline. Lander's
  touchdown frontier advances after one >=90% deterministic 20-start
  evaluation; harder Lander frontiers use >=75%. Once a Lander frontier has
  been mastered, half of resets
  stay on the active frontier and the other half retain mastered easier work;
  curriculum evaluation is therefore a
  training-control signal, not a passive diagnostic. Its state is checkpointed
  exactly, while its results remain excluded from full-course checkpoint
  selection.
- Drone Course uses a fixed, episode-number schedule that never reads
  evaluation outcomes. Episodes 1--200 use 100% short approaches; 201--450 use
  30% approaches and 70% half-segments; 451--800 use 10% approaches, 20%
  half-segments, and 70% hard handoffs; episode 801 onward uses 5% approaches,
  5% half-segments, 30% hard handoffs, and 60% unchanged canonical courses.
  The segment is sampled uniformly across all five directions. Approaches begin
  120 units before a target at rest, half-segments begin halfway from the prior
  waypoint, and hard handoffs reproduce inbound 60--100 unit/s momentum. A
  training-only segment ends after its one capture and awards only the ordinary
  +20 capture reward; it is never labeled a canonical solve.
- Drone Course is explicitly **not** a pure model-free-from-scratch experiment.
  Before PPO, its actor receives a deterministic behavior-cloning warm start
  from 80 successful physics-controller trajectories at seeds 600,000--600,079
  (66,978 state/action pairs in the current dataset). Adam fits the shared torso
  and bounded-action mean head for 60 unshuffled passes; the expert is absent at
  inference and PPO subsequently fine-tunes the policy. The demonstrations are
  disjoint from checkpoint-selection seeds (100,000+) and default holdout seeds
  (200,000+), and their SHA-256 digest, optimizer contract, sample count, and
  final loss are stored with checkpoints. Drone exploration begins at latent
  log standard deviation -2.0 per rotor.
- The Drone observation exposes 12 Markov control features: target-relative
  position, velocity, desired-velocity error, desired-tilt error, sine/cosine
  tilt, angular rate, waypoint progress, and remaining time. Velocity terms use
  a 100 unit/s scale so hard handoffs stay near the network's training range.
  Canonical evaluation, horizons, and holdout starts remain unchanged.
- Drone Course uses an undiscounted finite-horizon objective
  (`gamma = 1.0`), and both crash and timeout apply the same -50 task penalty.
  Its segment-local training
  auxiliary is the change in a cost potential combining waypoint distance
  (0.05), error from a 20--80 unit/s braking-envelope velocity target (0.10),
  and error from the corresponding one-second desired tilt (5.0). A waypoint
  capture resets the baseline for the next target without charging the target
  switch; a crash or timeout retains its physical terminal potential instead
  of forcing it to zero. This auxiliary deliberately changes failed-attempt
  ordering by terminal segment cost and is not policy invariant. It exists for
  training credit assignment: a lower-cost failure ranks above a high-cost
  crash, and finite-lambda GAE receives no artificial positive terminal
  correction. Attitude and thrust regularizers are accumulated and charged
  only when the course is completed, ranking successful controllers by
  efficiency. Traffic Rush, Lunar Lander, and Orbital Docking also use
  `gamma = 1.0`; other scenarios use `gamma = 0.995`. Each scenario declares its exploration settings.
  The next scheduled episode, selected training mode/segment, and RNG state are
  checkpointed exactly. Training starts and demonstrations never enter
  full-course checkpoint selection.
- The benchmark campaign freezes checkpoint selection before an optional
  100-start holdout at a disjoint seed range. Within one campaign, those starts
  cannot affect early stopping or checkpoint choice. The same holdout range has
  been reused while iterating across branches, however, so these results are
  validation evidence rather than a never-seen project-wide final test.
- The default campaign selects and stops at the first statistically qualifying
  fixed-suite checkpoint; the post-selection 100-start holdout is the
  within-campaign confirmation layer. `--confirmations` remains available for
  stricter selection studies.
- A fixed canonical rollout provides comparable ghost playback and a completion
  check for automatic pause and policy qualification. Statistical success
  estimates come from the separate evaluation starts.
- The UI ranks only checkpoints from the same versioned evaluation suite and
  engine source; older protocols remain inspectable without being mixed in.

These environments remain educational approximations. The driving, lander, and
drone models are intentionally lightweight rather than validated engineering
simulators. Results should not be interpreted as real-world vehicle or flight
performance. Continuous Cart-Pole uses bounded force and semi-implicit Euler,
so it is intentionally not score-compatible with Gymnasium CartPole-v1;
Mountain Car preserves Continuous Mountain Car's step dynamics but normalizes
the observation exposed to PPO.

## Interface

The dark experiment library scales through family filters and search. Every
experiment publishes its exact observation dimensions, reward terms, and
termination conditions. The active workspace leads with the simulator, then a
Top Runs section. Evaluated runs with at least one success can earn
gold/silver/bronze medals; zero-success runs use neutral Best Attempt ranks
instead. Technical evidence is collapsed by default.
It also combines:

- an experiment brief and reproducibility settings;
- a responsive simulator with fullscreen mode;
- a live learning lens showing observations, action, immediate reward, and PPO
  diagnostics;
- separately scaled performance and optimization charts;
- saved-policy comparison, evaluation uncertainty, ghost replay controls, and
  recoverable archived branches;
- quick, study, and extended run presets with an advanced configuration panel.

Keyboard focus, reduced-motion preferences, narrow screens, and color contrast
are supported without hiding the underlying scientific context.

## Architecture

| Area | Location | Responsibility |
|---|---|---|
| Scenario contracts | `backend/app/scenarios/` | Catalog metadata and environment factories |
| Environments | `backend/app/envs/` | Dynamics, rewards, observations, success, and rendering payloads |
| PPO | `backend/app/ppo/` | Actor-critic network, squashed policy, GAE buffer, and updates |
| Training | `backend/app/trainer.py` | Seeded runs, evaluation, checkpoints, and WebSocket events |
| API | `backend/app/main.py` | FastAPI routes, validation, and live stream |
| Interface | `frontend/src/features/simulator/` | Library, simulator, learning lens, evidence, and policies |
| Contracts | `frontend/src/api/types.ts` | Frontend message types and pure UI helpers |

## Verification

The reproducible test path uses the same container runtime as production:

```bash
# Backend focused scientific tests
docker run --rm -v "$PWD/backend:/app" -w /app \
  rl-simulator-backend:local python -m unittest discover -s tests -v

# Every scenario can reset, step, summarize, and render
docker run --rm -v "$PWD/backend:/app" -w /app \
  rl-simulator-backend:local python smoke_test.py

# Frontend types, behavior contracts, and production bundle
npm --prefix frontend test
npm --prefix frontend run build
```

Fresh campaigns can be capped at 2,000 episodes and stopped after the first
statistically qualifying checkpoint, before its frozen policy is tested on the
within-campaign post-selection holdout. See
[`docs/benchmark-campaigns.md`](docs/benchmark-campaigns.md) for the exact solve
contract, read-only checkpoint-volume holdout, and Docker commands.

With the stack running, `npm --prefix frontend run test:live` verifies the
catalog → scenario switch → live frame → PPO update → fixed-suite evaluation →
seeded reset → archived-branch restore flow.

## API overview

- `GET /api/health`, `/api/scenarios`, `/api/scene`, `/api/checkpoints`,
  `/api/runs`, `/api/training/status`
- `GET /api/reference/{scenario_id}` for a labelled analytic-controller replay
- `POST /api/evaluation` for paired frozen-policy and baseline diagnostics
- `POST /api/scenario` with `{"id": "cartpole-balance"}`
- `POST /api/training/start`, `/api/training/stop`, `/api/training/reset`,
  `/api/runs/{archive_id}/restore`
- `WS /ws/training` for frames, episode results, PPO metrics, checkpoint events,
  scenario changes, replay selection, and run controls

## Good next experiments

Contextual bandits and tabular Gridworld are the highest-value additions because
they make exploration, value estimates, and credit assignment directly visible.
MountainCar and CartPole establish continuous-control baselines today; a future
partially observable maze would then demonstrate recurrence and belief state.

For research-grade use, the next engineering priorities are an append-only run
log instead of embedding cumulative history in every checkpoint, traction-circle
and tire-load dynamics for driving, and strict library-backed benchmark variants
with confidence intervals over more independent training seeds.
