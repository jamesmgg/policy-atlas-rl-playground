# Policy Atlas — RL Playground

Policy Atlas is an interactive reinforcement-learning laboratory. It makes the
agent's observation → action → reward loop visible while PPO trains, and keeps
training evidence, fixed-suite evaluation results, saved policies, and ghost
replays in one workspace.

The playground currently contains 16 experiments across four families:

- driving tasks, including wet grip, traffic, endurance, efficiency, and drift;
- control benchmarks: a continuous-force CartPole variant and Continuous Mountain Car;
- aerospace tasks: Lunar Lander and a five-waypoint Drone Course;
- classic control: Pendulum Swing-up.

Each experiment describes its objective, success condition, observation space,
action space, metric direction, difficulty, and horizon before a run starts.

## Run locally

```bash
docker compose up --build
# Open http://localhost:8900
```

Training defaults to CPU. To expose a compatible NVIDIA GPU to PyTorch:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Saved policies persist in the `rl-checkpoints` Docker volume. **New seeded run**
archives the active run before clearing the workspace, so reset data remains
recoverable from the checkpoint archive.

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
  The current campaign reports use one optimizer/training seed (`42`); ten
  evaluation starts or 100 holdout starts measure environment-start robustness,
  not optimizer-seed replication. Reports include mean, standard deviation,
  success rate, a 95% Wilson interval, evaluation count, seed, and update count.
- Driving training uses a disclosed 75% canonical / 25% rolling-checkpoint
  mixture. Rolling states reconstruct speed, clock, task progress, fuel, and
  traffic from a curvature/grip braking envelope; selection and replay keep
  their unchanged fixed start-line distributions.
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
- Drone Course uses a fixed-suite, performance-gated reverse curriculum that
  learns the final waypoint first, then unlocks each earlier course segment
  after one >=90% segment evaluation. Before the first unlock, every reset
  targets the active frontier; afterward, half target the active frontier while
  half uniformly rehearse mastered later segments to prevent forgetting.
  While the final segment is locked, a nested momentum control progresses from
  signed -20 to 20 units/s starts, through inbound 20 to 60, to the unchanged
  hard inbound 60 to 100 range. Each band advances after one >=80% deterministic
  10-start control suite; 75% of resets use the active band and 25% uniformly
  retain mastered easier bands. The unchanged hard v3 segment gate runs only
  after all three bands are proficient. Later outer frontiers use hard inbound
  starts. Canonical evaluation and holdout starts remain unchanged at rest.
  Fresh Drone policies also begin at the physically neutral -2/7 action on
  both rotors (total thrust equals gravity), with exploration variance unchanged.
  Drone uses an undiscounted finite-horizon objective (`gamma = 1.0`), and both
  crash and timeout apply the same terminal failure cost so hovering until the
  deadline is not an artificially safe strategy. Its terminal-zero shaping
  potential combines waypoint distance (0.05), error from a 20--80 unit/s
  braking-envelope velocity target (0.10), and error from the corresponding
  one-second desired tilt (5.0). At `gamma = 1.0` it telescopes to a fixed
  start-state constant, preserving terminal-outcome ordering while immediately
  rewarding the counter-thrust needed to reverse inbound momentum. Attitude
  and thrust regularizers are accumulated and charged only when the course is
  completed, ranking successful controllers by efficiency without making an
  early crash cheaper than a longer failed attempt. Other scenarios retain
  `gamma = 0.995`.
  Segment gates use suite v3 at seeds 400,000 and above; momentum controls use
  versioned suites at 500,000 and above. Both are disjoint from checkpoint
  selection (100,000+) and the default holdout (200,000+).
  Curriculum state is checkpointed exactly, and its diagnostics never enter
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
- A fixed canonical rollout is retained only for comparable ghost playback; it
  is not presented as the statistical evaluation result.
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
