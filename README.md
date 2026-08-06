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

- Generalized advantage estimation cuts traces at the correct transition and
  distinguishes external truncations from intrinsic puzzle deadlines.
- Finite-horizon observations expose remaining time. Driving policies also see
  track phase, cumulative objective state, wrong-way margin, recent progress,
  and one-time traffic-pass state used by rewards or termination.
- PPO uses a tanh-squashed Gaussian and applies the matching log-probability
  correction, so sampled actions and optimized likelihoods agree.
- Raw scores remain unchanged for people and leaderboards; PPO receives the
  same rewards multiplied by a disclosed positive constant so critic targets
  remain numerically well-conditioned. Constant entropy pressure is disabled,
  allowing continuous controls to become precise.
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
  independent training seeds are directly comparable. Reports include mean, standard
  deviation, success rate, a 95% Wilson interval, evaluation count, seed, and
  update count.
- Driving training uses a disclosed 75% canonical / 25% rolling-checkpoint
  mixture. Rolling states reconstruct speed, clock, task progress, fuel, and
  traffic from a curvature/grip braking envelope; selection and replay keep
  their unchanged fixed start-line distributions.
- Lunar Lander uses a fixed-suite, performance-gated reverse-altitude
  curriculum: touchdown rehearsals first, then overlapping 30-100, 50-200,
  and 100-500 unit approach bands before the canonical descent. Drone Course
  likewise learns the final waypoint first, then unlocks each earlier course segment.
  Lander's touchdown frontier advances after one >=90% deterministic 20-start
  evaluation; harder Lander frontiers use >=75%, while Drone frontiers retain
  their >=90% gate. Once a Lander frontier has been
  mastered, half of resets stay on the active frontier and the other half
  retain mastered easier work. Curriculum state is checkpointed exactly, and
  its diagnostics never enter full-course checkpoint selection.
- The benchmark campaign freezes checkpoint selection before an optional
  100-start holdout at a disjoint seed range. Holdout outcomes cannot affect
  early stopping or checkpoint choice.
- The default campaign selects and stops at the first statistically qualifying
  fixed-suite checkpoint; the independent 100-start holdout is the confirmation
  layer. `--confirmations` remains available for stricter selection studies.
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
gold/silver/bronze policy podium; technical evidence is collapsed by default.
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
disjoint holdout. See
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
