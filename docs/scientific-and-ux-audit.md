# RL Playground scientific and UX audit

This audit separates three questions that are easy to blur together:

1. Can PPO find a policy on the fixed ten-start checkpoint-selection suite?
2. Does the frozen policy generalize to 100 disjoint environment seeds?
3. Is the experiment understandable and enjoyable to operate without exposing
   every diagnostic by default?

The campaign cap is 2,000 **episodes** per training run. A puzzle is called
solved only when a checkpoint first passes the fixed selection contract and
then passes the untouched 100-start holdout. Training successes, reward peaks,
and curriculum-only frontier tests are diagnostic evidence, not solves.

## Experiment evidence matrix

| Experiment | Main learning improvement or finding | Selected episode | Disjoint holdout |
| --- | --- | ---: | ---: |
| Apex GP | Fully observed driving state and rolling-start curriculum | 725 | 100/100 |
| Velocità | Curvature/grip braking-envelope starts | 1,377 | 97/100 |
| Grandville Streets | Dense signed progress with canonical replay retained | 726 | 100/100 |
| Thunder Oval | Removed the scenario's harmful positive-throttle actor prior | 300 | 100/100 |
| Apex GP — Wet | 90 s horizon plus equal unsafe-terminal costs removed the inactivity optimum | 1,150 | 100/100 |
| Glacier Lake | Grip-aware observations and rolling speed reconstruction | 100 | 100/100 |
| Rally Ridge | Correct crash termination/visual persistence and rally progress/drift shaping | 1,825 | 97/100 |
| Kart Sprint | Shared driving curriculum with unchanged fixed start | 1,501 | 93/100 |
| Drift Trial | Progress coefficient restored while style remained the task metric | 1,704 | 96/100 |
| Eco GP | Fuel state and objective completion made observable | 876 | 96/100 |
| Traffic Rush | 90 s reachability, traffic-state reconstruction, and staged overtake rehearsal; terminal-cost follow-up running | — | Active |
| Lunar Lander | Low-variance actor prior and performance-gated altitude curriculum; frontier-threshold follow-up pending | — | Active |
| Pendulum Swing-Up | Correct continuous-action likelihood and timeout bootstrapping | 252 | 100/100 |
| Drone Course | Reverse waypoint curriculum and momentum-aware handoff rehearsal; follow-up pending | — | Active |
| Continuous Cart-Pole | Added a disclosed continuous-force, semi-implicit control task | 276 | 100/100 |
| Mountain Car | Added normalized continuous Mountain Car dynamics | 53 | 100/100 |

The machine-readable reports live in [`docs/results`](results/). Selection uses
seeds 100000–100009; holdout uses seeds 200000–200099. Reports record the
engine source digest, checkpoint digest, evaluation suite, Wilson interval,
and whether evidence influenced selection.

## AI engineer viewpoint

### What materially improved scientific correctness

- The continuous policy is a tanh-squashed Gaussian whose optimized log
  probability matches the bounded action actually sent to the environment.
- Time-limit endings are marked as truncations and bootstrap the critic;
  physical failures remain true terminals.
- Partial rollouts are trained instead of being discarded, and episode state
  is not silently reset at arbitrary rollout boundaries.
- Observations expose task-relevant hidden state: remaining time, checkpoint
  progress, stall margin, fuel, traffic positions/pass masks, and objective
  completion. This substantially reduces accidental partial observability.
- Rewards shown to users remain in task units while PPO receives a disclosed
  positive scaling constant. Critic scale, bias, clipping, explained variance,
  action spread, and KL diagnostics are saved with checkpoints.
- Curriculum states reconstruct speed, heading, elapsed clock, fuel, and
  traffic rather than teleporting an otherwise impossible state. Curriculum
  tests are training diagnostics only; canonical policy selection is unchanged.
- Fixed selection and disjoint holdout are separated. The holdout cannot alter
  checkpoint choice or early stopping.
- RNG state, schema versions, engine hashes, checkpoint hashes, and archived
  branch integrity checks make continuation and comparison auditable.

### Highest-priority remaining scientific work

1. Replicate final configurations over at least 3–5 independent **training**
   seeds and report median sample efficiency plus dispersion. A 100-seed
   holdout measures environment-start robustness, not optimizer-seed variance.
2. Add reference baselines: random, scripted heuristic, tabular where
   applicable, and at least one second deep-RL algorithm such as SAC. PPO-only
   success cannot distinguish task quality from algorithm compatibility.
3. Add observation/reward normalization ablations and gradient/advantage
   diagnostics. Manual normalization is explicit but still scenario-specific.
4. Add property-based dynamics tests for conservation/bounds, collision
   geometry, checkpoint monotonicity, and invariance under coordinate transforms.
5. Version environment dynamics independently from training protocol so a
   renderer-only change never appears to alter a scientific task identity.
6. Publish learning curves with across-seed confidence bands and area-under-
   curve/sample-efficiency comparisons, not only the winning checkpoint.

The vehicle, lander, and drone dynamics remain educational approximations.
They are coherent control environments, not validated engineering models, and
their results should not be interpreted as real vehicle or flight performance.

## Common-user viewpoint

The strongest product decision is to treat the playground as something to
watch first and inspect second. The active experiment now opens with the live
visualization, immediately followed by a Top Runs podium. Gold, silver, and
bronze cards expose success and the task-native score, with accessible medal
labels and replay/resume actions. Learning charts, confidence intervals,
protocol hashes, saved-run tables, and compute statistics remain available but
start collapsed behind clearly named controls.

The dark instrument-panel visual language gives the simulator a consistent
identity without turning every value into a glowing dashboard widget. Family
filters, search, difficulty labels, responsive navigation, a mobile setup dock,
minimum touch targets, visible connection/training state, and explicit
pause/resume/new-run actions reduce the cost of moving among 16 experiments.
Only protocol-compatible runs enter the podium, so a friendly leaderboard does
not silently compare incompatible science.

### Highest-priority remaining UX work

1. Add a first-run, three-step tour: choose an experiment, choose a budget,
   then watch and compare. It should be dismissible and never block the canvas.
2. Visualize failure causes directly on replay (crash, stall, timeout, contact)
   and mark curriculum start versus canonical start.
3. Add a comparison workspace for two or three frozen policies with synchronized
   replay and small, plain-language difference cards.
4. Offer named run presets such as Quick look, Standard, and Full 2,000 rather
   than making episode count the first decision for a new user.
5. Add export/share for one run bundle: replay, task contract, seed, scores,
   protocol, and source/checkpoint hashes.
6. Preserve experiment/filter/scroll state in the URL so exploration is
   navigable and browser Back behaves predictably.

## Simulator roadmap

The existing registry-driven experiment contract should remain the source of
truth: observations, actions, reward terms, termination, metric direction,
scene payload, horizon, schema, and renderer capability. New experiments then
reuse the same page order—visualization, podium, optional diagnostics, setup—
rather than introducing a bespoke dashboard per simulator.

| Simulator family | What it teaches | UI/renderer addition |
| --- | --- | --- |
| Contextual bandits | Exploration, regret, nonstationarity | Arm/reward strip and cumulative-regret chart |
| Tabular Gridworld | Values, policies, credit assignment | Grid cells with value heatmap and policy arrows |
| Partially observable maze | Belief state and memory | Fog-of-war canvas plus optional belief overlay |
| Multi-agent pursuit/traffic | Cooperation, competition, emergent conventions | Multiple color-coded agents and per-agent policy selector |
| Manipulator reach/stack | Continuous control and sparse goals | Jointed-arm primitive with target/contact layers |
| Inventory/resource control | Long-horizon planning and constraints | Timeline, stock bars, and event annotations |
| Offline RL lab | Dataset bias and extrapolation error | Dataset coverage view beside frozen-policy evaluation |
| Robust-control lab | Domain randomization and distribution shift | Train/test distribution controls and robustness curve |
| Safe/constrained RL | Reward versus risk budgets | Constraint gauge and Pareto-front podium mode |

Renderer capabilities should be declared (track, rigid body, grid, graph,
timeline, multi-agent) and lazily composed from primitives. The experiment
switcher can then filter by family, control type, difficulty, observation type,
and renderer capability while the primary interaction stays identical.
