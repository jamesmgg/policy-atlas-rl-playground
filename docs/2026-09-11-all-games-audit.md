# All-games learning and playback audit — 2026-09-11

This revision expands the lab to **23 games** with selected neural policies from **two PPO training seeds, 42 and 123, for every game**. All **46 policies passed final combined-engine qualification**: 2,300/2,300 qualification starts, 460/460 fixed starts and 46 successful canonical replays. The training evidence below comes from recorded, frozen engines; the final qualification separately checks those frozen weights on the released engine.

## What changed

- Added Paddle Rally, Flappy Flight and Coin Collector, with seeded starts, explicit observations and deadlines, real collision/failure conditions, and absorbing terminal states.
- Added disclosed demonstration initialization for difficult tasks, including full-descent Lunar Lander, Mountain Car, Orbital Docking, Robot Arm Reach, Ball & Beam and the remaining driving games. Existing Drone Course and Traffic Rush assistance remains disclosed. Experts provide training labels; evaluated gameplay uses the neural actor alone.
- Added a scenario-specific PPO learning rate. Apex GP, Drift Trial and Robot Arm Reach use `3e-5` to reduce disruption of their learned demonstration policies; other tasks retain their declared defaults. Robot Reach also needed corrective demonstrations near its targets.
- Corrected success reporting: an unfinished Pendulum attempt cannot report a completed balance, and a tipped Drone cannot win by touching its last waypoint.
- All newly saved replays, including driving, preserve complete frames: secondary objects, targets, counters, actuator visuals and the terminal outcome. Legacy archives can still contain only a trajectory. Playback ends on its final frame rather than looping into another attempt. Scenario identity accompanies replay messages so delayed messages cannot become another project's scene.
- Added optional automatic pause at successful checkpoint evaluation, visible learning-method disclosure, and archived verified-policy playback that leaves the user's active training weights in place.

All games except **Continuous Cart-Pole, Pendulum Swing-Up and Robot Target Tracking** now use demonstration-assisted training. The assisted actor first fits expert examples, sometimes with additional labels from its own rollout states, then continues with PPO practice. These results must not be presented as unaided PPO discovering every task. Two PPO seeds can share the same deterministic demonstration initialization; they replicate online training randomness, not independent demonstration datasets.

## Training evidence index

The entries identify selected **original training episodes**, not qualification-export episode numbers. Each listed policy passed **50/50 post-selection starts on its recorded training engine**. These suites varied by study; the report paths below preserve their seed ranges, selection rules, checksums and individual outcomes. They are not evidence that the later combined engine has already passed.

| Game ID | Seed 42 episode | Seed 123 episode | Evidence |
| --- | ---: | ---: | --- |
| `apex-gp` | 10 | 10 | A |
| `velocita` | 10 | 12 | D |
| `grandville` | 12 | 12 | D |
| `thunder-oval` | 10 | 12 | D |
| `apex-gp-wet` | 12 | 10 | D |
| `glacier` | 12 | 10 | D |
| `rally-ridge` | 12 | 10 | R |
| `kart-sprint` | 12 | 21 | R |
| `drift-trial` | 20 | 31 | R, learning rate `3e-5` |
| `eco-gp` | 11 | 81 | R |
| `traffic-rush` | 28 | 63 | C |
| `lunar-lander` | 57 | 61 | L |
| `pendulum-swingup` | 264 | 402 | C |
| `drone-hover` | 33 | 64 | C |
| `cartpole-balance` | 212 | 251 | C |
| `mountain-car` | 26 | 26 | M |
| `orbital-docking` | 27 | 28 | O |
| `robot-reach` | 70 | 69 | H, corrective demonstrations and learning rate `3e-5` |
| `robot-tracking` | 525 | 427 | O |
| `ball-beam` | 62 | 62 | B, assisted replacement |
| `paddle-rally` | 52 | 52 | G |
| `flappy-flight` | 56 | 56 | G |
| `coin-collector` | 55 | 53 | G |

Local evidence paths, relative to the repository:

- The [existing-policy manifest](reports/2026-09-11-existing-policy-manifest.json) indexes all 22 policies in A, D, C and M with exact tensor/sidecar paths and hashes. Canonical checks for older seed-123 foundation policies and Mountain Car remain part of final qualification.
- **A:** `data/existing-learning-20260911/revision4/apex-ppo.json`.
- **D:** `data/existing-learning-20260911/revision2/remaining-driving-ppo.json`.
- **R:** `data/guided-driving-arcade-20260911/selected-manifest.json`, including exact checkpoint/sidecar paths, independently checked tensor hashes and per-run report/source paths.
- **C:** seed 42 in `data/existing-learning-20260911/revision4/classic-seed42.json`; seed 123 in `data/existing-learning-20260911/remaining-seed123.json`.
- **M:** `data/existing-learning-20260911/revision1/mountain-ppo.json`.
- **O:** `data/control-learning-20260911-v1/control-study.json`; its older Ball & Beam and Robot Reach entries are superseded by B and H respectively.
- **L:** [Lander report](results/lander-v12-demonstration-ppo-holdouts.json) and [learning diagnosis](2026-09-11-lander-learning.md).
- **B:** [Ball & Beam report](results/ballbeam-guided-qualification.json) and [learning diagnosis](2026-09-11-ballbeam-learning.md).
- **H:** [Robot Reach report](results/robot-reach-corrective-qualification.json), [correction study](2026-09-11-robot-reach-corrections.md), and `data/robot-reach-corrective-study-20260911/study.json`.
- **G:** `data/arcade-validation/runs-v1/report.json` and [arcade protocol](arcade-validation.md).

The runtime weights and frozen source copies remain local artifacts under `data/`; original training provenance must be retained when exporting them for another engine.

## Failures retained, not erased

**Ball & Beam:** the earlier unassisted seed-123 policy passed its ten fixed starts but only **48/50** starts in the control study. A fresh unassisted run selected episode 611 and reached **49/50** qualification starts at seeds 5300000–5300049; the failure stopped nearly motionless 13.5 cm from the target. Adding demonstrations preserved the physics and strict settling criterion, and the new seed-42/123 policies each passed 50/50 on a different suite at 5400000–5400049.

**Robot Reach:** the earlier seed-123 episode-56 policy passed ten fixed starts and its canonical replay, but only **46/50** qualification starts at seeds 5200000–5200049. Reducing the learning rate alone still yielded **49/50 for both seeds** at 5500000–5500049. The demonstration clone already missed that target before PPO, so smaller updates alone could not fix the near-goal approximation gap. Adding supervised corrective states around target joint configurations, while retaining `3e-5`, produced the selected episode-70/69 policies. Each passed ten fixed starts, **50/50 new starts at 5600000–5600049**, and its canonical replay, with terminal errors of 0.0066 m and 0.0148 m. Physics, observations, resets, success tolerances and the horizon were unchanged. Both older suites now pass as regression checks; they are not fresh holdouts. See the [correction study](2026-09-11-robot-reach-corrections.md) and its [detailed report](results/robot-reach-corrective-qualification.json).

**Drift Trial:** the original `3e-4` seed-42 run regressed for hundreds of episodes. It eventually saved a **10/10 fixed-suite checkpoint at episode 430 during interruption**, before any holdout or canonical check. It is an interrupted, unverified baseline, not proof that this learning rate can never solve the game. The `3e-5` runs qualified much earlier at episodes 20 and 31 and each passed 50/50 separate starts plus the canonical lap. See `data/guided-driving-arcade-20260911/interrupted-baseline-summary.json`.

**Apex GP:** a successful demonstration clone deteriorated under the larger PPO learning rate. The `3e-5` variation preserved 10/10 fixed-suite performance through the tested 50-episode continuation for both seeds; the selected episode-10 policies also passed 50/50 separate starts. This supports the smaller update size, not a promise that indefinite continued training cannot regress.

The [Lander study](2026-09-11-lander-learning.md) also retains later PPO regressions. More training and a higher training reward are not substitutes for evaluated task completion.

## Using the new arcade games

Open **Projects → Arcade**, choose a game, then use **Train** for agent practice or **Results → Watch verified policy** when a qualified archive is available.

- **Paddle Rally:** the agent moves the paddle horizontally to return five ricocheting balls within 18 seconds. One miss ends the game. Watch the returns counter; the landing-point sensor is explicitly supplied to the policy.
- **Flappy Flight:** the agent adjusts continuous wing thrust through six changing pipe gaps within 15.2 seconds. Pipe, floor or ceiling contact ends the flight. This is continuous thrust control, not a binary tap interface.
- **Coin Collector:** the agent steers and brakes onto five separated coins within 24 seconds. A coin requires three consecutive steps within 0.1 m at less than 0.2 m/s; flying through does not collect it. Crossing the electric arena boundary ends the run.

Automatic pause saves and stops training when a checkpoint passes **all fixed tests, with at least ten starts, and its canonical replay succeeds**. It then shows that saved rollout. This is a limited selection rule, not a 50-start holdout guarantee or proof about all possible initial conditions. The option can be disabled for longer experiments; successful saved policies remain recoverable.

**Watch verified policy** reads an archived recorded rollout. It does not restore that archive into the active training branch, retrain an actor, or replace active weights. **Load for training** is a different operation. Requalified exports preserve the original training episode/source/hash but initialize new optimizer and RNG state; they are not exact continuations of the original run.

## Final combined-engine qualification — passed

All **46 selected policies**, covering 23 game IDs × seeds 42/123, were evaluated with frozen neural weights and no expert control or additional optimizer updates. Each passed the ten fixed starts, **50/50 qualification starts at seeds 5200000–5200049**, and its canonical replay. This range was observed during integration and reused after the Robot Reach correction: the final run is regression verification, not a new independent holdout. Individual studies above also retain their post-selection suites. A 50/50 result has a 95% Wilson lower bound of approximately 92.9%; it does not guarantee universal success or indefinite stability under continued PPO training.

Completion record:

- Final engine SHA-256: `0cc1565b15ba8dc3ef3a2b34bbe5bd5771c7d3eb926f1bd96162a5bca8a43da7`.
- [Selected-policy manifest](results/all-games-selected-policy-manifest.json), [full qualification report](results/all-games-final-qualification.json), and [qualified package manifest](results/all-games-qualified-package.json).
- Accepted policies / requested policies: **46 / 46**; no final failures or exclusions.
- Earlier integration evidence remains in `data/final-qualification-20260911/parallel/qualification-report.json`: 43 accepted, one real Robot Reach failure (46/50), and two older-schema policies rejected before evaluation. The Robot Reach policy was replaced after corrective training. Drone seed 123 (schema 17 → 18) and Pendulum seed 123 (2 → 3) were explicitly authorized for unchanged-weight requalification under the corrected terminal rules; both passed every final gate. Original metadata was never relabeled or overwritten.
- Fresh complete backend regression: **306 tests passed** in four independent processes against the final frozen source. All **23 scenario smoke tests passed** in the built production image. **32 frontend tests**, TypeScript and the production bundle passed.
- Runtime policy pairs remain local under `data/release-validation-20260911/package`. Exports create new episode-zero archive branches and retain the original training episode, schema, engine, tensor hash, demonstration contract and training counts in hash-bound provenance. No old optimizer state is represented as a fresh engine continuation.
- The complete disposable live E2E test passed: catalog → scenario switch → live frame → PPO update → fixed evaluation → seeded reset → archived branch restore, plus all four control reference replays and paired evaluations.
- Mobile browser checks at 390 × 844 covered project categories and switching, training controls, automatic success pause, verified seed-42/123 replay switching, reference-to-neural replay switching, restart and held terminal outcomes. Desktop fullscreen was checked separately. A final visual correction moved telemetry below the canvas, aligned the lander's feet to its point-contact physics, and hid exhaust after termination; both the mobile and fullscreen lander visibly rest on the pad. No dynamics or saved policy weights changed for these display fixes.
- Deployment reused the existing `rl-simulator_rl-checkpoints` volume. A 1,262,858,240-byte pre-upgrade backup is retained at `data/deploy-backup-20260911-games/checkpoints-before-games.tar`. Installation added 46 archive branches and verified all 315 existing active files were byte-for-byte unchanged. A read-only post-deploy check validated all 46 archive pairs and confirmed the previous Pendulum episode-500 checkpoint remains in its schema archive.
- Production reports the exact qualified engine hash, 23 scenarios and an idle training state. Both qualified Pendulum policies are visibly available in Results. The frontend serves successfully at `http://100.72.181.81:8900` through the existing Tailscale route; no public exposure was added.
- Older engine/schema checkpoints remain archived for inspection. They are not silently resumed under changed physics or relabeled as newly qualified policies. The active Pendulum branch starts at episode zero on the new schema; its old episode-500 run is retained.
- [Deployment verification](results/all-games-deployment-verification.json) records the checks and exact frontend assets.
