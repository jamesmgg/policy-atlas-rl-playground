# Arcade games: full-game validation

Validated on 2026-09-11 using a frozen offline engine. The new games are:

- **Paddle Rally:** return five consecutive wall-bouncing balls; one miss ends the game. The actor receives an explicitly engineered reflected-landing sensor.
- **Flappy Flight:** fly through six seeded pipe gaps without touching pipes, floor or ceiling. This uses continuous wing thrust, with zero command balancing gravity.
- **Coin Collector:** navigate to five separated destinations, braking below 0.2 m/s and staying within 0.1 m for three steps to collect each coin. The arena boundary kills the run; fast fly-throughs do not collect coins.

All tasks expose their remaining time in the observation and treat deadline expiry as failure. Terminals are absorbing: further actions cannot move the agent, change the scene, or award more reward. Full frame payloads contain every dynamic game object for truthful replay.

## Learning method

These are **demonstration-assisted PPO** runs. They are not evidence of from-scratch PPO discovering the games. Each scenario's contract exposes the complete initialization protocol:

1. Collect 48 full-game expert rollouts on seeds 710000–710047, adding clipped Gaussian action noise with standard deviation 0.16 to visited trajectories. Keep expert labels on every second pre-action observation.
2. Fit the actor for 55 epochs using Adam at 0.001, batch size 1024, initialization seed 42.
3. Perform two DAgger rounds of 20 learned-actor rollouts, starting at seeds 720000 and 721000. Refit the accumulated expert-labelled dataset for 35 epochs per round.
4. Fine-tune with the existing production PPO trainer, continuous exploration log standard deviation initially −2, for 200 online episodes per seed.

The final warm-start sample counts were 7,304 for Paddle, 11,880 for Flappy and 8,706 for Coin. Both PPO training seeds start from the same deterministic demonstration-trained actor; they differ in online training randomness and critic initialization. The expert is never consulted at inference or substituted for supplied environment actions.

## Selection and unseen evaluation

Selection chose the **earliest scheduled checkpoint** with at least 90% success on the production fixed suite of ten full starts (seeds 100000–100009). Checkpoints become durable at PPO rollout episode boundaries, so actual episode numbers are slightly beyond the requested multiples of 50. Only after selection, evaluate the selected actor on 50 unseen full starts, seeds 950000–950049. Neither demonstration collection nor checkpoint selection uses those seeds.

| Game | PPO seed | Selected episode | PPO updates | Online steps at selection | Unseen successes | Canonical replay |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Paddle Rally | 42 | 52 | 4 | 8,632 | 50/50 | Five returns |
| Paddle Rally | 123 | 52 | 4 | 8,632 | 50/50 | Five returns |
| Flappy Flight | 42 | 56 | 7 | 15,120 | 50/50 | Six gates |
| Flappy Flight | 123 | 56 | 7 | 15,120 | 50/50 | Six gates |
| Coin Collector | 42 | 55 | 5 | 11,051 | 50/50 | Five coins |
| Coin Collector | 123 | 53 | 5 | 10,799 | 50/50 | Five coins |

Each 50/50 result has a 95% Wilson interval of approximately **92.9%–100%**. The same paired unseen seed suite is used for both policies within a game; 300/300 should not be treated as 300 independent draws from one population. These results support reliable play in the tested start distribution, not guaranteed success on every future seed or continued training checkpoint.

On a separate paired diagnostic suite (50 seeds starting at 830000), every game's reference controller succeeds 50/50; zero-action and uniformly random controllers each succeed 0/50. The demonstration warm starts already succeed 50/50 on a different diagnostic suite starting at 820000.

PPO performance is not monotonic: Coin seed 42 briefly falls to 9/10 on its fixed suite at episode 154, returning to 10/10 at episode 200. All other saved checkpoints are 10/10. Keep the verified selected actor available rather than replacing it merely because a newer checkpoint exists.

## Reproduction and artifacts

Run `python benchmark_arcade.py --output /tmp/arcade-validation --episodes 200` from `backend/` in the backend runtime. The output must be empty, and the harness does not use any live API or persistent application volume. It writes the selection trace, complete evaluation trials, demonstration diagnostics and normal trainer checkpoint pairs.

Local artifacts are under `data/arcade-validation/runs-v1/`; its `report.json` contains all six runs. Per-seed directories contain the production checkpoint files. The exact frozen Python source tree is preserved under `data/arcade-validation/engine-v1/`.

The training engine source digest was `09fa25613d3527a9a4c565876a8d01abf54a040ce2ccbacc6df4d905e6a4933a`. This snapshot predates concurrent changes to the rest of the lab; publishing its weights under the final application requires a fresh compatible inference evaluation and retained source provenance.

`python -m unittest discover -s tests -p test_arcade_games.py -v` runs eight behavioral tests covering seeded equality, finite observations, full-game reference success, absorbing terminals, clock failure, actual collision outcomes, bird radius, coin capture dwell and braking, demonstration provenance, and the absence of inference-time reference calls. The validation container had no published ports and no mounts of live state.
