"""Render benchmark_all.py reports without loading an agent or changing state.

Requires matplotlib. Example:
  python backend/plot_control_study.py docs/reports/study.json --output docs/reports/study.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

NAMES = {
    "robot-reach": "Robot Arm Reach", "robot-tracking": "Robot Target Tracking",
    "ball-beam": "Ball & Beam", "orbital-docking": "Orbital Docking",
}
COLORS = ["#087f8c", "#b6570f", "#7755a2", "#536b27"]


def render(report, output):
    scenarios = report["selected_scenarios"]
    seeds = report["seeds"]
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(14, 3.25*len(scenarios)), squeeze=False)
    for row, scenario in enumerate(scenarios):
        runs = [run for run in report["runs"] if run["scenario_id"] == scenario]
        for run in runs:
            seed = run["seed"]
            color = COLORS[seeds.index(seed) % len(COLORS)]
            trace = run["checkpoint_trace"]
            episodes = [point["episode"] for point in trace]
            axes[row, 0].plot(episodes, [point["metric"] for point in trace],
                              color=color, label=f"Training seed {seed}", linewidth=1.6)
            axes[row, 1].plot(episodes, [point["success_rate"] for point in trace],
                              color=color, linewidth=1.6)
            holdout = run.get("holdout", {})
            if holdout.get("state") == "complete":
                rate = holdout["success_rate"]
                axes[row, 2].errorbar(seeds.index(seed), rate,
                    yerr=[[rate-holdout["success_ci_low"]], [holdout["success_ci_high"]-rate]],
                    fmt="o", color=color, capsize=6, markersize=7, linewidth=2)
                axes[row, 2].annotate(f"{holdout['successes']}/{holdout['episodes']}",
                    (seeds.index(seed), rate), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=10, color=color)
        axes[row, 0].set_title(NAMES.get(scenario, scenario), loc="left", fontweight="bold")
        axes[row, 0].set_ylabel("Position / tracking error (m) ↓")
        axes[row, 0].set_ylim(bottom=0)
        for col in (0, 1):
            axes[row, col].set_xlabel("Completed training episodes")
            axes[row, col].set_xlim(0, report["max_episodes"])
        axes[row, 1].set_ylabel("Selection-suite success")
        axes[row, 2].set_ylabel("Post-selection success")
        axes[row, 2].set_xticks(range(len(seeds)), [f"Seed {seed}" for seed in seeds])
        axes[row, 2].set_xlim(-0.55, len(seeds)-0.35)
        for col in (1, 2):
            axes[row, col].set_ylim(-0.04, 1.06)
            axes[row, col].yaxis.set_major_formatter(PercentFormatter(1))
        for axis in axes[row]:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#e4e8eb", linewidth=0.7)
            axis.set_axisbelow(True)
    axes[0, 0].legend(loc="best", fontsize=9, frameon=False)
    axes[0, 1].set_title("Repeated fixed starts", loc="left")
    axes[0, 2].set_title("Separate starts · 95% Wilson intervals", loc="left")
    holdout = report["holdout_protocol"]
    fig.suptitle("Policy Atlas | Control learning study", x=0.05, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.05, 0.94,
        f"PPO from scratch · {len(seeds)} optimizer seeds · {report['max_episodes']:,} episodes per run · "
        f"{holdout['episodes']} post-selection starts per policy", fontsize=11)
    fig.text(0.05, 0.02,
        f"Holdout seeds {holdout['seed_base']}–{holdout['seed_end']}. Curves use the repeatedly inspected selection suite.\n"
        "Intervals measure start-state uncertainty within each seed, not replication across optimizer seeds. "
        "Analytic controllers are excluded from PPO results.", fontsize=9, color="#41505a")
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.97, hspace=0.48, wspace=0.38)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.report.read_text(encoding="utf-8")), args.output)
