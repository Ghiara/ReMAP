import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PNG = "cost_comparison_reuse_figure.png"
OUTPUT_PDF = "cost_comparison_reuse_figure.pdf"


OURS_TOY = 16.0
OURS_DEPLOY_PER_AGENT = 0.1
OURS_LOW_LEVEL = {
    "ant": 3.30,
    "walker": 6.87,
    "hopper": 5.52,
    "cheetah": 5.99,
}

BASELINES_PER_AGENT = {
    "PEARL": 50.0,
    "RL2": 100.0,
    "MELTS": 40.0,
    "CEMRL": 45.0,
}

COLORS = {
    "toy": "#C7F9F1",
    "cheetah": "#0F766E",
    "hopper": "#F59E0B",
    "walker": "#3B82F6",
    "ant": "#EF4444",
    "deploy": "#374151",
    "PEARL": "#DC2626",
    "RL2": "#7C3AED",
    "MELTS": "#EA580C",
    "CEMRL": "#2563EB",
}


def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2.0,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def main():
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

    cheetah_total = OURS_TOY + OURS_LOW_LEVEL["cheetah"] + OURS_DEPLOY_PER_AGENT
    four_agent_low_level = sum(OURS_LOW_LEVEL.values())
    four_agent_deploy = 4 * OURS_DEPLOY_PER_AGENT
    four_agent_total = OURS_TOY + four_agent_low_level + four_agent_deploy

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
    fig.patch.set_facecolor("white")

    # Left panel: single cheetah total cost
    ax = axes[0]
    ax.set_facecolor("#FCFCFD")
    methods = ["Ours", "PEARL", "RL2", "MELTS", "CEMRL"]
    x = np.arange(len(methods))
    width = 0.72

    ax.bar(x[0], OURS_TOY, width=width, color=COLORS["toy"], label="Toy high-level")
    ax.bar(x[0], OURS_LOW_LEVEL["cheetah"], width=width, bottom=OURS_TOY, color=COLORS["cheetah"], label="Cheetah low-level")
    ax.bar(
        x[0],
        OURS_DEPLOY_PER_AGENT,
        width=width,
        bottom=OURS_TOY + OURS_LOW_LEVEL["cheetah"],
        color=COLORS["deploy"],
        label="Deployment",
    )

    baseline_names = ["PEARL", "RL2", "MELTS", "CEMRL"]
    baseline_bars = []
    for idx, name in enumerate(baseline_names, start=1):
        bars = ax.bar(x[idx], BASELINES_PER_AGENT[name], width=width, color=COLORS[name], label=name)
        baseline_bars.extend(bars)

    ours_bar = ax.bar(x[0], cheetah_total, width=width, color="none", edgecolor="#0B3B37", linewidth=1.5)
    add_value_labels(ax, ours_bar)
    add_value_labels(ax, baseline_bars)

    ax.set_title("Single-Agent Total Cost (Half-Cheetah)")
    ax.set_ylabel("Env steps (M)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    # Right panel: transfer to 4 agents total cost
    ax2 = axes[1]
    ax2.set_facecolor("#FCFCFD")
    methods2 = ["Ours", "PEARL", "RL2", "MELTS", "CEMRL"]
    x2 = np.arange(len(methods2))

    ax2.bar(x2[0], OURS_TOY, width=width, color=COLORS["toy"], label="Toy high-level")
    bottom = OURS_TOY
    agent_order = ["cheetah", "hopper", "walker", "ant"]
    agent_labels = {
        "cheetah": "Cheetah low-level",
        "hopper": "Hopper low-level",
        "walker": "Walker low-level",
        "ant": "Ant low-level",
    }
    for agent_name in agent_order:
        ax2.bar(
            x2[0],
            OURS_LOW_LEVEL[agent_name],
            width=width,
            bottom=bottom,
            color=COLORS[agent_name],
            label=agent_labels[agent_name],
        )
        bottom += OURS_LOW_LEVEL[agent_name]
    ax2.bar(
        x2[0],
        four_agent_deploy,
        width=width,
        bottom=bottom,
        color=COLORS["deploy"],
        label="Transfer deployment",
    )

    baseline_totals = {name: 4 * value for name, value in BASELINES_PER_AGENT.items()}
    baseline_bars2 = []
    for idx, name in enumerate(baseline_names, start=1):
        bars = ax2.bar(x2[idx], baseline_totals[name], width=width, color=COLORS[name], label=name)
        baseline_bars2.extend(bars)

    ours_bar2 = ax2.bar(x2[0], four_agent_total, width=width, color="none", edgecolor="#0B3B37", linewidth=1.5)
    add_value_labels(ax2, ours_bar2)
    add_value_labels(ax2, baseline_bars2)

    ax2.set_title("Total Cost After Transfer To 4 Agents")
    ax2.set_ylabel("Env steps (M)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods2)
    ax2.set_ylim(0, 430)
    ax2.grid(axis="y", linestyle=":", alpha=0.35)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["toy"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["cheetah"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["hopper"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["walker"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["ant"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["deploy"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["PEARL"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["RL2"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["MELTS"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["CEMRL"]),
    ]
    labels = [
        "Toy high-level",
        "Cheetah low-level",
        "Hopper low-level",
        "Walker low-level",
        "Ant low-level",
        "Deployment",
        "PEARL",
        "RL2",
        "MELTS",
        "CEMRL",
    ]

    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Decoupled Reuse Greatly Reduces Total Training Cost",
        fontsize=15,
        fontweight="bold",
        y=1.08,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82, wspace=0.22)

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
