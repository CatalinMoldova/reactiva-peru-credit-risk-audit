import matplotlib.pyplot as plt
import pandas as pd

from config import ASSETS_DIR, PAPER_ALPHAS, RESULTS_DIR, ensure_output_dirs


def save_model_performance(results):
    ensure_output_dirs()
    rows = []
    for name, metrics in results.items():
        alpha = metrics.get("alpha")
        rows.append(
            {
                "model": name,
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "alpha": alpha,
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "model_performance.csv", index=False)


def plot_model_comparison(results):
    comparison_models = ["OLS", "Ridge (Opt)", "Lasso (Opt)", "Ridge (Paper)", "Lasso (Paper)"]
    labels = [name for name in comparison_models if name in results]
    rmses = [results[name]["rmse"] for name in labels]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, rmses, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])
    plt.title("Linear Model RMSE Comparison")
    plt.ylabel("RMSE (scaled target)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()


def create_methodology_diagram():
    ensure_output_dirs()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    steps = [
        "Load Reactiva Perú data",
        "Clean & encode features",
        "Replicate OLS / Ridge / Lasso",
        "Run leakage diagnostics",
        "Validate with Random Forest",
        "Export results & figures",
    ]
    x_positions = [0.02, 0.20, 0.38, 0.56, 0.74, 0.92]

    for x, step in zip(x_positions, steps):
        ax.text(
            x,
            0.55,
            step,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EAF2FB", "edgecolor": "#4C72B0"},
            fontsize=10,
        )

    for x_start, x_end in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate(
            "",
            xy=(x_end - 0.04, 0.55),
            xytext=(x_start + 0.08, 0.55),
            arrowprops={"arrowstyle": "->", "color": "#4C72B0", "lw": 1.5},
        )

    ax.set_title("Replication and Leakage-Audit Pipeline", fontsize=13, pad=20)
    plt.tight_layout()

    output_path = ASSETS_DIR / "methodology_diagram.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
