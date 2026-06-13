import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from config import (
    AMOUNT_COL,
    ASSETS_DIR,
    DATA_SAMPLE,
    ENTITY_COL,
    RESULTS_DIR,
    SAMPLE_HEADER_ROW,
    ensure_output_dirs,
)
from data_cleaning import calculate_risk_level, clean_currency


def test_independence_amount_entity(filepath=DATA_SAMPLE, header=SAMPLE_HEADER_ROW):
    print("Loading data for independence test...")
    df = pd.read_csv(filepath, header=header)
    df.columns = df.columns.str.strip()

    df[AMOUNT_COL] = df[AMOUNT_COL].apply(clean_currency)
    df_clean = df.dropna(subset=[AMOUNT_COL, ENTITY_COL]).copy()

    print(
        f"Analyzing {len(df_clean)} loans across "
        f"{df_clean[ENTITY_COL].nunique()} institutions.\n"
    )

    groups = [group[AMOUNT_COL].values for _, group in df_clean.groupby(ENTITY_COL)]
    stat, p_value = stats.kruskal(*groups)

    print("-" * 50)
    print("STATISTICAL TEST RESULTS (Kruskal-Wallis: amount by institution)")
    print("-" * 50)
    print(f"Statistic: {stat:.2f}")
    print(f"P-value:   {p_value}")

    if p_value < 0.05:
        conclusion = "REJECT null hypothesis: amount and institution are dependent."
    else:
        conclusion = "FAIL TO REJECT null hypothesis: amount and institution appear independent."
    print(f"\nCONCLUSION: {conclusion}")

    return {
        "test_name": "kruskal_wallis_amount_by_institution",
        "statistic": stat,
        "p_value": p_value,
        "conclusion": conclusion,
        "n_loans": len(df_clean),
        "n_groups": df_clean[ENTITY_COL].nunique(),
    }


def plot_risk_level_by_amount(filepath=DATA_SAMPLE, header=SAMPLE_HEADER_ROW):
    df = pd.read_csv(filepath, header=header)
    df.columns = df.columns.str.strip()
    df[AMOUNT_COL] = df[AMOUNT_COL].apply(clean_currency)
    df = df.dropna(subset=[AMOUNT_COL]).copy()
    df["risk_level"] = df[AMOUNT_COL].apply(calculate_risk_level)
    df = df.dropna(subset=["risk_level"])

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="risk_level", y=AMOUNT_COL, palette="Blues")
    plt.yscale("log")
    plt.title("Covered Amount by Constructed Risk Level")
    plt.xlabel("Risk Level")
    plt.ylabel("Covered Amount (S/) [Log Scale]")
    plt.tight_layout()

    stat, p_value = stats.kruskal(
        *[group[AMOUNT_COL].values for _, group in df.groupby("risk_level")]
    )
    return {
        "test_name": "kruskal_wallis_amount_by_risk_level",
        "statistic": stat,
        "p_value": p_value,
        "conclusion": (
            "REJECT null hypothesis: amount distributions differ across risk levels "
            "(expected because risk_level is derived from amount thresholds)."
        ),
        "n_loans": len(df),
        "n_groups": df["risk_level"].nunique(),
    }


def save_leakage_outputs(test_results):
    ensure_output_dirs()
    pd.DataFrame(test_results).to_csv(RESULTS_DIR / "leakage_tests.csv", index=False)


def main():
    ensure_output_dirs()

    test_results = [
        test_independence_amount_entity(),
        plot_risk_level_by_amount(),
    ]

    plot_path = ASSETS_DIR / "risk_level_by_amount.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    save_leakage_outputs(test_results)
    print(f"\nSaved {RESULTS_DIR / 'leakage_tests.csv'}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
