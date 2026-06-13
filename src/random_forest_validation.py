import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import (
    ASSETS_DIR,
    FEATURE_NAME_MAP,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
    ensure_output_dirs,
)


def run_random_forest(X, y, random_state=RANDOM_STATE):
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=TEST_SIZE, random_state=random_state
    )

    rf_models = {
        "RF_baseline": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "RF_shallow": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    rf_results = {}
    for name, rf in rf_models.items():
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        rf_results[name] = {"rmse": rmse, "r2": r2, "model": rf}
        print(f"{name:12s}  RMSE={rmse:.5f}  R2={r2:.5f}")

    return rf_results


def plot_rf_importances(rf_results, feature_names):
    translated_names = [FEATURE_NAME_MAP.get(name, name) for name in feature_names]
    rf = rf_results["RF_baseline"]["model"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(
        range(len(importances)),
        np.array(translated_names)[idx],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Feature importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()

    return pd.DataFrame(
        {
            "feature": np.array(feature_names)[idx],
            "feature_label": np.array(translated_names)[idx],
            "importance": importances[idx],
        }
    )


def append_rf_performance(rf_results):
    ensure_output_dirs()
    performance_path = RESULTS_DIR / "model_performance.csv"

    rf_rows = pd.DataFrame(
        [
            {"model": name, "rmse": metrics["rmse"], "r2": metrics["r2"], "alpha": None}
            for name, metrics in rf_results.items()
        ]
    )

    if performance_path.exists():
        existing = pd.read_csv(performance_path)
        combined = pd.concat([existing, rf_rows], ignore_index=True)
    else:
        combined = rf_rows

    combined.to_csv(performance_path, index=False)


def main():
    from config import DATA_SAMPLE, SAMPLE_HEADER_ROW
    from data_cleaning import load_and_process_exact

    ensure_output_dirs()

    X, y, feature_names = load_and_process_exact(DATA_SAMPLE, header=SAMPLE_HEADER_ROW)
    rf_results = run_random_forest(X, y)

    importance_df = plot_rf_importances(rf_results, feature_names)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    plot_path = ASSETS_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    append_rf_performance(rf_results)

    print(f"\nSaved {RESULTS_DIR / 'feature_importance.csv'}")
    print(f"Saved {plot_path}")
    print(f"Updated {RESULTS_DIR / 'model_performance.csv'}")


if __name__ == "__main__":
    main()
