import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import PAPER_ALPHAS, RANDOM_STATE, TEST_SIZE
from generate_outputs import create_methodology_diagram, plot_model_comparison, save_model_performance


def run_optimized_regression(X, y, feature_names):
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("\nFinding Optimal Lambdas (Alphas) via Cross-Validation...")

    alphas_ridge = np.logspace(-5, 1, 100)
    ridge_cv = RidgeCV(alphas=alphas_ridge, scoring="neg_mean_squared_error", cv=5)
    ridge_cv.fit(X_train, y_train)
    best_alpha_ridge = ridge_cv.alpha_
    print(
        f"Best Ridge Alpha: {best_alpha_ridge:.6f} "
        f"(Paper: {PAPER_ALPHAS['ridge']})"
    )

    lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso_cv.fit(X_train, y_train)
    best_alpha_lasso = lasso_cv.alpha_
    print(
        f"Best Lasso Alpha: {best_alpha_lasso:.6f} "
        f"(Paper: {PAPER_ALPHAS['lasso']})"
    )

    models = {
        "OLS": LinearRegression(),
        "Ridge (Opt)": Ridge(alpha=best_alpha_ridge),
        "Lasso (Opt)": Lasso(alpha=best_alpha_lasso),
        "Ridge (Paper)": Ridge(alpha=PAPER_ALPHAS["ridge"]),
        "Lasso (Paper)": Lasso(alpha=PAPER_ALPHAS["lasso"]),
    }

    results = {}
    print("\nResults (Target: Min-Max Scaled Risk Level [0,1]):")
    print("-" * 60)
    print(f"{'Model':<15} | {'RMSE':<8} | {'R²':<8} | {'Alpha'}")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        alpha_val = getattr(model, "alpha", 0)

        print(f"{name:<15} | {rmse:.5f}   | {r2:.5f}   | {alpha_val}")
        results[name] = {
            "rmse": rmse,
            "r2": r2,
            "alpha": alpha_val if alpha_val else None,
            "coefficients": model.coef_,
        }

    return results


def main():
    from config import ASSETS_DIR, DATA_SAMPLE, SAMPLE_HEADER_ROW
    from data_cleaning import load_and_process_exact

    X, y, feature_names = load_and_process_exact(DATA_SAMPLE, header=SAMPLE_HEADER_ROW)
    results = run_optimized_regression(X, y, feature_names)

    save_model_performance(results)
    plot_model_comparison(results)

    comparison_path = ASSETS_DIR / "model_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()

    methodology_path = create_methodology_diagram()

    print("\nSaved results/model_performance.csv")
    print(f"Saved {comparison_path}")
    print(f"Saved {methodology_path}")


if __name__ == "__main__":
    main()
