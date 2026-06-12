from pathlib import Path
import json
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "songs.db"
IMAGE_DIR = PROJECT_ROOT / "images"
RESULTS_PATH = PROJECT_ROOT / "results.json"

FEATURE_COLUMNS = [
    "bpm",
    "dynamic_variation_score",
    "sub_bass",
    "bass",
    "low_mid",
    "mid",
    "high_mid",
    "presence",
    "brilliance",
    "spectral_spread",
    "spectral_balance",
    "timing_deviation",
    "asymmetry_bias",
    "asymmetry_index",
]


def load_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            """
            SELECT
                Track,
                bpm,
                dynamic_variation_score,
                sub_bass,
                bass,
                low_mid,
                mid,
                high_mid,
                presence,
                brilliance,
                spectral_spread,
                spectral_balance,
                timing_deviation,
                asymmetry_bias,
                asymmetry_index,
                rating
            FROM songs
            """,
            con,
        )


def evaluate_cv(model, X, y, cv) -> tuple[float, float]:
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mae_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )

    mean_r2 = r2_scores.mean()
    mean_mae = -mae_scores.mean()

    return mean_r2, mean_mae


def save_actual_vs_predicted_plot(y_test, predictions) -> None:
    IMAGE_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions)
    plt.plot([1, 10], [1, 10])
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Actual vs Predicted Rating")
    plt.tight_layout()
    plt.savefig(
        IMAGE_DIR / "actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_feature_importance_plot(model, feature_names) -> None:
    IMAGE_DIR.mkdir(exist_ok=True)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(
        IMAGE_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_results(results: dict) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


def main() -> None:
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    ridge = Ridge(alpha=1.0)
    ridge_cv_r2, ridge_cv_mae = evaluate_cv(ridge, X, y, cv)

    random_forest = RandomForestRegressor(
    n_estimators= 300,
    random_state= 42
    )

    random_forest_cv_r2, random_forest_cv_mae = evaluate_cv(random_forest, X, y, cv)

    extra_tree = ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
    )

    extra_tree_cv_r2, extra_tree_cv_mae = evaluate_cv(extra_tree, X, y, cv)

    extra_tree.fit(X_train, y_train)
    predictions = extra_tree.predict(X_test)

    test_r2 = r2_score(y_test, predictions)
    test_mae = mean_absolute_error(y_test, predictions)

    save_actual_vs_predicted_plot(y_test, predictions)
    save_feature_importance_plot(extra_tree, FEATURE_COLUMNS)

    results = {
        "dataset_size": len(df),
        "ridge_cv_r2": round(ridge_cv_r2, 3),
        "ridge_cv_mae": round(ridge_cv_mae, 3),
        "random_forest_cv_r2": round(random_forest_cv_r2, 3),
        "random_forest_cv_mae": round(random_forest_cv_mae, 3),
        "extra_tree_cv_r2": round(extra_tree_cv_r2, 3),
        "extra_tree_cv_mae": round(extra_tree_cv_mae, 3),
        "extra_tree_test_r2": round(test_r2, 3),
        "extra_tree_test_mae": round(test_mae, 3),
    }

    save_results(results)

    print("Training complete.")
    print(f"Dataset size: {results['dataset_size']} songs")
    print(f"Extra Tree CV R²: {results['extra_tree_cv_r2']}")
    print(f"Extra Tree CV MAE: {results['extra_tree_cv_mae']}")
    print("Plots saved to images/")
    print("Results saved to results.json")


if __name__ == "__main__":
    main()