"""
draft_model.py — Predict draft value (continuous overall-pick-equivalent, where
undrafted players are slotted beyond the last real pick of their class) from
combine measurables + position.

Note on model choice: the sandbox this was built in has no network access, so
xgboost/lightgbm could not be installed. We use scikit-learn's
GradientBoostingRegressor and HistGradientBoostingRegressor as drop-in
substitutes (same core algorithm family: gradient-boosted trees) alongside
RandomForest. Swap in xgboost.XGBRegressor / lightgbm.LGBMRegressor with the
same fit/predict API if running this locally with those packages installed —
the pipeline does not otherwise depend on sklearn internals.
"""
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

FEATURES_NUM = ["Height", "Weight", "40-yd Dash", "Vertical Jump", "Bench Press",
                 "Broad Jump", "3-Cone Drill", "20-yd Shuttle"]
FEATURES_CAT = ["PosGroup"]
TARGET = "DraftValue"

MODELS = {
    "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=3,
                                            random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                    subsample=0.8, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05,
                                                            random_state=42),
}


def build_pipeline(model):
    numeric_transform = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_transform = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([
        ("num", numeric_transform, FEATURES_NUM),
        ("cat", cat_transform, FEATURES_CAT),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def pick_to_round_label(pick_value, max_pick_lookup):
    """Convert a predicted DraftValue back into an approximate round label for readability."""
    if pick_value <= 32:
        return "Round 1"
    elif pick_value <= 64:
        return "Round 2"
    elif pick_value <= 105:
        return "Round 3"
    elif pick_value <= 141:
        return "Round 4"
    elif pick_value <= 178:
        return "Round 5"
    elif pick_value <= 220:
        return "Round 6"
    elif pick_value <= 262:
        return "Round 7"
    else:
        return "Undrafted"


def main():
    df = pd.read_csv("data/processed/combine_with_sai.csv")

    # Require at least height/weight/40 present to be modelable (minimum combine footprint)
    model_df = df.dropna(subset=["Height", "Weight"]).copy()

    X = model_df[FEATURES_NUM + FEATURES_CAT]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, model_df.index, test_size=0.2, random_state=42
    )

    results = {}
    fitted = {}
    for name, model in MODELS.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
        fitted[name] = pipe
        print(f"{name:22s} MAE={mae:7.2f}  RMSE={rmse:7.2f}  R2={r2:6.4f}")

    best_name = min(results, key=lambda k: results[k]["MAE"])
    best_pipe = fitted[best_name]
    print(f"\nBest model by MAE: {best_name}")

    # Feature importance: use permutation importance rather than native
    # feature_importances_, since HistGradientBoostingRegressor (often the best
    # model by MAE) doesn't expose native importances the way RF/GBM do. This
    # also makes importances directly comparable across all three model types.
    from sklearn.inspection import permutation_importance
    pre = best_pipe.named_steps["pre"]
    cat_names = list(pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(FEATURES_CAT))
    feature_names = FEATURES_NUM + cat_names

    perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    # permutation_importance operates on the raw X_test columns (pre-encoding), so
    # importances are already per-original-feature (no need to expand one-hot cols)
    raw_cols = FEATURES_NUM + FEATURES_CAT
    fi = sorted(zip(raw_cols, perm.importances_mean), key=lambda x: -x[1])
    print("\nGlobal permutation feature importances (best model, on held-out test set):")
    for name, imp in fi:
        print(f"  {name:20s} {imp:.4f}")
    fi_dict = {n: round(float(i), 5) for n, i in fi}

    # Full-data predictions for downstream steals / SAI-vs-predicted analysis
    all_preds = best_pipe.predict(X)
    model_df["PredictedDraftValue"] = all_preds
    model_df["PredictedRound"] = model_df["PredictedDraftValue"].apply(lambda v: pick_to_round_label(v, None))
    model_df["DraftDelta"] = model_df["DraftValue"] - model_df["PredictedDraftValue"]  # positive = fell further than expected (steal if drafted well) -> see draft_steals.py for sign convention

    model_df.to_csv("data/processed/combine_with_predictions.csv", index=False)

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_pipe, f)

    with open("models/model_results.json", "w") as f:
        json.dump({
            "results_by_model": results,
            "best_model": best_name,
            "feature_importance_best_model": fi_dict,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "target_definition": "DraftValue = actual overall pick number; undrafted players "
                                  "set to (max pick in their class + 50) so the model treats "
                                  "them as 'drafted much later' on a continuous scale.",
        }, f, indent=2)

    print(f"\nSaved model_df with predictions: {len(model_df)} rows")


if __name__ == "__main__":
    main()
