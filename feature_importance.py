"""
feature_importance.py — Which combine drills actually matter, by position?

For each position group with enough data, we:
  1. Fit a RandomForestRegressor on DraftValue using only that position's players
  2. Extract feature importances (how much each drill matters for draft outcome
     WITHIN that position, not across positions)
  3. Cross-check with a simple correlation (Spearman) between each drill and
     draft value, since importance and correlation can disagree in
     informative ways (e.g. a drill can be "important" to a tree model due to
     interactions even with weak marginal correlation)

Output: outputs/position_profiles/drill_importance.json + printed summary.
"""
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr

DRILLS = ["Height", "Weight", "40-yd Dash", "Vertical Jump", "Bench Press",
           "Broad Jump", "3-Cone Drill", "20-yd Shuttle"]
POSITIONS = ["QB", "RB", "WR", "TE", "OL", "EDGE", "LB", "CB", "S", "DL"]
MIN_N = 60


def analyze_position(df: pd.DataFrame, pos: str) -> dict:
    sub = df[df["PosGroup"] == pos].copy()
    if len(sub) < MIN_N:
        return None

    X = sub[DRILLS]
    y = sub["DraftValue"]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_imp, y)
    importances = dict(zip(DRILLS, [round(float(i), 4) for i in rf.feature_importances_]))

    correlations = {}
    for drill in DRILLS:
        valid = sub[[drill, "DraftValue"]].dropna()
        if len(valid) < 20:
            correlations[drill] = None
            continue
        rho, p = spearmanr(valid[drill], valid["DraftValue"])
        # Negative rho = better drill value -> lower (better) draft slot, i.e. drill DOES matter directionally.
        correlations[drill] = {"spearman_rho": round(float(rho), 3), "p_value": round(float(p), 4),
                                 "n": int(len(valid))}

    ranked = sorted(importances.items(), key=lambda x: -x[1])
    return {
        "position": pos, "n": int(len(sub)),
        "rf_importance": importances,
        "rf_importance_ranked": ranked,
        "spearman_vs_draftvalue": correlations,
    }


def main():
    df = pd.read_csv("data/processed/combine_with_sai.csv")
    results = {}
    for pos in POSITIONS:
        res = analyze_position(df, pos)
        if res is None:
            continue
        results[pos] = res
        print(f"\n=== {pos} (n={res['n']}) — drill importance for draft outcome ===")
        for drill, imp in res["rf_importance_ranked"]:
            corr = res["spearman_vs_draftvalue"].get(drill)
            rho_str = f"rho={corr['spearman_rho']:+.3f} (p={corr['p_value']:.3f})" if corr else "rho=n/a"
            print(f"  {drill:16s} importance={imp:.3f}   {rho_str}")

    with open("outputs/position_profiles/drill_importance.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
