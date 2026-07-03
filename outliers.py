"""
outliers.py — Athletic outlier detection.

Two complementary lenses, computed within each position group (since "outlier"
only means something relative to positional peers):
  - Isolation Forest: finds globally unusual body-type/measurable combinations
  - Local Outlier Factor (LOF): finds players unusual relative to their local
    neighborhood density (catches rare position profiles that IF can miss)

We report both "freak athlete" outliers (unusual AND elite, via SAI) and
"unusual body type" outliers (unusual regardless of whether it's good or bad).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

FEATURES = ["Height", "Weight", "40-yd Dash", "Vertical Jump", "Bench Press",
             "Broad Jump", "3-Cone Drill", "20-yd Shuttle"]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY with missing drills mean-imputed, for internal model fitting only.
    The original df's real NaNs must NOT be overwritten in what gets saved to disk -
    imputation here is a modeling convenience, not a claim that the player tested."""
    d = df.copy()
    d[FEATURES] = d.groupby("PosGroup")[FEATURES].transform(lambda s: s.fillna(s.mean()))
    d[FEATURES] = d[FEATURES].fillna(d[FEATURES].mean())
    return d


def detect_outliers(df: pd.DataFrame, min_group_size: int = 30) -> pd.DataFrame:
    imputed = _prep(df)
    out = df.copy()  # preserve real NaNs for anything downstream/display
    out["iso_score"] = np.nan
    out["lof_score"] = np.nan

    for pos, sub in imputed.groupby("PosGroup"):
        if len(sub) < min_group_size:
            continue
        X = StandardScaler().fit_transform(sub[FEATURES])

        iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
        iso.fit(X)
        # decision_function: lower = more anomalous. Flip sign so higher = more unusual.
        iso_scores = -iso.decision_function(X)

        n_neighbors = min(20, len(sub) - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
        lof.fit_predict(X)
        # negative_outlier_factor_: closer to -1 = normal, more negative = more anomalous
        lof_scores = -lof.negative_outlier_factor_

        out.loc[sub.index, "iso_score"] = iso_scores
        out.loc[sub.index, "lof_score"] = lof_scores

    # IF/LOF scores are only meaningful WITHIN a position group (each fit
    # separately). Convert to within-group percentiles for fair cross-group ranking.
    out["iso_pctile"] = out.groupby("PosGroup")["iso_score"].rank(pct=True) * 100
    out["lof_pctile"] = out.groupby("PosGroup")["lof_score"].rank(pct=True) * 100

    return out


def main():
    df = pd.read_csv("data/processed/combine_with_sai.csv")
    d = detect_outliers(df)
    d.to_csv("data/processed/combine_with_outliers.csv", index=False)

    print("=== Most unusual overall (Isolation Forest, within-position percentile, top 15) ===")
    top_iso = d.dropna(subset=["iso_pctile"]).sort_values("iso_pctile", ascending=False)
    print(top_iso[["Player", "PosGroup", "Year", "SAI", "iso_pctile"]].head(15).to_string(index=False))

    print("\n=== 'Freak athletes' (unusual AND elite: top-15% iso_pctile within position, SAI >= 90) ===")
    freaks = d[(d["iso_pctile"] >= 85) & (d["SAI"] >= 90)].sort_values("SAI", ascending=False)
    print(freaks[["Player", "PosGroup", "Year", "SAI", "iso_pctile"]].head(15).to_string(index=False))

    print("\n=== Worst athletes relative to position (bottom 15 SAI among drafted players with full combine) ===")
    full_combine = d.dropna(subset=FEATURES[2:])  # drills present (not just imputed)
    worst = full_combine[full_combine["is_drafted"] == 1].sort_values("SAI").head(15)
    print(worst[["Player", "PosGroup", "Year", "Round", "Pick", "SAI"]].to_string(index=False))

    print("\n=== Rare position profiles (LOF, top 15) ===")
    top_lof = d.dropna(subset=["lof_score"]).sort_values("lof_score", ascending=False)
    print(top_lof[["Player", "PosGroup", "Year", "SAI", "lof_score"]].head(15).to_string(index=False))

    for name in ["DK Metcalf", "Calvin Johnson", "Jordan Davis", "Anthony Richardson"]:
        row = d[d["Player"].str.contains(name, case=False, na=False)]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n{name}: SAI={r['SAI']}, iso_score={r['iso_score']:.3f}, lof_score={r['lof_score']:.3f}, "
                  f"iso_percentile={ (d['iso_score']<r['iso_score']).mean()*100:.1f}")


if __name__ == "__main__":
    main()
