"""
comparables.py — Find closest historical athletic comparables for a given player,
using position-normalized combine metrics.

Offers three distance/similarity methods as requested:
  - Euclidean distance (on standardized features)
  - Cosine similarity (on standardized features)
  - Mahalanobis distance (accounts for correlation between drills, computed
    within the player's position group's covariance structure)

Missing drills are mean-imputed within position group before distance computation,
since raw NaNs can't be compared.
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = ["Height", "Weight", "40-yd Dash", "Vertical Jump", "Bench Press",
             "Broad Jump", "3-Cone Drill", "20-yd Shuttle"]


def _prep_group(df: pd.DataFrame, pos_group: str):
    sub = df[df["PosGroup"] == pos_group].copy()
    sub[FEATURES] = sub.groupby("PosGroup")[FEATURES].transform(lambda s: s.fillna(s.mean()))
    # any remaining all-NaN column (shouldn't happen) -> fill with global mean
    sub[FEATURES] = sub[FEATURES].fillna(df[FEATURES].mean())
    return sub


def find_comparables(df: pd.DataFrame, player_name: str, method: str = "euclidean", k: int = 5):
    matches = df[df["Player"].str.lower() == player_name.lower()]
    if matches.empty:
        # fuzzy contains fallback
        matches = df[df["Player"].str.lower().str.contains(player_name.lower(), na=False)]
    if matches.empty:
        return None, f"No player found matching '{player_name}'"

    target_row = matches.iloc[0]
    pos_group = target_row["PosGroup"]
    sub = _prep_group(df, pos_group)
    sub = sub.reset_index(drop=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(sub[FEATURES])

    target_idx_matches = sub.index[(sub["Player"] == target_row["Player"]) & (sub["Year"] == target_row["Year"])]
    if len(target_idx_matches) == 0:
        return None, "Player found but missing from prepared position group (unexpected)."
    target_idx = target_idx_matches[0]
    target_vec = X[target_idx]

    if method == "euclidean":
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
        dist, idx = nn.kneighbors([target_vec])
        dist, idx = dist[0][1:], idx[0][1:]
        sims = 1 / (1 + dist)  # convert distance to a 0-1-ish similarity
    elif method == "cosine":
        sims_all = cosine_similarity([target_vec], X)[0]
        order = np.argsort(-sims_all)
        order = [i for i in order if i != target_idx][:k]
        idx = np.array(order)
        sims = sims_all[idx]
    elif method == "mahalanobis":
        cov = np.cov(X, rowvar=False)
        # regularize in case of singular covariance (small groups / collinear drills)
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        dists = np.array([mahalanobis(target_vec, X[i], inv_cov) for i in range(len(X))])
        order = np.argsort(dists)
        order = [i for i in order if i != target_idx][:k]
        idx = np.array(order)
        d = dists[idx]
        sims = 1 / (1 + d)
    else:
        raise ValueError(f"Unknown method {method}")

    results = sub.iloc[idx][["Player", "PosGroup", "School", "Year", "Round", "Pick", "SAI"]].copy()
    results["similarity_pct"] = (sims / sims.max() * 100).round(1) if sims.max() > 0 else 0
    results = results.sort_values("similarity_pct", ascending=False)
    return {
        "target": {"Player": target_row["Player"], "PosGroup": pos_group, "Year": int(target_row["Year"]),
                    "School": target_row["School"]},
        "method": method,
        "comparables": results.to_dict(orient="records"),
    }, None


def main():
    df = pd.read_csv("data/processed/combine_with_sai.csv")

    test_players = ["Travis Hunter", "Tom Brady", "Patrick Mahomes", "Saquon Barkley", "Calvin Johnson"]
    for p in test_players:
        for method in ["euclidean", "cosine", "mahalanobis"]:
            result, err = find_comparables(df, p, method=method, k=4)
            if err:
                print(f"{p} [{method}]: {err}")
                continue
            print(f"\n{p} ({result['target']['PosGroup']}, {result['target']['Year']}) — {method}:")
            for c in result["comparables"]:
                print(f"   {c['Player']:22s} {c['similarity_pct']:5.1f}%  (SAI={c['SAI']})  {c['Year']}")


if __name__ == "__main__":
    main()
