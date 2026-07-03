"""
export_dashboard_data.py — Bundle everything the dashboard needs into a single
JS data file (players + precomputed comparables + profiles + clusters +
drill importance + outliers), so the dashboard runs as a static HTML file
with no backend/server required.
"""
import pandas as pd
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURES = ["Height", "Weight", "40-yd Dash", "Vertical Jump", "Bench Press",
             "Broad Jump", "3-Cone Drill", "20-yd Shuttle"]


def precompute_all_comparables(df: pd.DataFrame, k: int = 6) -> dict:
    comps = {}
    for pos, sub in df.groupby("PosGroup"):
        if len(sub) < k + 1:
            continue
        sub = sub.reset_index(drop=True)
        X = sub[FEATURES].copy()
        X = X.fillna(X.mean())
        Xs = StandardScaler().fit_transform(X)
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(sub)), metric="euclidean").fit(Xs)
        dist, idx = nn.kneighbors(Xs)
        for i in range(len(sub)):
            row = sub.iloc[i]
            key = f"{row['Player']}|{int(row['Year'])}"
            neighbors = []
            for d, j in zip(dist[i][1:], idx[i][1:]):
                nrow = sub.iloc[j]
                sim = round(float(1 / (1 + d)) * 100, 1)
                neighbors.append({
                    "player": nrow["Player"], "year": int(nrow["Year"]),
                    "school": nrow.get("School", ""), "sai": None if pd.isna(nrow.get("SAI")) else round(float(nrow["SAI"]), 1),
                    "similarity": sim,
                })
            # renormalize similarity so top neighbor isn't always exactly ~ same scale issue
            if neighbors:
                maxsim = max(n["similarity"] for n in neighbors)
                if maxsim > 0:
                    for n in neighbors:
                        n["similarity"] = round(n["similarity"] / maxsim * 100, 1)
            comps[key] = neighbors
    return comps


def main():
    df = pd.read_csv("data/processed/combine_with_outliers.csv")

    # Merge in cluster assignments
    with open("outputs/position_profiles/clusters.json") as f:
        clusters = json.load(f)
    cluster_lookup = {}
    for pos, info in clusters.items():
        for p in info["player_assignments"]:
            cluster_lookup[f"{p['Player']}|{p['Year']}"] = {"cluster": p["ClusterName"], "pca1": round(p["PCA1"], 2), "pca2": round(p["PCA2"], 2)}

    players = []
    for _, r in df.iterrows():
        key = f"{r['Player']}|{int(r['Year']) if not pd.isna(r['Year']) else 0}"
        c = cluster_lookup.get(key, {})
        players.append({
            "player": r["Player"], "pos": r["PosGroup"], "school": r.get("School", ""),
            "year": None if pd.isna(r["Year"]) else int(r["Year"]),
            "height": None if pd.isna(r["Height"]) else r["Height"],
            "weight": None if pd.isna(r["Weight"]) else r["Weight"],
            "forty": None if pd.isna(r["40-yd Dash"]) else r["40-yd Dash"],
            "vertical": None if pd.isna(r["Vertical Jump"]) else r["Vertical Jump"],
            "bench": None if pd.isna(r["Bench Press"]) else r["Bench Press"],
            "broad": None if pd.isna(r["Broad Jump"]) else r["Broad Jump"],
            "cone": None if pd.isna(r["3-Cone Drill"]) else r["3-Cone Drill"],
            "shuttle": None if pd.isna(r["20-yd Shuttle"]) else r["20-yd Shuttle"],
            "drafted": bool(r.get("is_drafted", 0)),
            "round": None if pd.isna(r.get("Round", np.nan)) else int(r["Round"]),
            "pick": None if pd.isna(r.get("Pick", np.nan)) else int(r["Pick"]),
            "team": r.get("Team") if isinstance(r.get("Team"), str) else None,
            "sai": None if pd.isna(r.get("SAI", np.nan)) else round(float(r["SAI"]), 1),
            "iso_pctile": None if pd.isna(r.get("iso_pctile", np.nan)) else round(float(r["iso_pctile"]), 1),
            "cluster": c.get("cluster"), "pca1": c.get("pca1"), "pca2": c.get("pca2"),
        })

    with open("outputs/position_profiles/profiles.json") as f:
        profiles = json.load(f)
    with open("outputs/position_profiles/drill_importance.json") as f:
        drill_importance = json.load(f)

    preds_df = pd.read_csv("data/processed/combine_with_predictions.csv")
    pred_lookup = {f"{r['Player']}|{int(r['Year'])}": {
        "predicted_pick": round(float(r["PredictedDraftValue"]), 1),
        "predicted_round": r["PredictedRound"],
        "value_delta": None if pd.isna(r.get("Pick", np.nan)) else round(float(r["DraftValue"] - r["PredictedDraftValue"]), 1),
    } for _, r in preds_df.iterrows()}
    for p in players:
        key = f"{p['player']}|{p['year']}"
        p["prediction"] = pred_lookup.get(key)

    with open("models/model_results.json") as f:
        model_results = json.load(f)

    # Trim clusters bundle for the dashboard: drop per-player assignments here since
    # they're already merged into players[] above (cluster/pca1/pca2 fields) -
    # keeps only the summary stats needed for the Position Explorer view.
    clusters_slim = {
        pos: {"position": info["position"], "k": info["k"],
              "cluster_summary": info["cluster_summary"],
              "pca_explained_variance": info["pca_explained_variance"]}
        for pos, info in clusters.items()
    }

    # Note: comparables are computed live in the dashboard's JS (nearest-neighbor
    # over the players array, position-scoped) rather than precomputed here -
    # keeps the shipped bundle small since ~9000 x 6 precomputed neighbor lists
    # would roughly double the payload for something trivial to compute client-side.
    bundle = {
        "players": players,
        "profiles": profiles,
        "clusters": clusters_slim,
        "drill_importance": drill_importance,
        "model_results": model_results,
    }

    with open("dashboard/data.js", "w") as f:
        f.write("const DASHBOARD_DATA = ")
        json.dump(bundle, f)
        f.write(";")

    import os
    size_mb = os.path.getsize("dashboard/data.js") / 1e6
    print(f"Wrote dashboard/data.js ({size_mb:.1f} MB, {len(players)} players)")


if __name__ == "__main__":
    main()
