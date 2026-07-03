"""
clustering.py — Cluster players within each position group into athletic
archetypes (e.g. WR -> Speedsters / Possession / Big-body / Vertical threats).

Methods: K-Means (primary, produces the named archetypes) + PCA for 2D
visualization + Agglomerative (hierarchical) clustering as a cross-check on
cluster count / structure. (UMAP unavailable offline - PCA used for the
dashboard scatter instead; noted in README.)
"""
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURES = ["Height", "Weight", "40-yd Dash", "Vertical Jump", "Bench Press",
             "Broad Jump", "3-Cone Drill", "20-yd Shuttle"]

# Curated k per position group (based on how many meaningfully distinct
# athletic roles exist within that position in real scouting)
K_BY_POSITION = {
    "QB": 3, "RB": 4, "WR": 5, "TE": 3, "OL": 3,
    "EDGE": 3, "LB": 3, "CB": 3, "S": 3, "DL": 3,
}

ARCHETYPE_NAMES = {
    "WR": ["Speedsters", "Possession WRs", "Big-body WRs", "Slot Receivers", "Vertical Threats"],
    "RB": ["Power Backs", "Receiving Backs", "Explosive Backs", "Balanced Backs"],
    "QB": ["Mobile QBs", "Pocket Passers", "Big-Armed/Big-Bodied QBs"],
    "TE": ["Blocking TEs", "Move/Athletic TEs", "Receiving TEs"],
    "OL": ["Athletic OL", "Power/Mass OL", "Balanced OL"],
    "EDGE": ["Speed Rushers", "Power Rushers", "Balanced EDGE"],
    "LB": ["Coverage LBs", "Downhill/Thumper LBs", "Balanced LBs"],
    "CB": ["Press/Man CBs", "Zone/Speed CBs", "Balanced CBs"],
    "S": ["Box Safeties", "Free/Range Safeties", "Balanced Safeties"],
    "DL": ["Penetrating DL", "Run-Stuffing DL", "Balanced DL"],
}


def _prep(sub: pd.DataFrame) -> np.ndarray:
    sub = sub.copy()
    sub[FEATURES] = sub[FEATURES].fillna(sub[FEATURES].mean())
    X = StandardScaler().fit_transform(sub[FEATURES])
    return X


def name_clusters(sub: pd.DataFrame, labels: np.ndarray, pos: str, k: int) -> dict:
    """Assign human-readable archetype names to numeric cluster ids by ranking
    cluster centroids on a couple of defining axes (speed vs size vs explosiveness)."""
    names_pool = ARCHETYPE_NAMES.get(pos, [f"Archetype {i+1}" for i in range(k)])
    tmp = sub.copy()
    tmp["_cluster"] = labels
    centroid_speed = tmp.groupby("_cluster")["40-yd Dash"].mean()  # lower = faster
    centroid_weight = tmp.groupby("_cluster")["Weight"].mean()
    # Rank clusters: fastest first, heaviest last, tie-broken by explosiveness
    order = centroid_speed.sort_values().index.tolist()
    name_map = {}
    for i, cid in enumerate(order):
        name_map[int(cid)] = names_pool[i] if i < len(names_pool) else f"Archetype {i+1}"
    return name_map


def cluster_position(df: pd.DataFrame, pos: str) -> dict:
    sub = df[df["PosGroup"] == pos].copy().reset_index(drop=True)
    if len(sub) < 40:
        return None
    k = K_BY_POSITION.get(pos, 3)
    X = _prep(sub)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)

    agg = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    name_map = name_clusters(sub, km_labels, pos, k)

    sub["Cluster"] = km_labels
    sub["ClusterName"] = sub["Cluster"].map(name_map)
    sub["AggCluster"] = agg_labels
    sub["PCA1"] = coords[:, 0]
    sub["PCA2"] = coords[:, 1]

    cluster_summary = {}
    for cid, name in name_map.items():
        members = sub[sub["Cluster"] == cid]
        cluster_summary[name] = {
            "n": int(len(members)),
            "mean_40": round(float(members["40-yd Dash"].mean()), 3) if members["40-yd Dash"].notna().any() else None,
            "mean_weight": round(float(members["Weight"].mean()), 1),
            "mean_SAI": round(float(members["SAI"].mean()), 1) if "SAI" in members else None,
            "example_players": members.sort_values("SAI", ascending=False)["Player"].head(5).tolist() if "SAI" in members else [],
        }

    pca_explained = pca.explained_variance_ratio_.tolist()
    return {
        "position": pos, "k": k,
        "cluster_summary": cluster_summary,
        "pca_explained_variance": [round(float(v), 3) for v in pca_explained],
        "player_assignments": sub[["Player", "Year", "Cluster", "ClusterName", "PCA1", "PCA2"]].to_dict(orient="records"),
    }


def main():
    df = pd.read_csv("data/processed/combine_with_sai.csv")
    all_results = {}
    for pos in K_BY_POSITION:
        res = cluster_position(df, pos)
        if res is None:
            continue
        all_results[pos] = res
        print(f"\n=== {pos} clusters (k={res['k']}) ===")
        for name, info in res["cluster_summary"].items():
            print(f"  {name:22s} n={info['n']:4d}  mean_40={info['mean_40']}  mean_wt={info['mean_weight']}  mean_SAI={info['mean_SAI']}")
            print(f"      e.g. {', '.join(info['example_players'])}")

    with open("outputs/position_profiles/clusters.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
