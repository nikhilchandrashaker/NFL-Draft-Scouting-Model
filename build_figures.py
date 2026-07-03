"""
build_figures.py — Generate the summary chart images used in report.pdf and
notebooks (position benchmark bars, drill importance heatmap, model comparison,
SAI distribution, clustering scatter).
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#444", "axes.labelcolor": "#222"})
TURF = "#3C6E47"
AMBER = "#D98E2B"
INK = "#222"

def fig_forty_by_position():
    df = pd.read_csv("data/processed/combine_with_sai.csv")
    order = df.groupby("PosGroup")["40-yd Dash"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df[df.PosGroup == p]["40-yd Dash"].dropna() for p in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(TURF); patch.set_alpha(0.6)
    ax.set_ylabel("40-yd Dash (seconds)")
    ax.set_title("40-Yard Dash Distribution by Position")
    plt.tight_layout()
    plt.savefig("outputs/figures/forty_by_position.png", dpi=140)
    plt.close(fig)

def fig_drill_importance_heatmap():
    with open("outputs/position_profiles/drill_importance.json") as f:
        di = json.load(f)
    positions = list(di.keys())
    drills = ["40-yd Dash", "Weight", "Height", "Vertical Jump", "Broad Jump", "3-Cone Drill", "20-yd Shuttle", "Bench Press"]
    mat = np.array([[di[p]["rf_importance"].get(d, 0) for d in drills] for p in positions])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(mat, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(drills))); ax.set_xticklabels(drills, rotation=40, ha="right")
    ax.set_yticks(range(len(positions))); ax.set_yticklabels(positions)
    for i in range(len(positions)):
        for j in range(len(drills)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if mat[i, j] > mat.max()*0.6 else "black")
    ax.set_title("RF Feature Importance for Draft Value, by Position")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Importance")
    plt.tight_layout()
    plt.savefig("outputs/figures/drill_importance_heatmap.png", dpi=140)
    plt.close(fig)

def fig_model_comparison():
    with open("models/model_results.json") as f:
        res = json.load(f)["results_by_model"]
    models = list(res.keys())
    mae = [res[m]["MAE"] for m in models]
    r2 = [res[m]["R2"] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].bar(models, mae, color=TURF)
    axes[0].set_title("MAE by Model (picks)"); axes[0].set_ylabel("MAE"); axes[0].tick_params(axis='x', rotation=20)
    axes[1].bar(models, r2, color=AMBER)
    axes[1].set_title("R-squared by Model"); axes[1].set_ylabel("R2"); axes[1].tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/figures/model_comparison.png", dpi=140)
    plt.close(fig)

def fig_sai_distribution():
    df = pd.read_csv("data/processed/combine_with_sai.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(df["SAI"].dropna(), bins=40, color=TURF, alpha=0.8)
    ax.set_xlabel("Scouting Athletic Index (SAI)"); ax.set_ylabel("Count")
    ax.set_title("SAI Distribution — All Prospects (2000-2026)")
    plt.tight_layout()
    plt.savefig("outputs/figures/sai_distribution.png", dpi=140)
    plt.close(fig)

def fig_wr_cluster_scatter():
    with open("outputs/position_profiles/clusters.json") as f:
        clusters = json.load(f)
    info = clusters["WR"]
    assignments = info["player_assignments"]
    names = sorted(set(a["ClusterName"] for a in assignments))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    color_map = dict(zip(names, colors))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name in names:
        pts = [(a["PCA1"], a["PCA2"]) for a in assignments if a["ClusterName"] == name]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, label=name, alpha=0.6, s=18, color=color_map[name])
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    ax.set_title("WR Athletic Archetypes (K-Means + PCA)")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig("outputs/figures/wr_clusters.png", dpi=140)
    plt.close(fig)

def fig_steals_reaches():
    df = pd.read_csv("data/processed/draft_steals.csv")
    top = df.sort_values("ValueDelta", ascending=False).head(10)
    bot = df.sort_values("ValueDelta", ascending=True).head(10)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].barh(top["Player"][::-1], top["ValueDelta"][::-1], color="#2F8F8A")
    axes[0].set_title("Top 10 Steals (fell later than expected)"); axes[0].set_xlabel("Pick slots later than model expected")
    axes[1].barh(bot["Player"][::-1], bot["ValueDelta"][::-1], color="#B23A2E")
    axes[1].set_title("Top 10 Reaches (drafted earlier than expected)"); axes[1].set_xlabel("Pick slots earlier than model expected")
    plt.tight_layout()
    plt.savefig("outputs/figures/steals_reaches.png", dpi=140)
    plt.close(fig)

if __name__ == "__main__":
    fig_forty_by_position()
    fig_drill_importance_heatmap()
    fig_model_comparison()
    fig_sai_distribution()
    fig_wr_cluster_scatter()
    fig_steals_reaches()
    print("Figures written to outputs/figures/")
