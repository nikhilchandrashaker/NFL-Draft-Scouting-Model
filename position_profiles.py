"""
position_profiles.py — Build position-specific athletic benchmark profiles.

For each position group: average, elite (90th pct, or 10th for time-based drills
where lower=better), poor (10th/90th), distributions, and an "archetype" summary.
"""
import pandas as pd
import numpy as np
import json

METRICS = ["40-yd Dash", "Vertical Jump", "Bench Press", "Broad Jump", "3-Cone Drill", "20-yd Shuttle", "Height", "Weight"]

# Lower is better for these (time-based)
LOWER_BETTER = {"40-yd Dash", "3-Cone Drill", "20-yd Shuttle"}

POSITIONS = ["QB", "RB", "WR", "TE", "OL", "EDGE", "LB", "CB", "S", "DL"]


def elite_pct(series: pd.Series, metric: str) -> float:
    p = series.dropna()
    if len(p) == 0:
        return np.nan
    return p.quantile(0.10 if metric in LOWER_BETTER else 0.90)


def poor_pct(series: pd.Series, metric: str) -> float:
    p = series.dropna()
    if len(p) == 0:
        return np.nan
    return p.quantile(0.90 if metric in LOWER_BETTER else 0.10)


def build_profile(df: pd.DataFrame, pos: str) -> dict:
    sub = df[df["PosGroup"] == pos]
    profile = {"position": pos, "n_players": int(len(sub)), "n_drafted": int(sub["is_drafted"].sum())}
    metrics = {}
    for m in METRICS:
        s = sub[m].dropna()
        if len(s) == 0:
            continue
        metrics[m] = {
            "n": int(len(s)),
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "median": round(float(s.median()), 3),
            "elite_pctile_value": round(float(elite_pct(sub[m], m)), 3),
            "poor_pctile_value": round(float(poor_pct(sub[m], m)), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
            "lower_is_better": m in LOWER_BETTER,
        }
    profile["metrics"] = metrics
    return profile


def build_all_profiles(df: pd.DataFrame) -> dict:
    return {pos: build_profile(df, pos) for pos in POSITIONS}


def archetype_label(row: pd.Series, profile: dict) -> str:
    """Very lightweight archetype tagging based on z-scores vs the position mean."""
    tags = []
    m = profile["metrics"]
    def z(metric):
        if metric not in m or pd.isna(row.get(metric, np.nan)):
            return None
        info = m[metric]
        if info["std"] == 0:
            return 0
        zval = (row[metric] - info["mean"]) / info["std"]
        if info["lower_is_better"]:
            zval = -zval
        return zval

    z40 = z("40-yd Dash")
    zvert = z("Vertical Jump")
    zbroad = z("Broad Jump")
    zbench = z("Bench Press")
    zweight = z("Weight")
    zcone = z("3-Cone Drill")
    zshuttle = z("20-yd Shuttle")

    if z40 is not None and z40 > 1.0:
        tags.append("Elite Speed")
    if zvert is not None and zbroad is not None and zvert > 1.0 and zbroad > 1.0:
        tags.append("Explosive")
    if zbench is not None and zbench > 1.0:
        tags.append("Powerful")
    if zcone is not None and zshuttle is not None and zcone > 1.0 and zshuttle > 1.0:
        tags.append("Elite Agility")
    if zweight is not None and zweight > 1.0:
        tags.append("Big-Bodied")
    if zweight is not None and zweight < -1.0:
        tags.append("Lean/Undersized")
    if not tags:
        tags.append("Balanced/Average")
    return ", ".join(tags)


def main():
    df = pd.read_csv("data/processed/combine_clean.csv")
    profiles = build_all_profiles(df)
    with open("outputs/position_profiles/profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    for pos in POSITIONS:
        p = profiles[pos]
        print(f"\n=== {pos} (n={p['n_players']}, drafted={p['n_drafted']}) ===")
        for metric, info in p["metrics"].items():
            print(f"  {metric:16s} mean={info['mean']:>7} elite={info['elite_pctile_value']:>7} poor={info['poor_pctile_value']:>7} (n={info['n']})")


if __name__ == "__main__":
    main()
