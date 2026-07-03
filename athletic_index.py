"""
athletic_index.py — Scouting Athletic Index (SAI): a 0-100, position-relative
athletic composite, built from first principles (not a clone of RAS).

Design:
  1. Four sub-scores, each 0-100, position-normalized (percentile vs same PosGroup
     + same era-ish weight class where relevant):
       - Speed Score      : 40-yd dash, weight-adjusted (Bill Barnwell-style
                             speed score formula, position-normalized after)
       - Explosiveness     : vertical jump + broad jump, weight-adjusted
       - Agility Score     : 3-cone + 20-yd shuttle (change of direction)
       - Power/Size Score  : bench press (weight-adjusted reps) + BMI-based size
  2. Each sub-score is computed on a physical formula first (so it's not "just a
     z-score"), then converted to a 0-100 percentile WITHIN the player's position
     group (so a 4.5s 40 doesn't unfairly crush a 330lb tackle vs a 180lb corner).
  3. Final SAI = weighted blend of the four sub-scores, with position-specific
     weights (a corner's SAI should weight speed/agility more than an OL's).
"""
import pandas as pd
import numpy as np

# Position-specific weighting of the four sub-scores: [speed, explosiveness, agility, power_size]
POSITION_WEIGHTS = {
    "QB":   {"speed": 0.30, "explosive": 0.25, "agility": 0.20, "power": 0.25},
    "RB":   {"speed": 0.35, "explosive": 0.30, "agility": 0.20, "power": 0.15},
    "WR":   {"speed": 0.40, "explosive": 0.30, "agility": 0.20, "power": 0.10},
    "TE":   {"speed": 0.30, "explosive": 0.25, "agility": 0.20, "power": 0.25},
    "OL":   {"speed": 0.15, "explosive": 0.20, "agility": 0.25, "power": 0.40},
    "EDGE": {"speed": 0.30, "explosive": 0.25, "agility": 0.20, "power": 0.25},
    "DL":   {"speed": 0.20, "explosive": 0.20, "agility": 0.20, "power": 0.40},
    "LB":   {"speed": 0.30, "explosive": 0.25, "agility": 0.25, "power": 0.20},
    "CB":   {"speed": 0.35, "explosive": 0.25, "agility": 0.30, "power": 0.10},
    "S":    {"speed": 0.35, "explosive": 0.25, "agility": 0.25, "power": 0.15},
}
DEFAULT_WEIGHTS = {"speed": 0.30, "explosive": 0.25, "agility": 0.25, "power": 0.20}


def speed_score_raw(weight, forty):
    """Barnwell-style speed score: weight * 200 / forty^4. Higher = faster for size."""
    if pd.isna(weight) or pd.isna(forty) or forty <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)


def explosiveness_raw(weight, vertical, broad):
    """Weight-adjusted jump explosiveness: (vertical(in) + broad(in)) scaled by bodyweight
    via a burst-style formula (jump work approx proportional to mass * height)."""
    if pd.isna(vertical) or pd.isna(broad) or pd.isna(weight):
        return np.nan
    return ((vertical + broad) * weight) / 1000.0


def agility_raw(cone, shuttle):
    """Lower is better; combine into a single change-of-direction time. Invert so
    higher = better agility."""
    if pd.isna(cone) or pd.isna(shuttle):
        return np.nan
    total = cone + shuttle
    return 1.0 / total  # higher = more agile


def power_size_raw(weight, bench, height):
    """Weight-adjusted power/size: bench reps normalized by bodyweight, blended
    with a BMI-style size term."""
    bmi = (weight / (height ** 2)) * 703 if not (pd.isna(weight) or pd.isna(height)) else np.nan
    if pd.isna(bench):
        # fall back to size only if bench is missing (very common — bench has ~40% missingness)
        return bmi
    bench_component = (bench * 10.0) / weight if not pd.isna(weight) and weight > 0 else np.nan
    if pd.isna(bench_component):
        return bmi
    if pd.isna(bmi):
        return bench_component
    return 0.6 * bench_component + 0.4 * (bmi / 10.0)


def to_percentile_within_group(series: pd.Series, groups: pd.Series) -> pd.Series:
    """Convert raw values to 0-100 percentile rank within each position group."""
    return series.groupby(groups).rank(pct=True) * 100


def compute_sai(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["_speed_raw"] = df.apply(lambda r: speed_score_raw(r["Weight"], r["40-yd Dash"]), axis=1)
    df["_explosive_raw"] = df.apply(lambda r: explosiveness_raw(r["Weight"], r["Vertical Jump"], r["Broad Jump"]), axis=1)
    df["_agility_raw"] = df.apply(lambda r: agility_raw(r["3-Cone Drill"], r["20-yd Shuttle"]), axis=1)
    df["_power_raw"] = df.apply(lambda r: power_size_raw(r["Weight"], r["Bench Press"], r["Height"]), axis=1)

    df["SpeedScore"] = to_percentile_within_group(df["_speed_raw"], df["PosGroup"])
    df["ExplosiveScore"] = to_percentile_within_group(df["_explosive_raw"], df["PosGroup"])
    df["AgilityScore"] = to_percentile_within_group(df["_agility_raw"], df["PosGroup"])
    df["PowerSizeScore"] = to_percentile_within_group(df["_power_raw"], df["PosGroup"])

    def blend(row):
        w = POSITION_WEIGHTS.get(row["PosGroup"], DEFAULT_WEIGHTS)
        parts, weights = [], []
        mapping = {"speed": row["SpeedScore"], "explosive": row["ExplosiveScore"],
                   "agility": row["AgilityScore"], "power": row["PowerSizeScore"]}
        for k, v in mapping.items():
            if not pd.isna(v):
                parts.append(v * w[k])
                weights.append(w[k])
        if not weights or sum(weights) == 0:
            return np.nan
        return sum(parts) / sum(weights)

    df["SAI"] = df.apply(blend, axis=1).round(1)

    df = df.drop(columns=["_speed_raw", "_explosive_raw", "_agility_raw", "_power_raw"])
    return df


def main():
    df = pd.read_csv("data/processed/combine_clean.csv")
    df = compute_sai(df)
    df.to_csv("data/processed/combine_with_sai.csv", index=False)

    print("SAI distribution overall:")
    print(df["SAI"].describe())
    print("\nTop 15 SAI all-time:")
    top = df.sort_values("SAI", ascending=False)[["Player", "PosGroup", "Year", "SAI"]].head(15)
    print(top.to_string(index=False))
    print("\nMean SAI by position:")
    print(df.groupby("PosGroup")["SAI"].mean().round(1).sort_values(ascending=False))


if __name__ == "__main__":
    main()
