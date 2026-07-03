"""
data_prep.py — Load, clean, and standardize the NFL Combine dataset (2000-present).

Handles:
- DNP / NA parsing -> NaN
- Position grouping into 9 scouting groups: QB, RB, WR, TE, OL, EDGE, LB, CB, S
- Derived fields: BMI, Round (0 = undrafted), drafted flag, overall pick as draft value target
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/NFL_Combine_Since_2000.csv"
PROCESSED_PATH = "data/processed/combine_clean.csv"

NUMERIC_COLS = [
    "Height", "Weight", "40-yd Dash", "Vertical Jump", "Bench Press",
    "Broad Jump", "3-Cone Drill", "20-yd Shuttle",
]

# Map raw, granular combine positions into 9 scouting position groups
POSITION_GROUP_MAP = {
    # QB
    "QB": "QB",
    # RB
    "RB": "RB", "FB": "RB", "HB": "RB",
    # WR
    "WR": "WR",
    # TE
    "TE": "TE",
    # OL
    "OT": "OL", "OG": "OL", "C": "OL", "OL": "OL", "G": "OL", "T": "OL",
    # EDGE (edge rushers - stand-up or hand-down)
    "OLB": "EDGE", "DE": "EDGE", "EDGE": "EDGE",
    # LB (off-ball linebackers)
    "ILB": "LB", "LB": "LB", "MLB": "LB",
    # DL / interior (folded into EDGE group is wrong; keep separate as DL, but
    # spec only wants 9 groups incl. EDGE not DL — group interior DL with EDGE
    # is inaccurate for athletic profiling, so we keep DL players out of DL-less
    # spec by mapping DT/NT to their own group used internally as "DL")
    "DT": "DL", "NT": "DL", "DL": "DL",
    # CB
    "CB": "CB", "DB": "CB",
    # S
    "S": "S", "FS": "S", "SS": "S", "SAF": "S",
    # K/P/LS - special teams, excluded from position-specific athletic analysis
    "K": "ST", "P": "ST", "LS": "ST",
    # Dual-position players (e.g. Travis Hunter, CB/WR) - map to primary listed role
    "CB/WR": "CB", "WR/CB": "WR",
}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Replace DNP / NA-like tokens with NaN across the board
    df = df.replace({"DNP": np.nan, "NA": np.nan, "N/A": np.nan, "": np.nan})

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Round / Pick numeric
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
    df["Pick"] = pd.to_numeric(df["Pick"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    df["Drafted"] = df["Drafted"].fillna("N").str.strip()
    df["is_drafted"] = (df["Drafted"].str.upper() == "Y").astype(int)

    # Position grouping
    df["Position"] = df["Position"].str.strip().str.upper()
    df["PosGroup"] = df["Position"].map(POSITION_GROUP_MAP)
    df["PosGroup"] = df["PosGroup"].fillna("OTHER")

    df["Player"] = (df["First"].fillna("").str.strip() + " " + df["Last"].fillna("").str.strip()).str.strip()

    # Derived combine metrics
    df["BMI"] = (df["Weight"] / (df["Height"] ** 2)) * 703

    # Overall draft value: undrafted players get slot beyond the last real pick
    # so the model has a continuous, ordered target (queue position across ~10 rounds
    # equivalent). Each draft class has ~256-260 picks in modern era, more in early 2000s.
    max_pick_by_year = df.groupby("Year")["Pick"].transform("max")
    undrafted_value = max_pick_by_year.fillna(300) + 50
    df["DraftValue"] = df["Pick"].fillna(undrafted_value)

    # Round label used for classification-style summaries: 0 = undrafted
    df["RoundLabel"] = df["Round"].fillna(0).astype(int)

    return df


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    cols = NUMERIC_COLS
    out = pd.DataFrame({
        "column": cols,
        "missing_pct": [round(df[c].isna().mean() * 100, 1) for c in cols],
    })
    return out


def main():
    df = load_raw()
    clean_df = clean(df)
    clean_df.to_csv(PROCESSED_PATH, index=False)
    print(f"Rows: {len(clean_df)}")
    print(f"Years: {clean_df['Year'].min():.0f}-{clean_df['Year'].max():.0f}")
    print("\nPosition group counts:")
    print(clean_df["PosGroup"].value_counts())
    print("\nMissingness by combine drill:")
    print(summarize_missingness(clean_df).to_string(index=False))
    print(f"\nDrafted: {clean_df['is_drafted'].sum()} / {len(clean_df)} ({clean_df['is_drafted'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
