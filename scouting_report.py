"""
scouting_report.py — Given a player name, auto-generate a scouting report:
athletic profile vs position benchmarks, strengths/weaknesses, closest
historical comparables, model-expected draft range, SAI grade, and a radar
chart image.

Combines outputs from position_profiles, comparables, draft_model, and
athletic_index — this is the "capstone" module that ties Parts 1-7 together
into the Part 10 deliverable.
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comparables import find_comparables

RADAR_METRICS = ["40-yd Dash", "Vertical Jump", "Broad Jump", "3-Cone Drill", "20-yd Shuttle", "Bench Press"]
LOWER_BETTER = {"40-yd Dash", "3-Cone Drill", "20-yd Shuttle"}


def _pctile_within_group(df, pos, metric, value):
    """Return the player's 'goodness' percentile within their position group.
    Higher = better, regardless of whether the raw metric is lower-is-better
    (40-yd dash, 3-cone, shuttle) or higher-is-better (vertical, broad, bench)."""
    if pd.isna(value):
        return None
    sub = df[df["PosGroup"] == pos][metric].dropna()
    if len(sub) == 0:
        return None
    if metric in LOWER_BETTER:
        # you're "better" than everyone whose time is slower (higher) than yours
        pct = (sub > value).mean() * 100
    else:
        pct = (sub < value).mean() * 100
    return round(float(pct), 1)


def grade_from_sai(sai):
    if pd.isna(sai):
        return "N/A"
    if sai >= 95: return "A+ (Elite Freak Athlete)"
    if sai >= 88: return "A (Elite)"
    if sai >= 75: return "B+ (Above Average)"
    if sai >= 55: return "B (Solid)"
    if sai >= 35: return "C (Below Average)"
    if sai >= 15: return "D (Poor)"
    return "F (Well Below Average)"


def build_report(df: pd.DataFrame, profiles: dict, player_name: str) -> dict:
    matches = df[df["Player"].str.lower() == player_name.lower()]
    if matches.empty:
        matches = df[df["Player"].str.lower().str.contains(player_name.lower(), na=False)]
    if matches.empty:
        return None
    row = matches.iloc[0]
    pos = row["PosGroup"]
    profile = profiles.get(pos, {}).get("metrics", {})

    strengths, weaknesses = [], []
    metric_pctiles = {}
    for m in RADAR_METRICS + ["Height", "Weight"]:
        val = row.get(m, np.nan)
        pct = _pctile_within_group(df, pos, m, val)
        metric_pctiles[m] = {"value": None if pd.isna(val) else round(float(val), 2), "position_percentile": pct}
        if pct is not None:
            if pct >= 80:
                strengths.append(f"{m} ({pct:.0f}th percentile at {pos})")
            elif pct <= 20:
                weaknesses.append(f"{m} ({pct:.0f}th percentile at {pos})")

    comps, _ = find_comparables(df, player_name, method="euclidean", k=5)

    report = {
        "player": row["Player"], "position": pos, "school": row.get("School"),
        "year": int(row["Year"]) if not pd.isna(row["Year"]) else None,
        "SAI": row.get("SAI"),
        "athletic_grade": grade_from_sai(row.get("SAI", np.nan)),
        "actual_draft": {"round": None if pd.isna(row.get("Round")) else int(row["Round"]),
                           "pick": None if pd.isna(row.get("Pick")) else int(row["Pick"]),
                           "drafted": bool(row.get("is_drafted", 0))},
        "model_expected_round": row.get("PredictedRound"),
        "model_expected_pick": None if pd.isna(row.get("PredictedDraftValue", np.nan)) else round(float(row["PredictedDraftValue"]), 1),
        "metrics": metric_pctiles,
        "strengths": strengths if strengths else ["No standout combine metrics (average across the board)"],
        "weaknesses": weaknesses if weaknesses else ["No significant combine weaknesses"],
        "closest_comparables": comps["comparables"] if comps else [],
    }
    return report


def render_radar(df, profiles, player_name, report, outpath):
    pos = report["position"]
    profile_metrics = profiles.get(pos, {}).get("metrics", {})
    metrics = [m for m in RADAR_METRICS if m in profile_metrics]
    if len(metrics) < 3:
        return None

    labels = metrics
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    def pct_series(source_getter):
        vals = []
        for m in labels:
            v = source_getter(m)
            vals.append(v if v is not None else 0)
        vals += vals[:1]
        return vals

    player_pcts = pct_series(lambda m: report["metrics"][m]["position_percentile"])
    avg_pcts = [50] * (n + 1)  # average is by definition 50th percentile
    elite_pcts = [90] * (n + 1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, player_pcts, linewidth=2, color="#C8102E", label=report["player"])
    ax.fill(angles, player_pcts, color="#C8102E", alpha=0.25)
    ax.plot(angles, elite_pcts, linewidth=1, linestyle="--", color="#1a1a1a", label="Elite (90th pct)")
    ax.plot(angles, avg_pcts, linewidth=1, linestyle=":", color="#888888", label="Position Average")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_title(f"{report['player']} ({pos}) — Athletic Profile\nSAI: {report['SAI']}  |  Grade: {report['athletic_grade']}", fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return outpath


def print_report(report: dict):
    print(f"\n{'='*60}\nSCOUTING REPORT: {report['player']} ({report['position']}, {report['school']}, {report['year']})\n{'='*60}")
    print(f"Scouting Athletic Index (SAI): {report['SAI']}  ->  {report['athletic_grade']}")
    ad = report["actual_draft"]
    if ad["drafted"]:
        print(f"Actual Draft: Round {ad['round']}, Pick {ad['pick']}")
    else:
        print("Actual Draft: Undrafted")
    print(f"Model-Expected Draft Range: ~pick {report['model_expected_pick']} ({report['model_expected_round']})")
    print("\nStrengths:")
    for s in report["strengths"]:
        print(f"  + {s}")
    print("Weaknesses:")
    for w in report["weaknesses"]:
        print(f"  - {w}")
    print("\nClosest Historical Comparables:")
    for c in report["closest_comparables"]:
        print(f"  {c['Player']:22s} {c['similarity_pct']:5.1f}%  (SAI={c['SAI']}, {c['Year']})")


def main():
    df = pd.read_csv("data/processed/combine_with_predictions.csv")
    with open("outputs/position_profiles/profiles.json") as f:
        profiles = json.load(f)

    sample_players = ["Travis Hunter", "Anthony Richardson", "Saquon Barkley", "Tom Brady", "Micah Parsons"]
    all_reports = {}
    for p in sample_players:
        report = build_report(df, profiles, p)
        if report is None:
            print(f"\n{p}: not found")
            continue
        print_report(report)
        all_reports[report["player"]] = report
        safe_name = report["player"].replace(" ", "_").replace(".", "")
        render_radar(df, profiles, p, report, f"outputs/scouting_reports/{safe_name}_radar.png")

    with open("outputs/scouting_reports/sample_reports.json", "w") as f:
        json.dump(all_reports, f, indent=2, default=str)


if __name__ == "__main__":
    main()
