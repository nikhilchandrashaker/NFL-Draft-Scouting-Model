"""
build_report.py — Assemble report.pdf: the executive analytics-department-style
summary of all 10 parts of the project, with embedded figures and player radar
charts generated earlier by build_figures.py and scouting_report.py.
"""
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                  TableStyle, PageBreak, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

TURF = colors.HexColor("#2C5233")
AMBER = colors.HexColor("#B9711F")
INK = colors.HexColor("#1A1A1A")
GREY = colors.HexColor("#666666")
LIGHT = colors.HexColor("#F2F0E8")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportTitle", fontSize=26, leading=30, textColor=INK, fontName="Helvetica-Bold", spaceAfter=6))
styles.add(ParagraphStyle("ReportSubtitle", fontSize=12, leading=16, textColor=GREY, fontName="Helvetica"))
styles.add(ParagraphStyle("SectionHead", fontSize=15, leading=18, textColor=TURF, fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("SubHead", fontSize=11.5, leading=14, textColor=AMBER, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("BodyBullet", fontSize=9.7, leading=14, textColor=INK, fontName="Helvetica", spaceAfter=4, leftIndent=10))
styles.add(ParagraphStyle("BodyText2", fontSize=9.7, leading=14, textColor=INK, fontName="Helvetica", spaceAfter=6))
styles.add(ParagraphStyle("Caption", fontSize=8, leading=10, textColor=GREY, fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle("Caveat", fontSize=9, leading=13, textColor=colors.HexColor("#8A4B14"), fontName="Helvetica-Oblique", spaceBefore=6, spaceAfter=6, borderPadding=6))

def para(text, style="BodyText2"):
    return Paragraph(text, styles[style])

def hr():
    return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#CCCCCC"), spaceBefore=4, spaceAfter=10)

def build():
    with open("models/model_results.json") as f:
        model_results = json.load(f)
    with open("outputs/position_profiles/profiles.json") as f:
        profiles = json.load(f)
    df = pd.read_csv("data/processed/combine_with_sai.csv")

    doc = SimpleDocTemplate("report.pdf", pagesize=letter,
                              topMargin=0.65*inch, bottomMargin=0.65*inch,
                              leftMargin=0.7*inch, rightMargin=0.7*inch)
    story = []

    # ---------- Cover ----------
    story.append(Spacer(1, 60))
    story.append(para("NFL DRAFT VALUE MODEL", "ReportTitle"))
    story.append(para("Which Combine Metrics Actually Matter?", "ReportSubtitle"))
    story.append(Spacer(1, 4))
    story.append(hr())
    story.append(para(f"Combine dataset: 2000&ndash;2026 &middot; {len(df):,} prospects across 10 position groups", "BodyText2"))
    story.append(para("Scope: position-specific athletic benchmarks, a draft-value prediction model, a custom "
                        "Scouting Athletic Index (SAI), historical player comparables, outlier/freak-athlete detection, "
                        "drill-importance-by-position analysis, draft steals/reaches, and position clustering into "
                        "athletic archetypes.", "BodyText2"))
    story.append(Spacer(1, 10))
    story.append(para("<b>Headline finding:</b> combine measurables alone explain only about "
                        f"{model_results['results_by_model'][model_results['best_model']]['R2']*100:.0f}% of the variance "
                        "in where a player is drafted (R&#178; &asymp; 0.21). Workout numbers matter, but tape, production, "
                        "and intangibles matter more &mdash; which is itself a finding, not a limitation to hide.", "Caveat"))
    story.append(PageBreak())

    # ---------- Part 1 ----------
    story.append(para("Part 1 &mdash; Position-Specific Athletic Profiles", "SectionHead"))
    story.append(para("Instead of one blended combine average, every position gets its own elite / average / poor "
                        "benchmark, built from position-specific percentiles (90th/10th, direction-aware so lower-is-better "
                        "drills like the 40 are handled correctly).", "BodyText2"))
    sample_rows = [["Position", "Elite 40yd", "Avg 40yd", "Elite Vert", "Elite Broad", "Elite Bench"]]
    for pos in ["QB", "RB", "WR", "TE", "OL", "EDGE", "CB", "S"]:
        m = profiles[pos]["metrics"]
        sample_rows.append([
            pos,
            f"{m.get('40-yd Dash', {}).get('elite_pctile_value', '—')}",
            f"{m.get('40-yd Dash', {}).get('mean', '—')}",
            f"{m.get('Vertical Jump', {}).get('elite_pctile_value', '—')}\"",
            f"{m.get('Broad Jump', {}).get('elite_pctile_value', '—')}\"",
            f"{m.get('Bench Press', {}).get('elite_pctile_value', '—')}",
        ])
    t = Table(sample_rows, hAlign="LEFT", colWidths=[0.7*inch]+[1.1*inch]*5)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), TURF), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8.3),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")), ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Image("outputs/figures/forty_by_position.png", width=6.2*inch, height=3.55*inch))
    story.append(para("40-yard dash distribution by position &mdash; skill positions (WR/CB/S/RB) cluster fast, "
                        "line positions (OL/DL) run much slower, exactly as scouting intuition predicts.", "Caption"))
    story.append(PageBreak())

    # ---------- Part 2 ----------
    story.append(para("Part 2 &mdash; Draft Prediction Model", "SectionHead"))
    story.append(para("Random Forest, Gradient Boosting, and HistGradientBoosting (scikit-learn's gradient-boosted-tree "
                        "family, used in place of XGBoost/LightGBM since this sandbox has no network access to install them "
                        "&mdash; drop-in swap locally with the same fit/predict API) were trained to predict draft value "
                        "(overall pick number, with undrafted players slotted beyond the last real pick of their class).", "BodyText2"))
    res_rows = [["Model", "MAE (picks)", "RMSE", "R\u00b2"]]
    for m, v in model_results["results_by_model"].items():
        res_rows.append([m, str(v["MAE"]), str(v["RMSE"]), str(v["R2"])])
    t2 = Table(res_rows, hAlign="LEFT", colWidths=[2.2*inch, 1.3*inch, 1.1*inch, 1.0*inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), TURF), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))
    story.append(Image("outputs/figures/model_comparison.png", width=6.2*inch, height=2.6*inch))
    story.append(para(f"Best model by MAE: <b>{model_results['best_model']}</b>, off by about "
                        f"{model_results['results_by_model'][model_results['best_model']]['MAE']:.0f} picks on average. "
                        "Global permutation importance: <b>40-yard dash and weight dominate</b>, with position, agility "
                        "drills, and bench press contributing much less.", "BodyText2"))
    story.append(PageBreak())

    # ---------- Part 3 ----------
    story.append(para("Part 3 &mdash; Scouting Athletic Index (SAI)", "SectionHead"))
    story.append(para("SAI is an original 0&ndash;100 composite (not a clone of RAS): four physics-grounded sub-scores "
                        "&mdash; <b>Speed Score</b> (weight-adjusted 40 time), <b>Explosiveness</b> (weight-adjusted "
                        "vertical+broad), <b>Agility</b> (inverse cone+shuttle), and <b>Power/Size</b> (weight-adjusted "
                        "bench, falling back to a BMI-style size term when bench is missing, since bench has ~40% "
                        "missingness in this dataset) &mdash; each converted to a within-position percentile, then blended "
                        "with position-specific weights (a corner's SAI weights speed/agility more; an offensive lineman's "
                        "weights power/size more).", "BodyText2"))
    story.append(Image("outputs/figures/sai_distribution.png", width=6.0*inch, height=3.1*inch))
    story.append(para("SAI is centered near 50 by construction (percentile-based), with a long tail of true freak "
                        "athletes above 90.", "Caption"))
    story.append(PageBreak())

    # ---------- Part 4 ----------
    story.append(para("Part 4 &mdash; Historical Comparables", "SectionHead"))
    story.append(para("Given a player, the model finds the closest historical athletes at the same position using "
                        "three interchangeable distance metrics: Euclidean, cosine similarity, and Mahalanobis distance "
                        "(which accounts for correlation between drills). Example outputs from the engine:", "BodyText2"))
    for label, rows in [
        ("Travis Hunter (CB, 2025) &mdash; Euclidean", ["Mansoor Delane (2026)", "Derek Stingley (2022)", "Benjamin Morrison (2025)", "Ashton Youboty (2006)"]),
        ("Calvin Johnson (WR, 2007) &mdash; Euclidean", ["Michael Jenkins (2004)", "Vincent Jackson (2005)", "Michael Floyd (2012)", "Savion Williams (2025)"]),
    ]:
        story.append(para(label, "SubHead"))
        story.append(para(" &middot; ".join(rows), "BodyBullet"))
    story.append(PageBreak())

    # ---------- Part 5 ----------
    story.append(para("Part 5 &mdash; Athletic Outlier Detection", "SectionHead"))
    story.append(para("Isolation Forest and Local Outlier Factor are fit separately within each position group (an "
                        "outlier only means something relative to positional peers), then converted to within-position "
                        "percentiles for fair cross-position comparison. 'Freak athletes' = unusual AND elite (SAI &ge; 90).", "BodyText2"))
    d2 = pd.read_csv("data/processed/combine_with_outliers.csv")
    freaks = d2[(d2["iso_pctile"] >= 85) & (d2["SAI"] >= 90)].sort_values("SAI", ascending=False).head(10)
    rows = [["Player", "Pos", "Year", "SAI", "Unusual %ile"]]
    for _, r in freaks.iterrows():
        rows.append([r["Player"], r["PosGroup"], str(int(r["Year"])), f"{r['SAI']:.1f}", f"{r['iso_pctile']:.0f}%"])
    t3 = Table(rows, hAlign="LEFT", colWidths=[1.8*inch, 0.7*inch, 0.7*inch, 0.7*inch, 1.0*inch])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), TURF), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
    ]))
    story.append(t3)
    story.append(PageBreak())

    # ---------- Part 6 ----------
    story.append(para("Part 6 &mdash; Which Combine Drills Actually Matter?", "SectionHead"))
    story.append(para("A separate Random Forest is fit per position (not one model for everyone), then feature "
                        "importances are extracted for that position only. This is the most actionable finding in the "
                        "project:", "BodyText2"))
    story.append(Image("outputs/figures/drill_importance_heatmap.png", width=6.3*inch, height=4.3*inch))
    story.append(para("<b>40-yard dash dominates for nearly every position</b> &mdash; even OL and DL, where it acts as "
                        "a proxy for overall explosiveness/athleticism rather than straight-line speed being the skill "
                        "itself. <b>Bench press is consistently the weakest predictor</b> across the board &mdash; "
                        "matching the common scouting critique that bench is more a measure of gym strength than "
                        "football-relevant power. <b>Weight matters heavily for OL/DL/TE.</b>", "BodyText2"))
    story.append(PageBreak())

    # ---------- Part 7 ----------
    story.append(para("Part 7 &mdash; Draft Steals &amp; Reaches", "SectionHead"))
    story.append(para("<b>Steal</b> = fell later than a measurables-only model expected. <b>Reach</b> = drafted "
                        "earlier than measurables alone would predict.", "BodyText2"))
    story.append(para("Since the model only explains ~20% of draft-position variance, predictions regress hard toward "
                        "the mean &mdash; so top-10 picks will systematically read as 'reaches' and Day 3 picks as "
                        "'steals.' This says more about what workout numbers can't capture (tape, production, scheme "
                        "fit) than about actual scouting quality. Tom Brady is the canonical illustration: his "
                        "measurables were simply below-average across the board, not predictive of either an all-time "
                        "great or specifically a Round 6 pick &mdash; the model would not have 'found' him from his "
                        "workout alone.", "Caveat"))
    story.append(Image("outputs/figures/steals_reaches.png", width=6.4*inch, height=2.8*inch))
    story.append(PageBreak())

    # ---------- Part 8 ----------
    story.append(para("Part 8 &mdash; Position Clustering", "SectionHead"))
    story.append(para("K-Means (primary) plus Agglomerative clustering as a structural cross-check, with PCA for 2D "
                        "visualization (UMAP was unavailable in this offline sandbox; PCA is used instead and noted in "
                        "the README). Each position gets a curated number of clusters matching how many distinct "
                        "athletic roles scouts actually use for that position.", "BodyText2"))
    story.append(Image("outputs/figures/wr_clusters.png", width=5.6*inch, height=4.4*inch))
    story.append(para("WR archetypes: Speedsters, Possession, Big-body, Slot Receivers, Vertical Threats &mdash; named "
                        "automatically by ranking cluster centroids on speed and size.", "Caption"))
    story.append(PageBreak())

    # ---------- Part 9 & 10 ----------
    story.append(para("Part 9 &mdash; Interactive Dashboard", "SectionHead"))
    story.append(para("A self-contained static HTML dashboard (<font face='Courier'>dashboard/index.html</font>) covers "
                        "Player Search (auto-generated scouting reports), Position Explorer (benchmarks, drill "
                        "importance, archetypes), a client-side Draft Predictor, Steals &amp; Reaches, and Outliers. "
                        "It runs entirely in the browser (no server) against a precomputed data bundle, so nearest-"
                        "neighbor comparables and percentiles are computed live in JavaScript.", "BodyText2"))
    story.append(para("Part 10 &mdash; Scouting Report Generator", "SectionHead"))
    story.append(para("Given a player name, <font face='Courier'>src/scouting_report.py</font> auto-produces an "
                        "athletic profile, strengths/weaknesses (percentile vs. position peers), closest comparables, "
                        "model-expected draft range, an SAI letter grade, and a radar chart. Sample below.", "BodyText2"))
    story.append(Image("outputs/scouting_reports/Anthony_Richardson_radar.png", width=4.6*inch, height=4.0*inch))
    story.append(para("Anthony Richardson (QB, 2023) &mdash; SAI 98.2, Grade A+ (Elite Freak Athlete). Drafted 4th "
                        "overall; model-expected range was Round 2 from measurables alone, since his elite testing "
                        "wasn't matched by elite accuracy on tape &mdash; a case where the model and actual value "
                        "diverge in an informative way.", "Caption"))

    doc.build(story)
    print("Wrote report.pdf")

if __name__ == "__main__":
    build()
