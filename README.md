# NFL Draft Value Model: Which Combine Metrics Actually Matter?

A full analytics-department-style project that turns 26 years of NFL Combine data (2000–2026, 8,968
prospects) into a scouting toolkit: position-specific athletic benchmarks, a draft-value prediction
model, an original athletic composite score, historical player comparables, outlier/"freak athlete"
detection, per-position drill-importance analysis, draft steals/reaches, and athletic archetype
clustering — plus an interactive dashboard and an automated scouting-report generator.

Built end-to-end in a sandboxed environment with **no network access** — see "Substitutions" below for
what that changed.

## Repository structure

```
NFL-Draft-Scouting-Model/
├── README.md
├── requirements.txt
├── report.pdf                     ← Part 9-page executive summary of all 10 analyses
├── resume_bullets.md
├── data/
│   ├── raw/NFL_Combine_Since_2000.csv
│   └── processed/                 ← cleaned + enriched datasets at each pipeline stage
├── src/                           ← the actual analysis pipeline (run in this order)
│   ├── data_prep.py               (1) clean data, map positions into 9 scouting groups
│   ├── position_profiles.py       (2) Part 1 — elite/avg/poor benchmarks per position
│   ├── athletic_index.py          (3) Part 3 — Scouting Athletic Index (SAI)
│   ├── draft_model.py             (4) Part 2 — draft value prediction model
│   ├── comparables.py             (5) Part 4 — historical comparables engine
│   ├── outliers.py                (6) Part 5 — Isolation Forest / LOF outlier detection
│   ├── feature_importance.py      (7) Part 6 — which drills matter, by position
│   ├── draft_steals.py            (8) Part 7 — steals vs. reaches
│   ├── clustering.py              (9) Part 8 — K-Means athletic archetypes
│   ├── scouting_report.py         (10) Part 10 — auto scouting report generator
│   ├── build_figures.py           report chart generation
│   ├── build_report.py            assembles report.pdf
│   └── export_dashboard_data.py   bundles everything for the dashboard
├── notebooks/                     3 notebooks walking through the same pipeline
├── models/                        trained model (best_model.pkl) + metrics (model_results.json)
├── outputs/
│   ├── position_profiles/         profiles.json, clusters.json, drill_importance.json
│   ├── scouting_reports/          sample player reports + radar chart PNGs
│   └── figures/                   charts embedded in report.pdf
└── dashboard/                     Part 9 — Part 9 dashboard, two forms:
    ├── standalone.html            single self-contained file (data + JS inlined) — open this one directly
    ├── index.html                 same dashboard, split into index.html + app.js + data.js for editing
    ├── app.js
    └── data.js
```

**Open `dashboard/standalone.html`** if you just want to use it — it's one file with everything inlined,
so it works by double-clicking it locally, and also works if previewed inside a sandboxed viewer that
can't fetch sibling files (some chat/artifact previews only serve a single HTML file, which silently
breaks the split `index.html` + `data.js` + `app.js` version — every button, including search, would
just do nothing since `DASHBOARD_DATA` never loads). The split version in `index.html` is kept as the
more maintainable source for further development; regenerate `standalone.html` from it after edits with:
```bash
python3 -c "
html = open('dashboard/index.html').read()
data_js = open('dashboard/data.js').read()
app_js = open('dashboard/app.js').read()
html = html.replace('<script src=\"data.js\"></script>\n<script src=\"app.js\"></script>',
                     f'<script>\n{data_js}\n</script>\n<script>\n{app_js}\n</script>')
open('dashboard/standalone.html','w').write(html)
"
```

## Running it

```bash
cd NFL-Draft-Scouting-Model
pip install -r requirements.txt
export PYTHONPATH=src

python3 src/data_prep.py               # writes data/processed/combine_clean.csv
python3 src/position_profiles.py       # writes outputs/position_profiles/profiles.json
python3 src/athletic_index.py          # writes data/processed/combine_with_sai.csv
python3 src/draft_model.py             # trains models, writes models/best_model.pkl
python3 src/outliers.py                # writes data/processed/combine_with_outliers.csv
python3 src/feature_importance.py      # writes outputs/position_profiles/drill_importance.json
python3 src/draft_steals.py            # writes data/processed/draft_steals.csv
python3 src/clustering.py              # writes outputs/position_profiles/clusters.json
python3 src/scouting_report.py         # generates sample player reports + radar charts
python3 src/build_figures.py           # generates report.pdf's chart images
python3 src/export_dashboard_data.py   # bundles everything into dashboard/data.js
python3 src/build_report.py            # assembles report.pdf
```

Or just open `dashboard/index.html` directly in a browser — it's fully self-contained (reads
`dashboard/data.js`, no server needed) and already built from the current data.

To pull a scouting report for any player from the command line:
```python
import sys; sys.path.insert(0, "src")
import pandas as pd, json
from scouting_report import build_report, print_report
df = pd.read_csv("data/processed/combine_with_predictions.csv")
profiles = json.load(open("outputs/position_profiles/profiles.json"))
print_report(build_report(df, profiles, "Saquon Barkley"))
```

## Key design decisions

**Position groups.** Raw combine positions (OT, OG, C, DE, OLB, ...) are mapped into 10 scouting
groups: the 9 requested (QB, RB, WR, TE, OL, EDGE, LB, CB, S) plus **DL** (interior DT/NT), which was
kept separate from EDGE since interior and edge rushers have very different athletic profiles — folding
them together would have made both benchmarks worse. Travis Hunter's `CB/WR` dual-position listing is
mapped to CB (primary defensive role in the historical record).

**Missing data is real, not imputed away for display.** Bench press is missing for ~40% of prospects
(many skip it), 3-cone/shuttle for ~42–43%. Every module that reports a player's raw measurables
preserves `NaN` for drills they didn't run — a player who DNP'd everything (e.g. Travis Hunter's actual
combine, since he did not test) shows as *not tested*, not as a fabricated average. Imputation is only
ever used *internally* for model fitting (Isolation Forest, K-Means, the draft model's pipeline), never
written back over real values in what gets displayed or saved to the processed CSVs.

**Draft value target.** Since "draft position" needs to be a single continuous, ordered target,
undrafted players are slotted at `(max pick in their draft class + 50)` rather than dropped — the model
needs to learn "this profile tends to go undrafted," and dropping those rows would remove that signal
entirely.

**SAI vs. RAS.** The Scouting Athletic Index intentionally isn't a percentile-of-a-percentile clone of
RAS. Each of its four sub-scores (Speed, Explosiveness, Agility, Power/Size) starts from a physics-style
formula (e.g. speed score ∝ weight × 200 / 40-time⁴) *before* being converted to a within-position
percentile, and the final blend uses position-specific weights (a corner's SAI weights speed/agility
more heavily; an offensive lineman's weights power/size more) rather than one fixed formula for every
position.

## Substitutions made for the offline sandbox

This was built with `bash_tool` network access **disabled**, so a few requested tools couldn't be
installed. Each substitute keeps the same interface, so swapping back in is a one-line change locally:

| Requested | Used instead | Why |
|---|---|---|
| XGBoost, LightGBM | `sklearn.ensemble.{GradientBoostingRegressor, HistGradientBoostingRegressor}` | Same gradient-boosted-tree family; `pip install xgboost lightgbm` failed with no matching distribution (no network) |
| SHAP | `sklearn.inspection.permutation_importance` | Also not installable; permutation importance works for *any* fitted model (including HistGB, which has no native `feature_importances_`) |
| Streamlit / Dash | A static HTML/JS dashboard (`dashboard/index.html`) | Not installable; the static dashboard needs no server and runs the same nearest-neighbor / percentile logic client-side in JavaScript |
| UMAP | PCA | Not installable; PCA is used for the 2D cluster visualization instead |

## The headline finding

**Combine measurables alone explain only ~20% of where a player gets drafted** (R² ≈ 0.21 on held-out
data). 40-yard dash and weight dominate feature importance almost everywhere; bench press is
consistently the weakest predictor at every position. That's not a modeling failure — it's a
quantification of how much a draft decision comes from tape, production, and scheme fit rather than the
workout. See `report.pdf` for the full write-up, including the Tom Brady case study and the
regression-to-the-mean caveat on the "steals/reaches" analysis.
