"""
run_pipeline.py — Run the entire NFL Draft Scouting Model pipeline end-to-end,
in dependency order. Run from the repo root: `PYTHONPATH=src python3 src/run_pipeline.py`
"""
import time
import data_prep
import position_profiles
import athletic_index
import draft_model
import outliers
import feature_importance
import draft_steals
import clustering
import scouting_report
import build_figures
import export_dashboard_data
import build_report

STEPS = [
    ("Data prep & cleaning", data_prep.main),
    ("Position athletic profiles (Part 1)", position_profiles.main),
    ("Scouting Athletic Index (Part 3)", athletic_index.main),
    ("Draft prediction model (Part 2)", draft_model.main),
    ("Outlier detection (Part 5)", outliers.main),
    ("Drill importance by position (Part 6)", feature_importance.main),
    ("Draft steals & reaches (Part 7)", draft_steals.main),
    ("Position clustering (Part 8)", clustering.main),
    ("Scouting report generator (Part 10)", scouting_report.main),
    ("Report figures", lambda: [
        build_figures.fig_forty_by_position(), build_figures.fig_drill_importance_heatmap(),
        build_figures.fig_model_comparison(), build_figures.fig_sai_distribution(),
        build_figures.fig_wr_cluster_scatter(), build_figures.fig_steals_reaches(),
    ]),
    ("Dashboard data export (Part 9)", export_dashboard_data.main),
    ("Final PDF report", build_report.build),
]

if __name__ == "__main__":
    t0 = time.time()
    for label, fn in STEPS:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        fn()
    print(f"\nPipeline complete in {time.time()-t0:.1f}s")
    print("Dashboard: dashboard/index.html  |  Report: report.pdf")
