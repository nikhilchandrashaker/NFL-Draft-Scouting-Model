"""
draft_steals.py — Compare predicted draft value (from combine measurables alone)
vs actual draft outcome to find "steals" (fell further than measurables/model
suggested, i.e. drafted LATER... wait, framed the way scouts think about it:

We define it from the team's perspective:
  - STEAL  = player was drafted LATER (worse pick number) than the model expected
             AND went on to be draftable/rostered — i.e. teams got a player who
             "graded out" athletically like an earlier pick, for a cheaper price.
             Concretely: ActualPick >> PredictedPick (model expected him gone sooner).
  - REACH  = player was drafted EARLIER (better pick number) than the model
             expected based on measurables alone — team paid a premium pick for
             someone whose workout numbers didn't scream "early-round".

Important caveat printed in the report: this model only sees combine numbers.
It has NO access to game tape, production, or intangibles — so "steal" here
specifically means "outperformed what a workout-numbers-only model expected",
not "was actually a good NFL player". Tom Brady is the canonical example: elite
production/intangibles, unremarkable workout, and he still fell to Round 6 --
but a workout-only model would NOT have predicted Round 6 for him from his
measurables alone, since his measurables were simply below-average, not
predictive of an all-time great OR a 6th rounder specifically. We surface him
separately as a "measurables told us nothing" case study rather than claim the
model "found" him.
"""
import pandas as pd
import numpy as np

def compute_steals(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["is_drafted"] == 1].copy()
    # Positive delta = actual pick number higher (later/worse) than predicted =>
    # team got him later than his measurables suggested => STEAL from value perspective
    d["ValueDelta"] = d["DraftValue"] - d["PredictedDraftValue"]
    return d


def main():
    df = pd.read_csv("data/processed/combine_with_predictions.csv")
    d = compute_steals(df)

    print("=== Top 20 STEALS (fell much later than measurables suggested) ===")
    steals = d.sort_values("ValueDelta", ascending=False).head(20)
    print(steals[["Player", "PosGroup", "Year", "Pick", "PredictedDraftValue", "ValueDelta"]]
          .round(1).to_string(index=False))

    print("\n=== Top 20 REACHES (drafted much earlier than measurables suggested) ===")
    reaches = d.sort_values("ValueDelta", ascending=True).head(20)
    print(reaches[["Player", "PosGroup", "Year", "Pick", "PredictedDraftValue", "ValueDelta"]]
          .round(1).to_string(index=False))

    brady = df[df["Player"].str.contains("Tom Brady", case=False, na=False)]
    if not brady.empty:
        r = brady.iloc[0]
        print(f"\nCase study — Tom Brady: actual pick={r['Pick']:.0f} (Round {r['Round']:.0f}), "
              f"model-predicted pick={r['PredictedDraftValue']:.1f} "
              f"({r['PredictedRound']}), delta={r['DraftValue']-r['PredictedDraftValue']:.1f}")

    d.to_csv("data/processed/draft_steals.csv", index=False)


if __name__ == "__main__":
    main()
