"""Aggregate VoD knowledge-condition results into a 2-cell summary.

Run from the moralsim repo root after a sweep has populated `results/`:

    python notebooks/analyze_vod.py

Outputs:
- prints the 2-row Wilson-CI table (private vs common) to stdout
- prints the chi-square test of independence
- writes long-format per-decision data to `results/vod_decisions.csv`
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint


EXPERIMENTS = {
    # experiment folder name -> knowledge condition
    "vd_base_private_cot": "private",
    "vd_base_common_cot": "common",
}


def load_rows(results_dir: Path) -> pd.DataFrame:
    """Walk results/<experiment>/dummy-*/log_env.json and return one row per LLM decision."""
    rows = []
    for exp_name, knowledge in EXPERIMENTS.items():
        exp_dir = results_dir / exp_name
        if not exp_dir.exists():
            continue
        for run_dir in exp_dir.glob("dummy-*"):
            log_path = run_dir / "log_env.json"
            if not log_path.exists():
                continue
            try:
                entries = json.load(open(log_path))
            except json.JSONDecodeError:
                continue
            for entry in entries:
                # entry["agent_id"] is [persona_id, "agent" | "dummy"]
                agent_type = entry["agent_id"][1]
                if agent_type != "agent":
                    continue
                chosen = entry.get("chosen_action")
                # Sanity-filter: action must be 1 (Volunteer) or 2 (Shirk).
                if chosen not in (1, 1.0, 2, 2.0):
                    continue
                rows.append(
                    {
                        "experiment": exp_name,
                        "knowledge": knowledge,
                        "run_id": run_dir.name,
                        "agent": entry["agent_id"][0],
                        "chose_volunteer": int(chosen == 1 or chosen == 1.0),
                        "payoff": entry.get("round_payoff"),
                    }
                )
    return pd.DataFrame(rows)


def wilson_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition: count, proportion volunteering, Wilson 95% CI."""
    grouped = df.groupby("knowledge")["chose_volunteer"].agg(["sum", "count"])
    lows, highs = proportion_confint(
        grouped["sum"], grouped["count"], alpha=0.05, method="wilson"
    )
    grouped["prop"] = grouped["sum"] / grouped["count"]
    grouped["ci_low"] = lows
    grouped["ci_high"] = highs
    return grouped[["sum", "count", "prop", "ci_low", "ci_high"]]


def chi2_knowledge(df: pd.DataFrame) -> dict:
    """Chi-square test of independence between knowledge condition and choice."""
    table = pd.crosstab(df["knowledge"], df["chose_volunteer"])
    chi2, p, dof, expected = chi2_contingency(table)
    return {"chi2": chi2, "dof": dof, "p_value": p, "contingency": table}


def main() -> None:
    results_dir = Path("results/llama-3.3-70b")
    df = load_rows(results_dir)

    if df.empty:
        print("No log_env.json files found under results/. Run the sweep first.")
        return

    print("=== runs found ===")
    print(
        df.groupby(["experiment"])["run_id"].nunique().rename("n_runs").to_frame()
    )
    print()

    print("=== per-condition proportions (Wilson 95% CI) ===")
    print(wilson_table(df).round(3))
    print()

    print("=== knowledge x choice contingency ===")
    result = chi2_knowledge(df)
    print(result["contingency"])
    print(
        f"chi2={result['chi2']:.3f}, dof={result['dof']}, p={result['p_value']:.4f}"
    )

    out_csv = results_dir / "vod_decisions.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
