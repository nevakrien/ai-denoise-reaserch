"""Export descriptive fits: .venv/bin/python scripts/analyze_trends.py."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lenslab.analysis import aggregate, load_runs, run_frame
from lenslab.trends import trend_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/depth-trends.csv")
    args = parser.parse_args()
    runs, errors = load_runs(args.results)
    for error in errors:
        print(error, file=sys.stderr)
    if not runs:
        parser.error("No valid runs")
    # Legacy smoke runs have no model label; keep their source names visible.
    for run in runs:
        if not run.get("experiment", {}).get("model_label") and not run.get("model_name"):
            run["model_name"] = Path(run["_path"]).stem
    frame = run_frame(runs).drop_duplicates(["run", "kept"])
    summary = trend_summary(aggregate(frame))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary[["model", "series", "scope", "metric", "linear_R2", "spearman_rho",
                   "isotonic_R2", "single_step_R2"]].to_string(index=False))
    print(f"\nSaved {output}. Fits are in-sample descriptions, not causal or significance tests.")


if __name__ == "__main__":
    main()
