"""Compare a fresh evaluation with a saved baseline. This script compares the latest evaluation run with a saved
baseline and informs if the average hit rate, MRR and faithfulness have dropped or not.

To run: python regression_check.py --save-baseline (first-time, saves baseline)
        python regression_check.py                 (every time after code change to check regression)
"""

import json
import sys

BASELINE_FILE = "evaluation_baseline.json"
RESULTS_FILE = "evaluation_results.json"
REGRESSION_TOLERANCE = 0.01

def main():
    if "--save-baseline" in sys.argv:
        results = json.load(open(RESULTS_FILE, 'r', encoding="utf-8"))
        with open(BASELINE_FILE, 'w', encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved current evaluation_results.json as the new baseline ({BASELINE_FILE})")
        print(f"  hit_rate = {results['summary']['hit_rate']:.2f} mrr = {results['summary']['mrr']:.2f} "
                f"faithfulness = {results['summary']['faithfulness']:.2f}")

    try:
        baseline = json.load(open(BASELINE_FILE, 'r', encoding="utf-8"))
    except FileNotFoundError:
        print(f"No baseline found. Please run \"python run_eval.py\" and then \"python regression_check.py --save-baseline\" first.")
        sys.exit(1)

    current = json.load(open(RESULTS_FILE, 'r', encoding="utf-8"))

    print(f"{'Metric':<15} {'Baseline':<10} {'Current':<10} {'Change':<10}")
    regressed = False

    for metric in ("hit_rate", "mrr", "faithfulness"):
        b_val = baseline["summary"][metric]
        c_val = current["summary"][metric]
        change = c_val - b_val
        flag = ""
        if change < -REGRESSION_TOLERANCE:
            flag = "  <--REGRESSION"
            regressed = True
        print(f"{metric:<15} {b_val:<10.2f} {c_val:<10.2f} {change:+.2f} {flag}")

    if regressed:
        print("\nREGRESSION DETECTED")
        sys.exit(1)
    else:
        print("\nNO REGRESSION")
        
if __name__ == "__main__":
    main()
    
    