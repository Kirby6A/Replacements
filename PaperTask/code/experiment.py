"""
PaperTask baseline — re-implementation of <REPLACE: paper title / company>.

THE CONTRACT (what InternAgent requires of this file — keep all three):
  1. Tunables are argparse arguments with the paper's values as `default=`
     (the coding agent improves the code by editing these defaults / the logic).
  2. Running the script executes the full method and computes the metrics.
  3. It writes final_info.json — keys MUST match prompt.json's "metrics" —
     to the parent directory of wherever this script lives (run_X convention).

Everything between the ==== markers is yours to replace. The scaffold
(argparse + main + save) mirrors tasks/AutoDebug and should stay as-is.
"""
import os
import json
import argparse


# ==== 1. YOUR DATA ===========================================================
# Load the public dataset, or generate a synthetic proxy (seeded!) so the
# task is self-contained and runs in seconds. See tasks/AutoDebug for a
# synthetic example.
def load_data(seed: int = 42):
    raise NotImplementedError("REPLACE: return whatever your method consumes")
# =============================================================================


# ==== 2. THE PAPER'S METHOD (the thing to be improved) =======================
# Implement it faithfully. Expose the paper's key hyperparameters as arguments
# so they surface in run_experiment()'s argparse below.
def run_method(data, alpha: float):
    raise NotImplementedError("REPLACE: run the baseline method, return outputs")
# =============================================================================


# ==== 3. THE SCOREBOARD ======================================================
# Compute the metrics named in prompt.json. Keys must match exactly.
def evaluate(outputs, data) -> dict:
    raise NotImplementedError(
        'REPLACE: return e.g. {"recall_at_10": 0.61, "latency_ms": 320.0}'
    )
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    # REPLACE: one argument per tunable, paper's value as the default
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_data(seed=args.seed)
    outputs = run_method(data, alpha=args.alpha)
    metrics = evaluate(outputs, data)
    metrics["config"] = vars(args)

    # Save to final_info.json in the PARENT directory of this script
    # (AutoDebug convention — the runner expects it there per run folder).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(os.path.dirname(script_dir), "final_info.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {output_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
