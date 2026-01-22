import argparse
import csv
import logging
import os
import re
import shutil
from typing import List, Dict, Any
from huggingface_hub import list_repo_refs
import lm_eval

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run MMLU evaluation on model checkpoints.")
parser.add_argument("--repo_id", type=str, required=True, help="Target HuggingFace repository ID")
parser.add_argument("--step_interval", type=int, default=1000, help="Steps between evaluated checkpoints")
parser.add_argument("--tasks", type=str, default="mmlu", help="Comma-separated list of tasks")
parser.add_argument("--batch_size", type=str, default="auto", help="Batch size for eval")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
args = parser.parse_args()

CLEAN_CACHE = True
RESULTS_CSV = "lmeval_%s.csv" % args.repo_id.split("/")[-1]


def get_target_branches(repo_id: str, interval: int) -> List[Dict[str, Any]]:
    """Returns sorted list of branches matching step intervals."""
    logger.warning(f"Fetching branches from {repo_id}...")
    try:
        refs = list_repo_refs(repo_id)
    except (OSError, ValueError) as e:
        logger.error(f"Error listing refs: {e}")
        return []

    branches = []
    pattern = re.compile(r"(?:.*)?step(\d+)(?:.*)?$")

    for b in refs.branches:
        match = pattern.match(b.name)
        if match:
            step = int(match.group(1))
            if step % interval == 0:
                branches.append({"step": step, "name": b.name})

    sorted_branches = sorted(branches, key=lambda x: x["step"], reverse=True)
    logger.warning(f"Identified {len(sorted_branches)} target branches.")
    return sorted_branches


def get_processed_steps(csv_path: str) -> set:
    """Check which steps have already been processed to resume runs."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, "r", encoding="utf-8") as f:
        return {int(row["step"]) for row in csv.DictReader(f) if row.get("step")}

def extract_metrics(results_dict: dict) -> dict:
    """
    Flattens lm_eval results into a single dictionary mapping Task -> Score.
    Prioritizes 'acc,none', then 'acc', then 'acc_norm'.
    """
    flat_metrics = {}
    for task_name, metrics in results_dict.items():
        # Handle different metric names depending on the task
        # MMLU typically uses 'acc' or 'acc,none'
        score = (metrics.get("acc,none")
                 or metrics.get("acc")
                 or metrics.get("acc_norm,none")
                 or metrics.get("acc_norm")
                 or 0.0)
        flat_metrics[task_name] = score
    return flat_metrics

def main():
    branches = get_target_branches(args.repo_id, args.step_interval)
    logger.warning("Found target branches: %s" , branches)
    processed_steps = get_processed_steps(RESULTS_CSV)
    logger.warning("Already processed branches: %s", processed_steps)
    task_input = args.tasks.split(",")

    # We do NOT write the header yet. We wait for the first result to determine
    # the full list of decomposed subtasks (e.g. mmlu_abstract_algebra, etc).

    for b in branches:
        logger.warning("Processing branch %s", b)
        step, branch_name = b["step"], b["name"]
        if step in processed_steps:
            logger.info(f"Skipping step {step} (already processed)"); continue

        logger.info(f"Evaluating Step {step} ({branch_name})")
        step_cache_dir = os.path.abspath(f"./tmp_cache_step_{step}")
        os.makedirs(step_cache_dir, exist_ok=True)

        try:
            model_args = f"pretrained={args.repo_id},revision={branch_name},trust_remote_code=True,cache_dir={step_cache_dir}"

            # Run LM Eval
            # Note: When 'mmlu' is passed, lm_eval runs all subtasks.
            results = lm_eval.simple_evaluate(model="hf", model_args=model_args, tasks=task_input,
                                              num_fewshot=5, batch_size=args.batch_size, device=args.device, log_samples=False)

            # --- Process Results ---
            # 'results["results"]' contains keys for the group AND all subtasks
            res_dict = results["results"]
            metric_row = extract_metrics(res_dict)

            # Prepare the full row data
            row_data = {"step": step, "branch": branch_name}
            row_data.update(metric_row)

            # --- CSV Handling ---
            file_exists = os.path.exists(RESULTS_CSV)

            # Determine fieldnames (columns)
            # If file exists, use existing columns. If new, generate from this first run.
            if file_exists:
                with open(RESULTS_CSV, "r", encoding="utf-8") as f:
                    fieldnames = csv.DictReader(f).fieldnames
            else:
                # Create columns: step, branch, then sorted task names
                task_cols = sorted(metric_row.keys())
                fieldnames = ["step", "branch"] + task_cols

            # Write to CSV
            with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)

            # Calculate average for logging purposes (simple mean of all columns found)
            avg_score = sum(metric_row.values()) / len(metric_row) if metric_row else 0
            logger.info(f"Completed step {step}. Approx Avg: {avg_score:.4f}")

        except Exception as e:
            logger.error(f"Failed to evaluate step {step}: {e}")
            # Optional: Log the full traceback if needed
            # import traceback; traceback.print_exc()
        finally:
            if CLEAN_CACHE:
                shutil.rmtree(step_cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()