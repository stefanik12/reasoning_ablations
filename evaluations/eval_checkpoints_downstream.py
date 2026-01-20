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
parser.add_argument("--output_file", type=str, default="mmlu_results.csv", help="Output CSV file")
args = parser.parse_args()

CLEAN_CACHE = True


def get_target_branches(repo_id: str, interval: int) -> List[Dict[str, Any]]:
    """Returns sorted list of branches matching step intervals."""
    logger.warning(f"Fetching branches from {repo_id}...")
    try:
        refs = list_repo_refs(repo_id)
    except (OSError, ValueError) as e:
        logger.error(f"Error listing refs: {e}")
        return []

    branches = []
    pattern = re.compile(r"^step(\d+)(?:.*)?$")

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
    if not os.path.exists(csv_path): return set()
    with open(csv_path, "r", encoding="utf-8") as f:
        return {int(row["step"]) for row in csv.DictReader(f) if row.get("step")}


def main():
    branches = get_target_branches(args.repo_id, args.step_interval)
    processed = get_processed_steps(args.output_file)
    task_list = args.tasks.split(",")

    # Initialize CSV header if file doesn't exist
    if not os.path.exists(args.output_file):
        with open(args.output_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "branch", "average_acc"] + [f"{t}_acc" for t in task_list])

    for b in branches:
        step, branch_name = b["step"], b["name"]
        if step in processed:
            logger.info(f"Skipping step {step} (already processed)");
            continue

        # Create a temporary cache directory for this specific step to ensure isolation
        logger.info(f"Evaluating Step {step} ({branch_name})")
        step_cache = os.path.abspath(f"./tmp_cache_step_{step}")
        os.makedirs(step_cache, exist_ok=True)

        try:
            # Construct model_args with specific revision and cache dir.
            # trust_remote_code=True is required for OLMo/Apertus.
            model_args = f"pretrained={args.repo_id},revision={branch_name},trust_remote_code=True,cache_dir={step_cache}"
            # Run LM Eval
            results = lm_eval.simple_evaluate(model="hf", model_args=model_args, tasks=task_list,
                                              num_fewshot=5, batch_size=args.batch_size, device=args.device,
                                              log_samples=False)
            # --- Parse Results ---
            res_dict = results["results"]
            row = {"step": step, "branch": branch_name, "average_acc": 0.0}
            total_acc, count = 0.0, 0

            # Extract scores for requested tasks
            for t in task_list:
                # MMLU usually reports 'acc' or 'acc,none'. Try 'acc,none' first then 'acc'
                score = res_dict.get(t, {}).get("acc,none") or res_dict.get(t, {}).get("acc") or 0.0
                row[f"{t}_acc"] = score;
                total_acc += score;
                count += 1
            if count > 0: row["average_acc"] = total_acc / count

            # --- Write to CSV ---
            with open(args.output_file, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f,
                               fieldnames=["step", "branch", "average_acc"] + [f"{t}_acc" for t in task_list]).writerow(
                    row)
            logger.info(f"Completed step {step}. Avg Acc: {row['average_acc']:.4f}")

        except Exception as e:
            logger.error(f"Failed step {step}: {e}")
        finally:
            # Cleanup Cache
            if CLEAN_CACHE: shutil.rmtree(step_cache, ignore_errors=True)


if __name__ == "__main__":
    main()