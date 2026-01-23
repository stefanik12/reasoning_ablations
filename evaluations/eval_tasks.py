import argparse
import csv
import logging
import os
import shutil
import lm_eval
from pathlib import Path

from evaluations.tools import get_target_branches, get_processed_steps, configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def _extract_metrics(results_dict: dict) -> dict:
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


def eval_tasks(repo_id,
               output_dir,
               step_interval,
               batch_size,
               tasks,
               only_final_model_eval,
               keep_cache):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = Path(output_dir) / f"lmeval_{repo_id.split('/')[-1]}.csv"

    branches = get_target_branches(repo_id, step_interval, only_final_model_eval)
    processed_steps = get_processed_steps(results_csv)
    task_input = tasks.split(",")

    # We do NOT write the header yet. We wait for the first result to determine
    # the full list of decomposed subtasks (e.g. mmlu_abstract_algebra, etc).

    for b in branches:
        step, branch_name = b["step"], b["name"]
        if step in processed_steps:
            logger.info(f"Skipping step {step} (already processed)"); continue

        logger.info(f"Evaluating Step {step} ({branch_name})")
        step_cache = Path(f"./tmp_cache_step_{b['step']}").resolve()
        step_cache.mkdir(parents=True, exist_ok=True)

        try:
            trust_remote_code = False if only_final_model_eval else True
            model_args = f"pretrained={repo_id},parallelize=True,revision={branch_name},trust_remote_code={trust_remote_code},cache_dir={step_cache}"

            # Run LM Eval
            # Note: When 'mmlu' is passed, lm_eval runs all subtasks.
            results = lm_eval.simple_evaluate(model="hf", model_args=model_args, tasks=task_input,
                                              num_fewshot=5, batch_size=batch_size, log_samples=False)

            # --- Process Results ---
            logger.info("Processing results")
            # 'results["results"]' contains keys for the group AND all subtasks
            res_dict = results["results"]
            metric_row = _extract_metrics(res_dict)

            # Prepare the full row data
            row_data = {"step": step, "branch": branch_name}
            row_data.update(metric_row)

            # --- CSV Handling ---
            file_exists = os.path.exists(results_csv)

            # Determine fieldnames (columns)
            # If file exists, use existing columns. If new, generate from this first run.
            if file_exists:
                with open(results_csv, "r", encoding="utf-8") as f:
                    fieldnames = csv.DictReader(f).fieldnames
            else:
                # Create columns: step, branch, then sorted task names
                task_cols = sorted(metric_row.keys())
                fieldnames = ["step", "branch"] + task_cols

            # Write to CSV
            with open(results_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)

            # Calculate average for logging purposes (simple mean of all columns found)
            avg_score = sum(metric_row.values()) / len(metric_row) if metric_row else 0
            logger.info(f"Completed step {step}. Approx Avg: {avg_score:.4f}\n")

        except Exception as e:
            logger.error(f"Failed to evaluate step {step}: {e}")
            # Optional: Log the full traceback if needed
            # import traceback; traceback.print_exc()
        finally:
            if not keep_cache:
                logger.debug("Clearing cache")
                shutil.rmtree(step_cache, ignore_errors=True)
    
    return results_csv


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MMLU evaluation on model checkpoints.")
    parser.add_argument("--repo_id", type=str, required=True, help="Target HuggingFace repository ID")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to output CSV results")
    parser.add_argument("--step_interval", type=int, default=1000, help="Steps between evaluated checkpoints")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size for eval")
    parser.add_argument("--tasks", type=str, default="mmlu", help="Comma-separated list of tasks")
    parser.add_argument("--only_final_model_eval", action="store_true", help="If set, only the final model will be evaluated.")
    parser.add_argument("--keep_cache", action="store_true", help="If set, keeps cache after running (normally clears by default)")
    
    args = parser.parse_args()

    eval_tasks(repo_id=args.repo_id,
               output_dir=args.output_dir,
               step_interval=args.step_interval,
               batch_size=args.batch_size,
               tasks=args.tasks,
               only_final_model_eval=args.only_final_model_eval,
               keep_cache=args.keep_cache)