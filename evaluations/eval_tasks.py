import argparse
import csv
import logging
import os
import shutil
import lm_eval
from pathlib import Path
from typing import Any, Dict

import torch.mps

from evaluations.tools import get_target_branches, get_processed_steps, configure_logging, parse_batch_size

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


def _coerce_choice_index(val: Any) -> int | None:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, (list, tuple)) and val:
        return _coerce_choice_index(val[0])
    if isinstance(val, str):
        s = val.strip()
        if s.isdigit():
            return int(s)
        if len(s) == 1 and s.lower() in "abcd":
            return ord(s.lower()) - ord("a")
    return None


def _extract_correct_choice_index(row: Dict[str, Any]) -> int | None:
    for key in ("gold", "answer", "label", "target"):
        if key in row:
            idx = _coerce_choice_index(row.get(key))
            if idx is not None:
                return idx

    doc = row.get("doc")
    if isinstance(doc, dict):
        for key in ("gold", "answer", "label", "target"):
            idx = _coerce_choice_index(doc.get(key))
            if idx is not None:
                return idx
    return None


def _aggregate_choice_probs_from_samples(samples: Any) -> Dict[str, float]:
    """Extract mean correct-choice logprobs/probs from lm_eval's `results["samples"]`."""
    if not isinstance(samples, dict):
        return {}

    out: Dict[str, float] = {}

    for task, rows in samples.items():
        if not isinstance(rows, list):
            continue

        # lm_eval stores per-choice logprobs in "resps" or "filtered_resps"
        # Format: list of (logprob, is_greedy) tuples per choice
        correct_logprobs: list[float] = []
        correct_probs: list[float] = []

        for r in rows:
            if not isinstance(r, dict):
                continue

            # Try "filtered_resps" first (post-processed), then "resps" (raw)
            resps = r.get("filtered_resps") or r.get("resps")
            if not isinstance(resps, (list, tuple)):
                continue

            vals: list[float] = []
            for resp in resps:
                # Each resp is typically (logprob, is_greedy) or just logprob
                if isinstance(resp, (list, tuple)) and len(resp) >= 1:
                    val = resp[0]
                elif isinstance(resp, (int, float)):
                    val = resp
                else:
                    continue

                if isinstance(val, (int, float)):
                    vals.append(float(val))

            correct_idx = _extract_correct_choice_index(r)
            if correct_idx is None or correct_idx < 0 or correct_idx >= len(vals):
                continue

            correct_logprobs.append(vals[correct_idx])

            if vals:
                import math
                max_lp = max(vals)
                exp_vals = [math.exp(v - max_lp) for v in vals]
                total = sum(exp_vals)
                if total > 0:
                    correct_probs.append(exp_vals[correct_idx] / total)

        if correct_logprobs:
            out[f"logprob_mean__{task}__choice_correct"] = sum(correct_logprobs) / len(correct_logprobs)
        if correct_probs:
            out[f"prob_mean__{task}__choice_correct"] = sum(correct_probs) / len(correct_probs)

    return out


def eval_tasks(repo_id,
               output_dir,
               step_interval,
               batch_size,
               tasks,
               only_final_model_eval,
               cache_dir: str = None,
               keep_cache: bool = False,
               limit: int = None):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cache_dir) if cache_dir else None

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
        cache = Path(f"./tmp_cache_tasks_{repo_id}_step_{b['step']}")
        step_cache = (cache_dir/cache).resolve() if cache_dir else cache.resolve()
        step_cache.mkdir(parents=True, exist_ok=True)

        try:
            trust_remote_code = False if only_final_model_eval else True
            model_args = f"pretrained={repo_id},revision={branch_name},trust_remote_code={trust_remote_code},cache_dir={step_cache}"

            # Run LM Eval
            # Note: When 'mmlu' is passed, lm_eval runs all subtasks.
            results = lm_eval.simple_evaluate(model="hf", model_args=model_args, tasks=task_input,
                                              device=None if not torch.mps.is_available() else "cpu",
                                              num_fewshot=5, batch_size=batch_size, log_samples=True, limit=limit)

            # Log first sample keys for debugging
            samples = results.get("samples", {})
            for task, rows in samples.items():
                if rows and isinstance(rows[0], dict):
                    logger.info(f"Sample keys for {task}: {list(rows[0].keys())}")
                break

            # --- Process Results ---
            logger.info("Processing results")
            res_dict = results["results"]
            metric_row = _extract_metrics(res_dict)
            prob_row = _aggregate_choice_probs_from_samples(results.get("samples"))

            row_data = {"step": step, "branch": branch_name}
            row_data.update(metric_row)
            row_data.update(prob_row)

            # --- CSV Handling ---
            file_exists = os.path.exists(results_csv)

            if file_exists:
                with open(results_csv, "r", encoding="utf-8") as f:
                    existing = csv.DictReader(f).fieldnames or []
                new_cols = sorted([c for c in row_data.keys() if c not in existing])
                fieldnames = list(existing) + new_cols
            else:
                task_cols = sorted(metric_row.keys())
                prob_cols = sorted(prob_row.keys())
                fieldnames = ["step", "branch"] + task_cols + prob_cols

            # Write to CSV
            with open(results_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)

            # Calculate average for logging purposes (simple mean of all columns found)
            avg_score = sum(metric_row.values()) / len(metric_row) if metric_row else 0
            logger.info(f"Completed step {step}. Approx Avg: {avg_score:.4f}\n")

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
    parser.add_argument("--batch_size", type=parse_batch_size, default="auto", help="Batch size for eval")
    parser.add_argument("--tasks", type=str, default="mmlu", help="Comma-separated list of tasks")
    parser.add_argument("--only_final_model_eval", action="store_true", help="If set, only the final model will be evaluated.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to store temp cache")
    parser.add_argument("--keep_cache", action="store_true", help="If set, keeps cache after running (normally clears by default)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per task for quicker eval")

    args = parser.parse_args()

    eval_tasks(repo_id=args.repo_id,
               output_dir=args.output_dir,
               step_interval=args.step_interval,
               batch_size=args.batch_size,
               tasks=args.tasks,
               only_final_model_eval=args.only_final_model_eval,
               cache_dir=args.cache_dir,
               keep_cache=args.keep_cache,
               limit=args.limit)