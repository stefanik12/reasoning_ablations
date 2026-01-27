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


def _aggregate_choice_probs_from_samples(samples: Any) -> Dict[str, float]:
    """Mean per-choice vectors from `results.get("samples")` where present."""
    if not isinstance(samples, dict):
        return {}

    out: Dict[str, float] = {}
    prob_keys = ("choices_probs", "choice_probs", "probs", "probabilities")
    lprob_keys = ("choices_logprobs", "choice_logprobs", "logprobs", "log_probs", "logprobabilities")

    for task, rows in samples.items():
        if not isinstance(rows, list):
            continue

        probs, lprobs = [], []
        for r in rows:
            if not isinstance(r, dict):
                continue
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}

            def pick(keys):
                for k in keys:
                    v = r.get(k, meta.get(k))
                    if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
                        return list(map(float, v))
                return None

            if (v := pick(prob_keys)) is not None: probs.append(v)
            if (v := pick(lprob_keys)) is not None: lprobs.append(v)

        for name, vecs in (("choice_prob_mean", probs), ("choice_logprob_mean", lprobs)):
            if not vecs:
                continue
            n = max((len(v) for v in vecs), default=0)
            for i in range(n):
                vals = [v[i] for v in vecs if i < len(v)]
                if vals:
                    out[f"{name}__{task}__{i}"] = sum(vals) / len(vals)

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