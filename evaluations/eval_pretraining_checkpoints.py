import argparse
import csv
import gc
import logging
import os
import random
import re
from collections import defaultdict
from typing import List, Dict, Any

import torch
from huggingface_hub import list_repo_refs, scan_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Import your internal libraries ---
from evaluations.counterfactual_transform import CounterfactualTransformer
from evaluations.developmental_skills import BenchmarkBuilder, BenchmarkSpec

logger = logging.getLogger(__name__)

# --- Configuration ---
# Toggle these to switch between OLMo and Pythia testing
# REPO_ID = "allenai/OLMo-7B"
# REPO_ID = "EleutherAI/pythia-14m"
# REPO_ID = "swiss-ai/Apertus-8B-2509"

parser = argparse.ArgumentParser(description="Run SchoolBench evaluation on model checkpoints.")
parser.add_argument("repo_id", type=str, help="Target HuggingFace repository ID (e.g., 'swiss-ai/Apertus-70B-2509')")
args = parser.parse_args()

# --- Configuration ---
REPO_ID = args.repo_id  # Dynamic value from CLI

STEP_INTERVAL = 10000
METRICS_CSV = "schoolbench_%s_metrics.csv" % REPO_ID.split("/")[-1]
SAMPLES_CSV = "schoolbench_%s_samples.csv" % REPO_ID.split("/")[-1]
CLEAN_CACHE = True

SCHOOLBENCH_CONFIG = {
    "seed": 42,
    "cf_seed": 123,
    "n_per_skill": 1000,  # Items per skill
    "shuffle": False,
    "max_new_tokens": 10
}


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

    sorted_branches = sorted(branches, key=lambda x: x["step"])
    logger.warning(f"Identified {len(sorted_branches)} target branches.")
    return sorted_branches


def cleanup_cache(repo_id: str, revision: str):
    """Deletes specific revision from HF cache."""
    logger.warning(f"Attempting to clean cache for revision: {revision}")
    try:
        info = scan_cache_dir()
        repo = next((r for r in info.repos if r.repo_id == repo_id), None)
        if repo:
            for rev in repo.revisions:
                if revision in rev.refs:
                    rev.delete_strategy.execute()
                    logger.warning(f"Successfully deleted cache for {revision}")
                    return
        logger.warning(f"Revision {revision} not found in cache during cleanup.")
    except (OSError, ValueError) as e:
        logger.error(f"Cache cleanup failed: {e}")


def get_processed_steps(csv_path: str) -> set:
    """Returns set of steps already present in the metrics CSV."""
    steps = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("step"):
                        steps.add(int(row["step"]))
        except (csv.Error, ValueError, OSError) as e:
            logger.warning(f"Could not read existing CSV {csv_path}: {e}")
    return steps


def prepare_data() -> List[Dict[str, Any]]:
    """Generates the Paired (Base + CF) dataset."""
    logger.warning("Generating evaluation dataset...")

    # Handle n_per_skill configuration
    n_cfg = SCHOOLBENCH_CONFIG["n_per_skill"]
    if isinstance(n_cfg, int):
        dummy_spec = BenchmarkSpec(n_per_skill={})
        dummy_builder = BenchmarkBuilder(dummy_spec)
        n_per_skill_arg = {s.name: n_cfg for s in dummy_builder.skills}
    else:
        n_per_skill_arg = n_cfg

    spec = BenchmarkSpec(
            seed=SCHOOLBENCH_CONFIG["seed"],
            n_per_skill=n_per_skill_arg,
            shuffle=SCHOOLBENCH_CONFIG["shuffle"]
    )
    base_items = BenchmarkBuilder(spec).generate()

    # Apply Counterfactual Transforms
    tfm = CounterfactualTransformer()
    rng = random.Random(SCHOOLBENCH_CONFIG["cf_seed"])

    pairs = []
    for item in base_items:
        shortcuts = tfm.applicable_shortcuts(item)
        cf_item = None
        if shortcuts:
            chosen = rng.choice(shortcuts)
            cf_item = tfm.transform(item, shortcut=chosen, rng=rng)

        pairs.append({"base": item, "cf": cf_item})

    logger.warning(f"Dataset ready: {len(pairs)} items generated.")
    return pairs


def evaluate_checkpoint(model, tokenizer, pairs, step_num, model_id):
    """Runs generation and scoring for a single checkpoint."""
    # Global Counters
    correct_base = 0
    correct_cf = 0
    count_cf = 0

    # Per-Skill Counters
    skill_stats = defaultdict(lambda: {"base_correct": 0, "base_total": 0, "cf_correct": 0, "cf_total": 0})

    logger.warning(f"Starting evaluation for {model_id}...")

    # We open the file here to stream results row-by-row
    file_exists = os.path.exists(SAMPLES_CSV)
    with open(SAMPLES_CSV, "a", newline="", encoding="utf-8") as f_samp:
        fieldnames = ["model_id", "skill", "prompt", "prediction", "cf_prompt", "cf_prediction"]
        writer = csv.DictWriter(f_samp, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, p in enumerate(pairs):
            if i > 0 and i % 50 == 0:
                logger.debug(f"Processed {i}/{len(pairs)} samples...")

            base = p["base"]
            cf = p["cf"]
            skill = base.get("skill", "unknown")

            # Update Totals
            skill_stats[skill]["base_total"] += 1
            if cf:
                skill_stats[skill]["cf_total"] += 1

            # --- Generation (UNIVERSAL FIX) ---
            def generate(prompt):
                # 1. Tokenize
                raw_inputs = tokenizer(prompt, return_tensors="pt")

                # 2. Filter & Move to Device (Exclude token_type_ids for OLMo)
                model_inputs = {
                    k: v.to(model.device) for k, v in raw_inputs.items()
                    if k != "token_type_ids"
                }

                # 3. Generate
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    with torch.no_grad():
                        out = model.generate(
                                **model_inputs,
                                max_new_tokens=SCHOOLBENCH_CONFIG["max_new_tokens"],
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                use_cache=False
                        )

                # 4. Decode
                input_len = model_inputs["input_ids"].shape[1]
                return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

            base_pred = generate(base["prompt"])
            cf_pred = generate(cf["prompt"]) if cf else ""

            # --- Scoring ---
            base_gold = base.get("gold", base.get("completion", "")).strip()
            base_hit = (base_gold and base_pred == base_gold)
            if base_hit:
                correct_base += 1
                skill_stats[skill]["base_correct"] += 1

            if cf:
                count_cf += 1
                cf_gold = cf.get("gold", cf.get("completion", "")).strip()
                cf_hit = (cf_gold and cf_pred == cf_gold)
                if cf_hit:
                    correct_cf += 1
                    skill_stats[skill]["cf_correct"] += 1

            # --- Logging ---
            row = {
                "model_id": model_id,
                "skill": skill,
                "prompt": base["prompt"],
                "prediction": base_pred,
                "cf_prompt": cf["prompt"] if cf else "",
                "cf_prediction": cf_pred
            }
            # Only write to CSV, do not print to stdout
            writer.writerow(row)

    logger.warning(f"Finished evaluation for {model_id}.")

    # --- Metrics ---
    metrics = {
        "step": step_num,
        "branch": model_id,
        "base_acc": correct_base / len(pairs) if pairs else 0,
        "cf_acc": correct_cf / count_cf if count_cf else 0,
        "n_samples": len(pairs)
    }

    # Per-Skill Metrics
    for skill, stats in skill_stats.items():
        b_acc = stats["base_correct"] / stats["base_total"] if stats["base_total"] > 0 else 0.0
        c_acc = stats["cf_correct"] / stats["cf_total"] if stats["cf_total"] > 0 else 0.0

        metrics[f"skill.{skill}.base_acc"] = b_acc
        metrics[f"skill.{skill}.cf_acc"] = c_acc

        if stats["cf_total"] > 0:
            metrics[f"skill.{skill}.gap"] = b_acc - c_acc
        else:
            metrics[f"skill.{skill}.gap"] = 0.0

    return metrics


def main():
    branches = get_target_branches(REPO_ID, STEP_INTERVAL)
    completed_steps = get_processed_steps(METRICS_CSV)
    logger.warning(f"Found {len(branches)} total checkpoints. {len(completed_steps)} already completed.")

    eval_data = prepare_data()

    # Pre-calculate CSV Headers for Metrics file
    all_skills = sorted(list(set(item["base"].get("skill", "unknown") for item in eval_data)))
    metrics_fieldnames = ["step", "branch", "base_acc", "cf_acc", "n_samples"]
    for skill in all_skills:
        metrics_fieldnames.append(f"skill.{skill}.base_acc")
        metrics_fieldnames.append(f"skill.{skill}.cf_acc")
        metrics_fieldnames.append(f"skill.{skill}.gap")

    logger.warning(f"CSV Headers will include metrics for skills: {all_skills}")

    for b in branches:
        step = b["step"]
        branch_name = b["name"]

        if step in completed_steps:
            logger.warning(f"Skipping step {step} (already in {METRICS_CSV})")
            continue

        logger.warning(f"Processing Step {step} ({branch_name})")

        model = None
        tokenizer = None

        try:
            logger.warning(f"Loading model {branch_name}...")
            tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=branch_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                    REPO_ID,
                    revision=branch_name,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            model.eval()

            metrics = evaluate_checkpoint(model, tokenizer, eval_data, step, branch_name)

            file_exists = os.path.exists(METRICS_CSV)
            with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=metrics_fieldnames, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics)

            logger.warning(f"Saved aggregated metrics to {METRICS_CSV}")

        finally:
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            if CLEAN_CACHE:
                cleanup_cache(REPO_ID, branch_name)


if __name__ == "__main__":
    main()
