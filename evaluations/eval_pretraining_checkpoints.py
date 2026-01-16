import os
import re
import csv
import json
import gc
import torch
import logging
import random
from collections import defaultdict
from typing import List, Dict, Any

from huggingface_hub import list_repo_refs, scan_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Import your internal libraries ---
from evaluations.developmental_skills import BenchmarkBuilder, BenchmarkSpec
from evaluations.counterfactual_transform import CounterfactualTransformer

logger = logging.getLogger(__name__)

# --- Configuration ---
# Toggle these to switch between OLMo and Pythia testing
# REPO_ID = "allenai/OLMo-7B"
REPO_ID = "EleutherAI/pythia-14m"

STEP_INTERVAL = 10000
METRICS_CSV = "schoolbench_%s_metrics.csv" % REPO_ID.split("/")[-1]
SAMPLES_CSV = "schoolbench_%s_samples.csv" % REPO_ID.split("/")[-1]
CLEAN_CACHE = False

SCHOOLBENCH_CONFIG = {
    "seed": 42,
    "cf_seed": 123,
    "n_per_skill": 2,  # Items per skill
    "shuffle": False,
    "max_new_tokens": 10
}


def get_target_branches(repo_id: str, interval: int) -> List[Dict[str, Any]]:
    """Returns sorted list of branches matching step intervals."""
    logger.info(f"Fetching branches from {repo_id}...")
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
    logger.info(f"Identified {len(sorted_branches)} target branches.")
    return sorted_branches


def cleanup_cache(repo_id: str, revision: str):
    """Deletes specific revision from HF cache."""
    logger.info(f"Attempting to clean cache for revision: {revision}")
    try:
        info = scan_cache_dir()
        repo = next((r for r in info.repos if r.repo_id == repo_id), None)
        if repo:
            for rev in repo.revisions:
                if revision in rev.refs:
                    rev.delete_strategy.execute()
                    logger.info(f"Successfully deleted cache for {revision}")
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
    logger.info("Generating evaluation dataset...")

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

    logger.info(f"Dataset ready: {len(pairs)} items generated.")
    return pairs


def evaluate_checkpoint(model, tokenizer, pairs, step_num, model_id):
    """Runs generation and scoring for a single checkpoint."""
    # Global Counters
    correct_base = 0
    correct_cf = 0
    count_cf = 0

    # Per-Skill Counters: {skill_name: {base_c, base_t, cf_c, cf_t}}
    skill_stats = defaultdict(lambda: {"base_correct": 0, "base_total": 0, "cf_correct": 0, "cf_total": 0})

    logger.info(f"Starting evaluation for {model_id}. Streaming samples to stdout/CSV...")

    header = "model_id,skill,prompt,prediction,cf_prompt,cf_prediction"
    print(header)

    file_exists = os.path.exists(SAMPLES_CSV)
    with open(SAMPLES_CSV, "a", newline="", encoding="utf-8") as f_samp:
        writer = csv.DictWriter(f_samp, fieldnames=header.split(","))
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

            # --- Generation ---
            def generate(prompt):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                            **inputs,
                            max_new_tokens=SCHOOLBENCH_CONFIG["max_new_tokens"],
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                    )
                return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            base_pred = generate(base["prompt"])
            cf_pred = generate(cf["prompt"]) if cf else ""

            # --- Scoring (Exact Match) ---
            # Base
            base_gold = base.get("gold", base.get("completion", "")).strip()
            base_hit = (base_gold and base_pred == base_gold)
            if base_hit:
                correct_base += 1
                skill_stats[skill]["base_correct"] += 1

            # Counterfactual
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
            print(
                f"{model_id},{skill},{repr(row['prompt'])},{repr(row['prediction'])},{repr(row['cf_prompt'])},{repr(row['cf_prediction'])}")
            writer.writerow(row)

    logger.info(f"Finished evaluation for {model_id}.")

    # --- Build Metrics Dictionary ---
    metrics = {
        "step": step_num,
        "branch": model_id,
        "base_acc": correct_base / len(pairs) if pairs else 0,
        "cf_acc": correct_cf / count_cf if count_cf else 0,
        "n_samples": len(pairs)
    }

    # Add Per-Skill Metrics
    for skill, stats in skill_stats.items():
        # Base Accuracy
        b_acc = stats["base_correct"] / stats["base_total"] if stats["base_total"] > 0 else 0.0
        metrics[f"skill.{skill}.base_acc"] = b_acc

        # CF Accuracy
        c_acc = 0.0
        if stats["cf_total"] > 0:
            c_acc = stats["cf_correct"] / stats["cf_total"]
            metrics[f"skill.{skill}.cf_acc"] = c_acc
        else:
            metrics[f"skill.{skill}.cf_acc"] = 0.0  # Or "N/A" if you prefer, but 0.0 keeps CSV clean

        # Gap (Robustness)
        if stats["base_total"] > 0 and stats["cf_total"] > 0:
            metrics[f"skill.{skill}.gap"] = b_acc - c_acc
        else:
            metrics[f"skill.{skill}.gap"] = 0.0

    return metrics


def main():
    branches = get_target_branches(REPO_ID, STEP_INTERVAL)
    completed_steps = get_processed_steps(METRICS_CSV)
    logger.info(f"Found {len(branches)} total checkpoints. {len(completed_steps)} already completed.")

    eval_data = prepare_data()

    # --- Pre-calculate Fieldnames for CSV Header ---
    # We must scan the data to know all possible skills to ensure the CSV header is valid
    all_skills = sorted(list(set(item["base"].get("skill", "unknown") for item in eval_data)))

    metrics_fieldnames = ["step", "branch", "base_acc", "cf_acc", "n_samples"]
    for skill in all_skills:
        metrics_fieldnames.append(f"skill.{skill}.base_acc")
        metrics_fieldnames.append(f"skill.{skill}.cf_acc")
        metrics_fieldnames.append(f"skill.{skill}.gap")

    logger.info(f"CSV Headers will include metrics for skills: {all_skills}")

    # --- Main Loop ---
    for b in branches:
        step = b["step"]
        branch_name = b["name"]

        if step in completed_steps:
            logger.info(f"Skipping step {step} (already in {METRICS_CSV})")
            continue

        logger.info(f"Processing Step {step} ({branch_name})")

        model = None
        tokenizer = None

        try:
            # 1. Load Resources
            logger.info(f"Loading model {branch_name}...")
            tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=branch_name)
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

            # 2. Evaluate
            metrics = evaluate_checkpoint(model, tokenizer, eval_data, step, branch_name)

            # 3. Save Metrics
            file_exists = os.path.exists(METRICS_CSV)
            with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
                # Use extrasaction='ignore' just in case, though we calculated headers carefully
                writer = csv.DictWriter(f, fieldnames=metrics_fieldnames, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics)

            logger.info(f"Saved aggregated metrics to {METRICS_CSV}")

        finally:
            # 4. Cleanup
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if CLEAN_CACHE:
                cleanup_cache(REPO_ID, branch_name)


if __name__ == "__main__":
    main()
