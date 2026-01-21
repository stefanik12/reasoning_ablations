import argparse
import csv
import gc
import logging
import os
import random
import re
import shutil
from collections import defaultdict
from typing import List, Dict, Any

import torch
from huggingface_hub import list_repo_refs
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Import your internal libraries ---
from evaluations.counterfactual_transform import CounterfactualTransformer
from evaluations.developmental_skills import BenchmarkBuilder, BenchmarkSpec
from evaluations.scoring import score_one, agg_add, agg_new, agg_finalize

logger = logging.getLogger(__name__)

# --- Configuration ---
# Toggle these to switch between OLMo and Pythia testing
# REPO_ID = "allenai/OLMo-7B"
# REPO_ID = "EleutherAI/pythia-14m"
# REPO_ID = "swiss-ai/Apertus-8B-2509"

parser = argparse.ArgumentParser(description="Run SchoolBench evaluation on model checkpoints.")
parser.add_argument("--repo_id", type=str, help="Target HuggingFace repository ID (e.g., 'swiss-ai/Apertus-70B-2509')")
parser.add_argument("--topk", type=str, default="1,10,100", help="Comma-separated top-k list for accuracy (e.g., '1,5,10')")
parser.add_argument("--num_samples_per_skill", type=int, default=1000, help="Number of evaluation samples per skill")
parser.add_argument("--step_interval", type=int, default=10000, help="Number of training steps between evaluated checkpoints")
parser.add_argument("--only_final_model_eval", action="store_true", help="If set, only the final model will be evaluated.")
args = parser.parse_args()

REPO_ID = args.repo_id
TOPK_LIST = sorted({int(x) for x in args.topk.split(",") if x.strip()}) if args.topk else []

STEP_INTERVAL = args.step_interval
METRICS_CSV = "schoolbench_%s_metrics.csv" % REPO_ID.split("/")[-1]
SAMPLES_CSV = "schoolbench_%s_samples.csv" % REPO_ID.split("/")[-1]
CLEAN_CACHE = True

SCHOOLBENCH_CONFIG = {
    "seed": 42,
    "cf_seed": 123,
    "n_per_skill": args.num_samples_per_skill,
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
    pattern = re.compile(r"(?:.*)?step(\d+)(?:.*)?$")

    for b in refs.branches:
        match = pattern.match(b.name)
        if match:
            step = int(match.group(1))
            if step % interval == 0:
                branches.append({"step": step, "name": b.name})
    if args.only_final_model_eval:
        return [{"step": 0, "name": "main"}]
    else:
        sorted_branches = sorted(branches, key=lambda x: x["step"], reverse=True)
        logger.warning(f"Identified {len(sorted_branches)} target branches.")
        return sorted_branches


def get_processed_steps(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, "r", encoding="utf-8") as f:
        return {int(r["step"]) for r in csv.DictReader(f) if r.get("step")}


def prepare_data() -> List[Dict[str, Any]]:
    n_cfg = SCHOOLBENCH_CONFIG["n_per_skill"]
    if isinstance(n_cfg, int):
        dummy = BenchmarkBuilder(BenchmarkSpec(n_per_skill={})).skills
        n_cfg = {s.name: SCHOOLBENCH_CONFIG["n_per_skill"] for s in dummy}

    spec = BenchmarkSpec(seed=SCHOOLBENCH_CONFIG["seed"], n_per_skill=n_cfg,
                         shuffle=SCHOOLBENCH_CONFIG["shuffle"])
    base_items = BenchmarkBuilder(spec).generate()

    tfm = CounterfactualTransformer()
    rng = random.Random(SCHOOLBENCH_CONFIG["cf_seed"])

    pairs = []
    for it in base_items:
        cfs = tfm.applicable_shortcuts(it)
        cf = tfm.transform(it, rng.choice(cfs), rng) if cfs else None
        pairs.append({"base": it, "cf": cf})
    return pairs


def evaluate_checkpoint(model, tokenizer, pairs, step_num, model_id):
    device = next(model.parameters()).device

    base_agg, cf_agg = agg_new(TOPK_LIST), agg_new(TOPK_LIST)
    base_ex = cf_ex = 0

    skills = defaultdict(lambda: {
        "base": agg_new(TOPK_LIST), "cf": agg_new(TOPK_LIST),
        "base_ex": 0, "cf_ex": 0
    })

    writer = None
    file_exists = os.path.exists(SAMPLES_CSV)

    with open(SAMPLES_CSV, "a", newline="", encoding="utf-8") as f:

        def write(kind, skill, prompt, gold):
            nonlocal writer
            out = score_one(model, tokenizer, prompt, gold, TOPK_LIST, device)
            if writer is None:
                fields = ["model_id", "step", "branch", "skill", "kind", "prompt", *out.keys()]
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
            row = {"model_id": model_id, "step": step_num, "branch": model_id, "skill": skill, "kind": kind,
                   "prompt": prompt, **out}
            writer.writerow(row)
            return out

        def acc(kind, skill, prompt, gold):
            nonlocal base_ex, cf_ex
            out = write(kind, skill, prompt, gold)
            if kind == "base":
                agg_add(base_agg, out, TOPK_LIST); base_ex += 1
                agg_add(skills[skill]["base"], out, TOPK_LIST); skills[skill]["base_ex"] += 1
            else:
                agg_add(cf_agg, out, TOPK_LIST); cf_ex += 1
                agg_add(skills[skill]["cf"], out, TOPK_LIST); skills[skill]["cf_ex"] += 1

        for p in pairs:
            base, cf = p["base"], p["cf"]
            skill = base.get("skill", "unknown")

            bg = base.get("gold", base.get("completion", "")).strip()
            if bg:
                acc("base", skill, base["prompt"], bg)

            if cf:
                cg = cf.get("gold", cf.get("completion", "")).strip()
                if cg:
                    acc("cf", skill, cf["prompt"], cg)

    metrics = {"step": step_num, "branch": model_id, "n_samples": len(pairs),
               "base_n_examples": float(base_ex), "cf_n_examples": float(cf_ex)}
    metrics.update(agg_finalize(base_agg, "base", TOPK_LIST))
    metrics.update(agg_finalize(cf_agg, "cf", TOPK_LIST))

    for skill, s in skills.items():
        metrics[f"skill.{skill}.base_n_examples"] = float(s["base_ex"])
        metrics[f"skill.{skill}.cf_n_examples"] = float(s["cf_ex"])

        bm = agg_finalize(s["base"], f"skill.{skill}.base", TOPK_LIST)
        cm = agg_finalize(s["cf"], f"skill.{skill}.cf", TOPK_LIST)
        metrics.update(bm)
        metrics.update(cm)

        metrics[f"skill.{skill}.gap"] = (
            metrics.get(f"skill.{skill}.cf_ppl", 0.0) - metrics.get(f"skill.{skill}.base_ppl", 0.0)
            if s["cf_ex"] else 0.0
        )

    return metrics


def main():
    branches = get_target_branches(REPO_ID, STEP_INTERVAL)
    completed = get_processed_steps(METRICS_CSV)
    data = prepare_data()

    all_skills = sorted({p["base"].get("skill", "unknown") for p in data})
    fields = [
        "step", "branch",
        "base_ppl", "base_n_tokens", "base_n_examples",
        "cf_ppl", "cf_n_tokens", "cf_n_examples",
        "n_samples",
    ]
    for k in TOPK_LIST:
        fields += [f"base_top{k}_acc", f"cf_top{k}_acc"]
    for s in all_skills:
        fields += [
            f"skill.{s}.base_ppl", f"skill.{s}.base_n_tokens", f"skill.{s}.base_n_examples",
            f"skill.{s}.cf_ppl", f"skill.{s}.cf_n_tokens", f"skill.{s}.cf_n_examples",
        ]
        for k in TOPK_LIST:
            fields += [f"skill.{s}.base_top{k}_acc", f"skill.{s}.cf_top{k}_acc"]
        fields.append(f"skill.{s}.gap")

    for b in branches:
        if b["step"] in completed:
            continue

        step_cache = os.path.abspath(f"./tmp_cache_step_{b['step']}")
        try:
            tok = AutoTokenizer.from_pretrained(REPO_ID, revision=b["name"],
                                                trust_remote_code=True, cache_dir=step_cache)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                REPO_ID, revision=b["name"], trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                cache_dir=step_cache
            ).eval()

            m = evaluate_checkpoint(model, tok, data, b["step"], b["name"])

            write_header = not os.path.exists(METRICS_CSV)
            with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerow(m)

        finally:
            del model, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if CLEAN_CACHE:
                shutil.rmtree(step_cache, ignore_errors=True)


if __name__ == "__main__":
    main()
