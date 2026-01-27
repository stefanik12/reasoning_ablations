import argparse
import csv
import gc
import logging
import os
import random
import hashlib
import shutil
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm

from evaluations.skillbench import generate_dataset
from evaluations.scoring import score_one
from evaluations.tools import get_processed_steps, get_target_branches, configure_logging, agg_add, agg_new, agg_finalize

configure_logging()
logger = logging.getLogger(__name__)


def _build_fewshot_prompt(test_prompt: str, test_gold: str, fewshot_examples: list, skill: str, seed: int) -> str:
    """Construct a few-shot prompt by prepending examples of the same skill."""
    # Filter examples by skill if possible
    skill_examples = [ex for ex in fewshot_examples if ex["skill"] == skill]
    if not skill_examples:
        skill_examples = fewshot_examples

    # Deterministic hash using hashlib instead of Python's hash()
    prompt_hash = int(hashlib.md5(test_prompt.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed + prompt_hash)

    skill_examples = sorted(skill_examples, key=lambda x: x.get("prompt", ""))
    skill_examples = skill_examples.copy()
    rng.shuffle(skill_examples)

    fewshot_str = ""
    used_prompts = set()
    for ex in skill_examples:
        ex_prompt = ex.get("prompt", "")
        ex_gold = ex.get("gold", ex.get("completion", "")).strip()
        # Skip if duplicate or matches test prompt
        if ex_prompt and ex_gold and ex_prompt not in used_prompts and ex_prompt != test_prompt:
            fewshot_str += f"{ex_prompt}{ex_gold}\n\n"
            used_prompts.add(ex_prompt)

    return fewshot_str + test_prompt


def _evaluate_checkpoint(model, tokenizer, pairs, fewshot_pool, n_fewshot, seed, step_num, model_id, samples_csv, topk_list):
    device = next(model.parameters()).device

    base_agg, cf_agg = agg_new(topk_list), agg_new(topk_list)
    base_ex = cf_ex = 0

    skills = defaultdict(lambda: {
        "base": agg_new(topk_list), "cf": agg_new(topk_list),
        "base_ex": 0, "cf_ex": 0
    })

    writer = None
    file_exists = os.path.exists(samples_csv)

    with open(samples_csv, "a", newline="", encoding="utf-8") as f:

        def write(kind, skill, prompt, gold):
            nonlocal writer
            out = score_one(model, tokenizer, prompt, gold, topk_list, device)
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
                agg_add(base_agg, out, topk_list); base_ex += 1
                agg_add(skills[skill]["base"], out, topk_list); skills[skill]["base_ex"] += 1
            else:
                agg_add(cf_agg, out, topk_list); cf_ex += 1
                agg_add(skills[skill]["cf"], out, topk_list); skills[skill]["cf_ex"] += 1

        for p in tqdm(pairs, desc=f"Scoring model for step {step_num}"):
            base, cf = p["base"], p["cf"]
            skill = base.get("skill", "unknown")

            bg = base.get("gold", base.get("completion", "")).strip()
            if bg:
                prompt = base["prompt"]
                if n_fewshot > 0 and fewshot_pool:
                    prompt = _build_fewshot_prompt(prompt, bg, fewshot_pool.get(skill, []), skill, seed)
                acc("base", skill, prompt, bg)

            if cf:
                cg = cf.get("gold", cf.get("completion", "")).strip()
                if cg:
                    prompt = cf["prompt"]
                    if n_fewshot > 0 and fewshot_pool:
                        prompt = _build_fewshot_prompt(prompt, cg, fewshot_pool.get(skill, []), skill, seed)
                    acc("cf", skill, prompt, cg)

    metrics = {"step": step_num, "branch": model_id, "n_samples": len(pairs),
               "base_n_examples": float(base_ex), "cf_n_examples": float(cf_ex)}
    metrics.update(agg_finalize(base_agg, "base", topk_list))
    metrics.update(agg_finalize(cf_agg, "cf", topk_list))

    for skill, s in skills.items():
        metrics[f"skill.{skill}.base_n_examples"] = float(s["base_ex"])
        metrics[f"skill.{skill}.cf_n_examples"] = float(s["cf_ex"])

        bm = agg_finalize(s["base"], f"skill.{skill}.base", topk_list)
        cm = agg_finalize(s["cf"], f"skill.{skill}.cf", topk_list)
        metrics.update(bm)
        metrics.update(cm)

        metrics[f"skill.{skill}.gap"] = (
            metrics.get(f"skill.{skill}.cf_ppl", 0.0) - metrics.get(f"skill.{skill}.base_ppl", 0.0)
            if s["cf_ex"] else 0.0
        )

    return metrics


def _split_fewshot_pool(data: list, n_fewshot: int, seed: int) -> tuple:
    """Split data into few-shot pool and test set, grouped by skill."""
    if n_fewshot <= 0:
        return {}, data

    rng = random.Random(seed)
    by_skill = defaultdict(list)
    for p in data:
        skill = p["base"].get("skill", "unknown")
        by_skill[skill].append(p)

    fewshot_pool = {}
    test_data = []

    for skill in sorted(by_skill.keys()):  # sort keys for determinism
        items = by_skill[skill]
        # Sort by prompt for determinism before shuffling
        items = sorted(items, key=lambda x: x["base"].get("prompt", ""))
        rng.shuffle(items)

        # Deduplicate by prompt
        seen_prompts = set()
        unique_items = []
        for item in items:
            prompt = item["base"].get("prompt", "")
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                unique_items.append(item)

        # Take n_fewshot examples for the pool, rest for testing
        pool_items = unique_items[:n_fewshot]
        test_items = unique_items[n_fewshot:]

        # Build fewshot pool from base examples
        fewshot_pool[skill] = [item["base"] for item in pool_items]
        test_data.extend(test_items)

    # Sort then shuffle for determinism
    test_data = sorted(test_data, key=lambda x: x["base"].get("prompt", ""))
    rng.shuffle(test_data)
    return fewshot_pool, test_data


def eval_skillbench(repo_id: str,
                    output_dir: str,
                    topk: str,
                    n_samples_per_skill: int,
                    n_fewshot: int,
                    seed: int,
                    step_interval: int,
                    only_final_model_eval: bool,
                    shuffle: bool,
                    cache_dir: str = None,
                    keep_cache: bool = False):

    # Set global seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    topk_list = sorted({int(x) for x in topk.split(",") if x.strip()}) if topk else []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fewshot_suffix = f"_fewshot{n_fewshot}" if n_fewshot > 0 else ""
    samples_csv = Path(output_dir) / f"schoolbench_{repo_id.split('/')[-1]}{fewshot_suffix}_samples.csv"
    metrics_csv = Path(output_dir) / f"schoolbench_{repo_id.split('/')[-1]}{fewshot_suffix}_metrics.csv"

    cache_dir = Path(cache_dir) if cache_dir else None

    logger.info("========== Generating data ==========")
    branches = get_target_branches(repo_id, step_interval, only_final_model_eval)
    completed = get_processed_steps(metrics_csv)
    data = generate_dataset(n_samples_per_skill, seed=seed, shuffle=shuffle)

    # Split into few-shot pool and test data
    fewshot_pool, test_data = _split_fewshot_pool(data, n_fewshot, seed)
    if n_fewshot > 0:
        logger.info(f"Using {n_fewshot}-shot evaluation. Pool: {sum(len(v) for v in fewshot_pool.values())} examples, Test: {len(test_data)} samples")

    all_skills = sorted({p["base"].get("skill", "unknown") for p in test_data})
    fields = [
        "step", "branch",
        "base_ppl", "base_n_tokens", "base_n_examples",
        "cf_ppl", "cf_n_tokens", "cf_n_examples",
        "n_samples",
    ]
    for k in topk_list:
        fields += [f"base_top{k}_acc", f"cf_top{k}_acc"]
    for s in all_skills:
        fields += [
            f"skill.{s}.base_ppl", f"skill.{s}.base_n_tokens", f"skill.{s}.base_n_examples",
            f"skill.{s}.cf_ppl", f"skill.{s}.cf_n_tokens", f"skill.{s}.cf_n_examples",
        ]
        for k in topk_list:
            fields += [f"skill.{s}.base_top{k}_acc", f"skill.{s}.cf_top{k}_acc"]
        fields.append(f"skill.{s}.gap")


    logger.info("\n========== Scoring model ==========")
    for b in branches:
        if b["step"] in completed:
            continue
        
        logger.info(f"\nStep {b['step']}")

        cache = Path(f"./tmp_cache_skillbench_{repo_id}_step_{b['step']}")
        step_cache = (cache_dir/cache).resolve() if cache_dir else cache.resolve()
        step_cache.mkdir(parents=True, exist_ok=True)

        model = None
        tok = None
        try:
            logger.info("Loading model...")
            trust_remote_code = False if only_final_model_eval else True
            tok = AutoTokenizer.from_pretrained(repo_id, revision=b["name"],
                                                trust_remote_code=trust_remote_code, cache_dir=step_cache)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                repo_id, revision=b["name"], trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                cache_dir=step_cache
            ).eval()
            logger.info("Model loaded successfully!")

            m = _evaluate_checkpoint(model, tok, test_data, fewshot_pool, n_fewshot, seed, b["step"], b["name"], samples_csv, topk_list)

            logger.info("Saving output")
            write_header = not os.path.exists(metrics_csv)
            with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerow(m)

        finally:
            logger.debug("Clearing model and tokenizer from memory")
            if model is not None:
                del model
            if tok is not None:
                del tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not keep_cache:
                logger.debug("Clearing cache")
                shutil.rmtree(step_cache, ignore_errors=True)
        
    logger.info("\nDone")

    return metrics_csv, samples_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SchoolBench evaluation on model checkpoints.")
    parser.add_argument("--repo_id", type=str, help="Target HuggingFace repository ID (e.g., 'swiss-ai/Apertus-70B-2509')")
    parser.add_argument("-n", "--n_samples_per_skill", type=int, default=2500, help="Number of samples to draw per skill (note, duplicates will be discarded so returned samples per skill will be less than specified value)")
    parser.add_argument("--n_fewshot", type=int, default=0, help="Number of few-shot examples to prepend (0 = zero-shot)")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to output CSV results")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed used for random number generation")
    parser.add_argument("--topk", type=str, default="1,3,5,10,20", help="Comma-separated top-k list for accuracy (e.g., '1,5,10')")
    parser.add_argument("--step_interval", type=int, default=10000, help="Number of training steps between evaluated checkpoints")
    parser.add_argument("--only_final_model_eval", action="store_true", help="If set, only the final model will be evaluated.")
    parser.add_argument("--shuffle", action="store_true", help="If set, data will be shuffled on generation")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to store temp cache")
    parser.add_argument("--keep_cache", action="store_true", help="If set, keeps cache after running (normally clears by default)")
    args = parser.parse_args()

    eval_skillbench(repo_id=args.repo_id,
                    output_dir=args.output_dir,
                    topk=args.topk,
                    n_samples_per_skill=args.n_samples_per_skill,
                    n_fewshot=args.n_fewshot,
                    seed=args.seed,
                    step_interval=args.step_interval,
                    only_final_model_eval=args.only_final_model_eval,
                    shuffle=args.shuffle,
                    cache_dir=args.cache_dir,
                    keep_cache=args.keep_cache)