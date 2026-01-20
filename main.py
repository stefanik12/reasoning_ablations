import json
import argparse
from pathlib import Path
import torch
from evaluations.scoring import score_one
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluations.eval_model import get_device
from evaluations.counterfactual_transform import generate_dataset
import tempfile
import os
from typing import List


def main(model_name: str, 
         topk_list: List[int],
         output_path: str = "data/results.json", 
         n_samples_per_skill: int = 2500,
         seed: int = 42,
         ):
    
    # Check if GPUs are properly exposed
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}\n")

    # --- Step 1: Generate Data (Temp File Workaround) ---
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        temp_path = tmp.name

    try:
        generate_dataset(n_samples_per_skill, temp_path, seed)
        with open(temp_path, "r", encoding="utf-8") as f:
            problems = json.load(f)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


    print(f"Loaded {len(problems)} problems")

    # Load model onto GPUs
    print(f"Loading model: {model_name}")
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # Mandatory for decoder-only batched inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto"
    )

    model.eval()

    # Construct results JSON
    results = []
    for i, problem in enumerate(problems):
        
        base_scoring = score_one(model, tokenizer, problem["base"], problem["base_completion"], topk_list, model.device)
        cf_scoring = score_one(model, tokenizer, problem["cf"], problem["cf_completion"], topk_list, model.device)

        result = {
            "base_prompt": problem["base"],
            "base_expected": problem["base_completion"],
            "base_nll_sum": base_scoring["nll_sum"],
            "base_tok": base_scoring["tok"],
            "base_topk_hits": base_scoring["topk_hits"],
            "cf_prompt": problem["cf"],
            "cf_expected": problem["cf_completion"],
            "cf_nll_sum": cf_scoring["nll_sum"],
            "cf_tok": cf_scoring["tok"],
            "cf_topk_hits": cf_scoring["topk_hits"],
            "cf_edit": problem["cf_edit"],
            "skill": problem["skill"],
        }
        
        results.append(result)

    print(f"Generated {len(results)} result pairs")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tests given LLM on task dataset")
    parser.add_argument("--model_name", help="Model tag found on Hugging Face")
    parser.add_argument("--output_path", help="Path to output dataset")
    parser.add_argument("-n", "--n_samples_per_skill", type=int, help="Number of samples to generate per skill (note, duplicates will be discarded so actual samples per skill will be less than this)")
    parser.add_argument("-s", "--seed", type=int, help="Seed used for random number generation")

    args = parser.parse_args()

    main(args.model_name, 
         args.topk_list,
         args.output_path, 
         args.n_samples_per_skill,
         args.seed
    )