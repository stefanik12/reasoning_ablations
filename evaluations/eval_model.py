# testable with
# python evaluations/eval_model.py --n_samples_per_skill 2 --seed 42 --model_name EleutherAI/pythia-14m

import argparse, json, os, tempfile, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluations.counterfactual_transform import generate_dataset


def get_device():
    """Selects the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # Avoid using Macbook's mps device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_samples_per_skill", type=int, required=True)
    parser.add_argument("-s", "--seed", type=int, required=True)
    parser.add_argument("-t", "--temperature", type=float, default=0.6, help="Model temperature")
    parser.add_argument("-m", "--max_new_tokens", type=int, default=10)
    parser.add_argument("--model_name", help="Model tag found on Hugging Face")
    args = parser.parse_args()

    # --- Step 1: Generate Data (Temp File Workaround) ---
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        temp_path = tmp.name

    try:
        generate_dataset(args.n_samples_per_skill, temp_path, args.seed)
        with open(temp_path, "r", encoding="utf-8") as f:
            problems = json.load(f)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # --- Step 2: Load Model & Tokenizer ---
    device = get_device()
    print(f"Loading {args.model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"  # Mandatory for decoder-only batched inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device
    )
    model.eval()

    # --- Step 3: Batched Inference ---
    all_prompts = [p["base"] for p in problems] + [p["cf"] for p in problems]
    batch_size = 8
    all_outputs = []

    print(f"Generating responses for {len(all_prompts)} prompts...")

    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i: i + batch_size]

        # Tokenize with explicit padding and attention masks
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
            )

        # Decode and Post-process
        decoded_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for prompt, full_text in zip(batch_prompts, decoded_batch):
            # Strip the input prompt from the output
            response = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            # Apply stop token logic (stop at newline)
            response = response.strip().split('\n')[0]
            all_outputs.append(response)

    # --- Step 4: Save Results ---
    results, n = [], len(problems)
    for i, p in enumerate(problems):
        b_res = all_outputs[i]
        c_res = all_outputs[n + i]
        b_exp, c_exp = p["base_completion"].strip(), p["cf_completion"].strip()

        results.append({
            **p,
            "base_response": b_res,
            "base_pass": {"exact": b_res == b_exp, "fuzzy": b_res.endswith(b_exp)},
            "cf_response": c_res,
            "cf_pass": {"exact": c_res == c_exp, "fuzzy": c_res.endswith(c_exp)}
        })

    filename = f"results_{args.model_name.replace('/', '_')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Done! Results saved to {filename}")
