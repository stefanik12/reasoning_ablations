from vllm import LLM, SamplingParams
import json
import argparse
from pathlib import Path
import torch

def check_response(response: str, expected: str) -> dict:
    
    matches = {}
    response = response.strip()
    expected = expected.strip()
    
    if not response or not expected:
        return {"exact": False, "fuzzy": False}
    
    # We're logging exact matched responses and if the final part of the answer contains the expected response
    matches["exact"] = True if response == expected else False
    matches["fuzzy"] = response.endswith(expected)

    return matches


def main(model_name: str, temperature: float = 0.6, input_path: str = "data/dataset.json", output_path: str = "data/results.json", trust_remote_code: bool = False):
    
    # Check if GPUs are properly exposed
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}\n")

    
    # Load dataset
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems")

    # Load model onto GPUs
    print(f"Loading model: {model_name}")
    num_gpus = torch.cuda.device_count()

    llm = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        max_model_len=4096,
        trust_remote_code=trust_remote_code
    )

    print("Model loaded successfully!")

    # Sampling parameters for generation
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=128,
        stop=["\n"]     # Weird quirk: if we remove this, the LLM will generate new questions
    )

    # Prepare all prompts
    base_prompts = [p["base"] for p in problems]
    cf_prompts = [p["cf"] for p in problems]
    all_prompts = base_prompts + cf_prompts

    print(f"Generating responses for {len(all_prompts)} prompts...")

    # Generate all responses in one batch
    outputs = llm.generate(all_prompts, sampling_params)

    # Split results back into base and cf
    n = len(problems)
    base_outputs = outputs[:n]
    cf_outputs = outputs[n:]

    # Construct results JSON
    results = []
    for i, problem in enumerate(problems):
        base_resp = base_outputs[i].outputs[0].text
        cf_resp = cf_outputs[i].outputs[0].text

        base_pass = check_response(base_resp, problem["base_completion"])
        cf_pass = check_response(cf_resp, problem["cf_completion"])

        result = {
            "base_prompt": problem["base"],
            "base_expected": problem["base_completion"],
            "base_response": base_resp,
            "base_pass_exact": base_pass["exact"],
            "base_pass_fuzzy": base_pass["fuzzy"],
            "cf_prompt": problem["cf"],
            "cf_expected": problem["cf_completion"],
            "cf_response": cf_resp,
            "cf_pass_exact": cf_pass["exact"],
            "cf_pass_fuzzy": cf_pass["fuzzy"],
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
    parser.add_argument("-t", "--temperature", type=float, default=0.6, help="Model temperature")
    parser.add_argument("--input_path", help="Path to the input dataset")
    parser.add_argument("--output_path", help="Path to output dataset")
    parser.add_argument("--trust_remote_code", default=False, help="Trust remote code (required for some models)")
    
    args = parser.parse_args()

    main(
        args.model_name,
        args.temperature,
        args.input_path,
        args.output_path,
        args.trust_remote_code
    )