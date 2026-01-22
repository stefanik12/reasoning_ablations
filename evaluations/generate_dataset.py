import random
from evaluations.counterfactual_transform import CounterfactualTransformer
from tqdm import tqdm
from pathlib import Path
import json
from collections import Counter
from evaluations.developmental_skills import BenchmarkBuilder, BenchmarkSpec

# -------------------------
# Dataset generation
# -------------------------

def generate_dataset(n_samples_per_skill: int = 10000, output_path: str = "data/dataset.json", seed: int = 42):
    rng = random.Random(seed)

    # 1) Build a dataset using your existing BenchmarkBuilder
    spec = BenchmarkSpec(
        seed=seed,        # Put as arg
        n_per_skill={
            "relational_reasoning": n_samples_per_skill,
            "rule_induction": n_samples_per_skill,
            "working_memory_maintenance": n_samples_per_skill,
            "working_memory_manipulation": n_samples_per_skill,
            "quantitative_reasoning": n_samples_per_skill,
            "cognitive_control_inhibition": n_samples_per_skill,
            "symbol_recognition": n_samples_per_skill,
            "vocabulary": n_samples_per_skill,
            "phonological_awareness": n_samples_per_skill,
            "instruction_comprehension": n_samples_per_skill,
            "fine_motor_proxy": n_samples_per_skill,
            # "social_emotional_awareness": n_samples_per_skill,
            "metacognitive_self_estimation": n_samples_per_skill,
        },
        shuffle=False,
    )

    builder = BenchmarkBuilder(spec)
    base_items = builder.generate()

    unique_bases = {}
    for it in base_items:
        unique_bases.setdefault(it["prompt"], it)
    base_items = list(unique_bases.values())

    # 2) Create the counterfactual transformer
    tfm = CounterfactualTransformer()

    # 3) For each base sample, generate ONE counterfactual by applying a randomly chosen applicable shortcut
    counterfactual_items = []
    for item in base_items:
        shortcuts = tfm.applicable_shortcuts(item)
        if not shortcuts:
            # No applicable shortcut for this skill (shouldn't happen with the provided map).
            continue

        chosen = rng.choice(shortcuts)
        cf_item = tfm.transform(item, shortcut=chosen, rng=rng)
        counterfactual_items.append(cf_item)

    # 4) Print base vs counterfactual side-by-side (first few)
    output = []
    for base, cf in tqdm(zip(base_items, counterfactual_items), total=min(len(base_items), len(counterfactual_items)), desc="Generating dataset"):

        # Filter out no cf_edit
        if cf["meta"].get("cf_edit") is not None:
            item = {
                "base": base["prompt"],
                "base_completion": base["completion"],
                "cf": cf["prompt"],
                "cf_completion": cf["completion"],
                "cf_edit": cf["meta"].get("cf_edit"),
                "cf_meta": cf["meta"],
                "skill": base["skill"]
            }
            # Check for duplicates before appending
            if item not in output:
                output.append(item)

    
    # 5) Save output
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    # Print counts per skill
    skill_counts = Counter(item["skill"] for item in output)
    print("\n=== Samples per skill ===")
    for skill, count in sorted(skill_counts.items()):
        print(f"  {skill}: {count}")

    print(f"Saved {len(output)} entries")