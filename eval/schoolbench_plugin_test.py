# test_schoolbench_plugin.py
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerControl

# Import the callback class from your plugin file
from eval.schoolbench_plugin import SchoolBenchEvalCallback


@dataclass
class DummyState:
    global_step: int = 123
    is_world_process_zero: bool = True


def main():
    # 1) Load a small model for quick test (swap to your target model if you want)
    model_name = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) Create TrainingArguments with model_kwargs used by our callback
    # NOTE: TrainingArguments accepts `report_to` but not `model_kwargs` by default.
    # In ms-swift, args.model_kwargs exists; here we add it manually.
    args = TrainingArguments(
        output_dir="./tmp_schoolbench_out",
        per_device_eval_batch_size=1,
        do_train=False,
        do_eval=True,
        report_to=[],  # add "wandb" if you want to test W&B logging
    )

    # Add the ms-swift-style model_kwargs attribute expected by the callback
    args.model_kwargs = {
        "schoolbench_enabled": True,
        "schoolbench_seed": 42,
        "schoolbench_cf_seed": 123,
        "schoolbench_topk": [1, 5],
        "schoolbench_shuffle": False,
        # Keep it tiny for a quick run:
        "schoolbench_n_per_skill": {
            "relational_reasoning": 1,
            "rule_induction": 1,
            "working_memory_maintenance": 1,
            "working_memory_manipulation": 1,
            "quantitative_reasoning": 1,
            "cognitive_control_inhibition": 1,
            "symbol_recognition": 1,
            "vocabulary": 1,
            "phonological_awareness": 1,
            "instruction_comprehension": 1,
            "fine_motor_proxy": 1,
            "social_emotional_awareness": 1,
            "metacognitive_self_estimation": 1,
        },
        "schoolbench_cf_strategy": "random",
        "schoolbench_write_samples": True,
    }

    # 3) Build dummy state/control and run callback.on_evaluate()
    cb = SchoolBenchEvalCallback()
    state = DummyState(global_step=123, is_world_process_zero=True)
    control = TrainerControl()

    cb.on_evaluate(
        args=args,
        state=state,     # our DummyState has the fields callback uses
        control=control,
        model=model,     # callback expects kwargs["model"]
        tokenizer=tok,   # callback expects kwargs["tokenizer"]
    )

    print("Done. Check ./tmp_schoolbench_out for:")
    print(" - schoolbench_eval_step0000123.json")
    print(" - schoolbench_samples_step0000123.json (if enabled)")

    # Optional: pretty-print the overall metrics
    with open("./tmp_schoolbench_out/schoolbench_eval_step0000123.json", "r", encoding="utf-8") as f:
        rep = json.load(f)
    print("\nOVERALL:")
    print(json.dumps(rep["metrics"].get("overall", {}), indent=2))
    print("\nBASE overall:")
    print(json.dumps(rep["metrics"].get("base::overall", {}), indent=2))
    print("\nCF overall:")
    print(json.dumps(rep["metrics"].get("cf::overall", {}), indent=2))
    print("\nGAPS:")
    print(json.dumps(rep["metrics"].get("robustness_summary", {}), indent=2))


if __name__ == "__main__":
    main()
