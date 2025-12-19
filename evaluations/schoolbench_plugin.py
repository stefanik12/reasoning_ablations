# plugins/schoolbench_plugin.py
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

# SWIFT pluginization: register extra callbacks and load via --external_plugins
from swift.plugin.callback import extra_callbacks

from evaluations.developmental_skills import BenchmarkBuilder, BenchmarkSpec
from evaluations.counterfactual_transform import CounterfactualTransformer, Shortcut

logger = logging.getLogger()

# -------------------------
# Metrics helpers
# -------------------------

def _maybe_wandb_log(args, state, scalars: Dict[str, float]) -> None:
    # Avoid any dependency if wandb isn't installed/active
    try:
        import wandb
    except Exception:
        logger.warning("Not logging to wandb because not installed")
        return

    # Only if Trainer is configured to use wandb
    report_to = getattr(args, "report_to", None) or []
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" not in report_to:
        logger.warning("Not logging to wandb because reporting to wandb is not among --report_to")
        return

    # Only if a run is active (Trainer usually initializes it)
    if wandb.run is None:
        logger.warning("Not logging to wandb because wandb is not initialized")
        return

    # Log at the Trainer step
    logger.warning("Calling wandb.log with %s", scalars)
    wandb.log(scalars, step=int(getattr(state, "global_step", 0)))


@dataclass
class _Agg:
    nll_sum: float = 0.0
    tok: int = 0
    topk: Dict[int, Tuple[int, int]] = None  # k -> (hits, total)

    def __post_init__(self):
        if self.topk is None:
            self.topk = {}


def _finalize(agg: _Agg) -> Dict[str, float]:
    tok = max(agg.tok, 1)
    ppl = float(math.exp(agg.nll_sum / tok))
    out: Dict[str, float] = {"ppl": ppl, "n_tokens": float(agg.tok)}
    for k, (h, t) in agg.topk.items():
        out[f"top{k}_acc"] = float(h / max(t, 1))
    return out


def _maybe_strip_one_space(s: str) -> str:
    return s[1:] if s.startswith(" ") else s


@torch.inference_mode()
def _score_one(
    model,
    tokenizer,
    prompt: str,
    gold: str,
    topk_list: List[int],
    device: torch.device,
) -> Tuple[float, int, Dict[int, Tuple[int, int]]]:
    """
    Gold-conditioned scoring:
      - input = prompt + " " + gold
      - labels masked on prompt tokens
      - NLL + top-k computed only on gold tokens
    """
    gold = _maybe_strip_one_space(gold)
    full = prompt + " " + gold

    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc_full = tokenizer(full, return_tensors="pt", add_special_tokens=False)

    input_ids = enc_full["input_ids"].to(device)
    attn_mask = enc_full.get("attention_mask", torch.ones_like(input_ids)).to(device)

    prompt_len = enc_prompt["input_ids"].shape[1]
    T = input_ids.shape[1]
    if T <= prompt_len:
        return 0.0, 0, {k: (0, 0) for k in topk_list}

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits  # [1,T,V]

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())

    mask = (shift_labels != -100)
    nll_sum = float((loss * mask).sum().item())
    tok = int(mask.sum().item())

    topk_hits: Dict[int, Tuple[int, int]] = {}
    for k in topk_list:
        if tok == 0:
            topk_hits[k] = (0, 0)
            continue
        topk = torch.topk(shift_logits, k=k, dim=-1).indices  # [1,T-1,k]
        hits = ((topk == shift_labels.unsqueeze(-1)) & mask.unsqueeze(-1)).any(dim=-1).sum().item()
        topk_hits[k] = (int(hits), tok)

    return nll_sum, tok, topk_hits


def _add(aggs: Dict[str, _Agg], key: str, nll_sum: float, tok: int, topk_hits: Dict[int, Tuple[int, int]]):
    a = aggs.setdefault(key, _Agg())
    a.nll_sum += nll_sum
    a.tok += tok
    for k, (h, t) in topk_hits.items():
        hh, tt = a.topk.get(k, (0, 0))
        a.topk[k] = (hh + h, tt + t)


# -------------------------
# Dataset building helpers
# -------------------------

def _build_benchmark_items(model_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Builds base benchmark items directly from your BenchmarkBuilder.
    Config is taken from args.model_kwargs.
    """
    # Minimal defaults; override via --model_kwargs
    seed = int(model_kwargs.get("schoolbench_seed", 42))
    shuffle = bool(model_kwargs.get("schoolbench_shuffle", False))

    # Default: 1 item per skill unless user sets counts
    n_per_skill = model_kwargs.get("schoolbench_n_per_skill", None)
    if isinstance(n_per_skill, str):
        n_per_skill = json.loads(n_per_skill)

    spec = BenchmarkSpec(
        seed=seed,
        n_per_skill=n_per_skill,
        shuffle=shuffle,
        overrides=model_kwargs.get("schoolbench_overrides", None),
    )
    builder = BenchmarkBuilder(spec)
    return builder.generate()


def _build_counterfactual_items(
    base_items: List[Dict[str, Any]],
    model_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Builds exactly one counterfactual item per base item, using your CounterfactualTransformer.

    Strategy:
      - choose shortcut per item:
          * "random" (default): pick one applicable shortcut uniformly
          * "fixed": always use a configured shortcut per skill if provided
    """
    tfm = CounterfactualTransformer()

    # determinism: use python random seeded from model_kwargs for CF choices
    import random as pyrandom
    cf_seed = int(model_kwargs.get("schoolbench_cf_seed", 123))
    prng = pyrandom.Random(cf_seed)

    strategy = str(model_kwargs.get("schoolbench_cf_strategy", "random"))
    fixed_map = model_kwargs.get("schoolbench_cf_fixed_map", None)
    if isinstance(fixed_map, str):
        fixed_map = json.loads(fixed_map)
    if fixed_map is None:
        fixed_map = {}

    cf_items: List[Dict[str, Any]] = []
    for it in base_items:
        shortcuts = tfm.applicable_shortcuts(it)
        if not shortcuts:
            continue

        if strategy == "fixed":
            # fixed_map example: {"relational_reasoning":"recency_bias", ...}
            want = fixed_map.get(it["skill"], None)
            if want is None:
                chosen = prng.choice(shortcuts)
            else:
                # allow user to pass either Shortcut.* names or raw string
                chosen = want
        else:
            chosen = prng.choice(shortcuts)

        cf_items.append(tfm.transform(it, shortcut=chosen, rng=prng))

    return cf_items


# -------------------------
# Callback
# -------------------------

class SchoolBenchEvalCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self._cached_base: Optional[List[Dict[str, Any]]] = None
        self._cached_cf: Optional[List[Dict[str, Any]]] = None
        self._cache_key: Optional[str] = None
        self._model = None
        self._tok = None
        logger.warning("SchoolBenchEvalCallback initialized")

    """
    Runs on each HF Trainer evaluation event and computes:
      - PPL (gold-conditioned)
      - top-k next-token accuracy
      - per-skill breakdown
      - base vs counterfactual gaps

    Configure via --model_kwargs JSON:
      {
        "schoolbench_enabled": true,
        "schoolbench_seed": 42,
        "schoolbench_n_per_skill": {"relational_reasoning":50, ...},
        "schoolbench_topk": [1,5,10],
        "schoolbench_cf_strategy": "random" | "fixed",
        "schoolbench_cf_fixed_map": {"relational_reasoning":"recency_bias", ...},
        "schoolbench_write_samples": false
      }
    """

    def on_train_begin(self, args, state, control, **kwargs):
        # Cache once. This is the best event to do it.
        self._model = kwargs.get("model", self._model)
        # Transformers >=4.5x often uses processing_class instead of tokenizer
        self._tok = kwargs.get("processing_class", kwargs.get("tokenizer", self._tok))
        if self._model is None or self._tok is None:
            logger.warning("on_train_begin all kwargs: %s", kwargs)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logger.warning("SchoolBenchEvalCallback on_evaluate called")
        if not getattr(state, "is_world_process_zero", True):
            return

        given_model_kwargs_arg = getattr(args, "model_kwargs", None)
        if given_model_kwargs_arg is not None:
            model_kwargs = json.loads(Path(given_model_kwargs_arg).read_text(encoding="utf-8"))
        else:
            model_kwargs = {}

        model = self._model
        tokenizer = self._tok

        tr = kwargs.get("trainer", None)
        if tr is not None:
            logger.warning("Tokenizer is not none!")
            if self._model is None and getattr(tr, "model", None) is not None:
                model = tr.model
            if self._tokenizer is None and getattr(tr, "tokenizer", None) is not None:
                tokenizer = tr.tokenizer

        if model is None or tokenizer is None:
            # If this happens in your setup, we can instead re-load tokenizer/model here,
            # but most Trainer runs pass them.
            raise ValueError("[schoolbench] skipped: model/tokenizer not provided to callback")

        topk_list = model_kwargs.get("schoolbench_topk", [1, 5, 10])
        if isinstance(topk_list, str):
            topk_list = json.loads(topk_list)
        topk_list = [int(x) for x in topk_list]

        device = next(model.parameters()).device

        # Build a deterministic cache key from all data-affecting params.
        cache_key = json.dumps({
            "schoolbench_seed": int(model_kwargs.get("schoolbench_seed", 42)),
            "schoolbench_shuffle": bool(model_kwargs.get("schoolbench_shuffle", False)),
            "schoolbench_n_per_skill": model_kwargs.get("schoolbench_n_per_skill", None),
            "schoolbench_overrides": model_kwargs.get("schoolbench_overrides", None),
            "schoolbench_cf_seed": int(model_kwargs.get("schoolbench_cf_seed", 123)),
            "schoolbench_cf_strategy": str(model_kwargs.get("schoolbench_cf_strategy", "random")),
            "schoolbench_cf_fixed_map": model_kwargs.get("schoolbench_cf_fixed_map", None),
        }, sort_keys=True, default=str)

        if self._cached_base is None or self._cache_key != cache_key:
            self._cache_key = cache_key
            self._cached_base = _build_benchmark_items(model_kwargs)
            self._cached_cf = _build_counterfactual_items(self._cached_base, model_kwargs)

        base_items = self._cached_base
        cf_items = self._cached_cf

        # 3) Score + aggregate
        aggs: Dict[str, _Agg] = {}
        model.eval()

        def score_and_add(group_prefix: str, items: List[Dict[str, Any]]):
            for it in items:
                nll, tok, topk_hits = _score_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=it["prompt"],
                    gold=it.get("gold", it.get("completion", "")).lstrip(),  # accept either field
                    topk_list=topk_list,
                    device=device,
                )
                skill = it.get("skill", "unknown")
                _add(aggs, "overall", nll, tok, topk_hits)
                _add(aggs, f"{group_prefix}::overall", nll, tok, topk_hits)
                _add(aggs, f"{group_prefix}::skill::{skill}", nll, tok, topk_hits)

                if group_prefix == "cf":
                    sc = it.get("meta", {}).get("cf_shortcut") or it.get("cf_shortcut")
                    if sc:
                        _add(aggs, f"cf::shortcut::{sc}", nll, tok, topk_hits)
                        _add(aggs, f"cf::skill_shortcut::{skill}::{sc}", nll, tok, topk_hits)

        score_and_add("base", base_items)
        score_and_add("cf", cf_items)

        report: Dict[str, Any] = {
            "meta": {
                "global_step": int(getattr(state, "global_step", -1)),
                "output_dir": args.output_dir,
                "topk": topk_list,
                "n_base": len(base_items),
                "n_cf": len(cf_items),
                "cf_strategy": model_kwargs.get("schoolbench_cf_strategy", "random"),
            },
            "metrics": {k: _finalize(v) for k, v in aggs.items()},
        }

        # Robustness summary (base - cf)  # TODO: make this relative?
        gaps: Dict[str, float] = {}
        base_overall = report["metrics"].get("base::overall", {})
        cf_overall = report["metrics"].get("cf::overall", {})
        for k in topk_list:
            a = base_overall.get(f"top{k}_acc")
            b = cf_overall.get(f"top{k}_acc")
            if a is not None and b is not None:
                gaps[f"top{k}_acc_gap_base_minus_cf"] = float(a - b)
        if "ppl" in base_overall and "ppl" in cf_overall:
            gaps["ppl_ratio_cf_over_base"] = float(cf_overall["ppl"] / max(base_overall["ppl"], 1e-12))
        report["metrics"]["robustness_summary"] = gaps

        # Example: prefix your metrics for clarity
        scalars = {}
        for k, v in base_overall.items():
            if isinstance(v, (int, float)):
                scalars[f"schoolbench/base/{k}"] = float(v)
        for k, v in cf_overall.items():
            if isinstance(v, (int, float)):
                scalars[f"schoolbench/cf/{k}"] = float(v)
        for k, v in gaps.items():
            scalars[f"schoolbench/gap/{k}"] = float(v)

        metrics = report["metrics"]
        for key, vals in metrics.items():
            if key.startswith("base::skill::") and "top1_acc" in vals:
                skill = key.split("base::skill::", 1)[1]
                scalars[f"schoolbench/skill_base_top1/{skill}"] = float(vals["top1_acc"])
            if key.startswith("cf::skill::") and "top1_acc" in vals:
                skill = key.split("cf::skill::", 1)[1]
                scalars[f"schoolbench/skill_cf_top1/{skill}"] = float(vals["top1_acc"])

        logger.warning("_maybe_wandb_log called")
        _maybe_wandb_log(args, state, scalars)

        # Write report
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"schoolbench_eval_step{int(getattr(state, 'global_step', 0)):07d}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Optional: dump a small sample for debugging
        if bool(model_kwargs.get("schoolbench_write_samples", False)):
            samp_path = out_dir / f"schoolbench_samples_step{int(getattr(state, 'global_step', 0)):07d}.json"
            samp_path.write_text(json.dumps({
                "base": base_items[:3],
                "cf": cf_items[:3],
            }, indent=2), encoding="utf-8")

        print(f"[schoolbench] step={report['meta']['global_step']} base={base_overall} cf={cf_overall} gaps={gaps}")

# Register callback
extra_callbacks.append(SchoolBenchEvalCallback())
