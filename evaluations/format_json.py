import argparse
import csv
import os
from collections import defaultdict
from typing import List, Dict, Any
import json
from pathlib import Path
from evaluations.scoring import agg_add, agg_new, agg_finalize

def scored_rows_to_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert scored dataset rows (flat base_* / cf_* keys) into prepare_data()-style pairs:
      [{"base": {...}, "cf": {...}}, ...]

    This preserves:
      - base/cf prompt as base["prompt"], cf["prompt"]
      - base/cf expected as base["expected"], cf["expected"]
      - scoring fields (nll, n_tokens, most_likely, topn, topk_hits, topk_total)
      - skill on both base and cf
      - cf_edit moved into cf["meta"]["cf_edit"]
    """
    pairs: List[Dict[str, Any]] = []

    for r in rows:
        skill = r.get("skill", "unknown")

        base = {
            "prompt": r.get("base_prompt", ""),
            "expected": r.get("base_expected"),
            "skill": skill,
            # scoring fields
            "nll": r.get("base_nll"),
            "n_tokens": r.get("base_n_tokens"),
            "most_likely": r.get("base_most_likely"),
            "topn": r.get("base_topn"),
            "topk_hits": r.get("base_topk_hits"),
            "topk_total": r.get("base_topk_total"),
        }

        cf = None
        # Some datasets might omit cf_*; handle gracefully
        if r.get("cf_prompt") is not None:
            cf = {
                "prompt": r.get("cf_prompt", ""),
                "expected": r.get("cf_expected"),
                "skill": skill,
                # scoring fields
                "nll": r.get("cf_nll"),
                "n_tokens": r.get("cf_n_tokens"),
                "most_likely": r.get("cf_most_likely"),
                "topn": r.get("cf_topn"),
                "topk_hits": r.get("cf_topk_hits"),
                "topk_total": r.get("cf_topk_total"),
                # metadata
                "meta": {
                    "cf_edit": r.get("cf_edit"),
                },
            }

        pairs.append({"base": base, "cf": cf})

    return pairs




def evaluate_checkpoint(pairs, topk_list, samples_csv):

    base_agg, cf_agg = agg_new(topk_list), agg_new(topk_list)
    base_ex = cf_ex = 0

    skills = defaultdict(lambda: {
        "base": agg_new(topk_list), "cf": agg_new(topk_list),
        "base_ex": 0, "cf_ex": 0
    })

    writer = None
    file_exists = os.path.exists(samples_csv)

    with open(samples_csv, "a", newline="", encoding="utf-8") as f:

        def write(kind, skill, prompt, gold, pair):
            nonlocal writer
            out = {
                "expected": pair["expected"],
                "nll": pair["nll"],
                "n_tokens":pair["n_tokens"],
                "most_likely":pair["most_likely"],
                "topn":pair["topn"],
                "topk_hits":pair["topk_hits"],
                "topk_total":pair["topk_total"],
             } # LOADED DATA
            
            if writer is None:
                fields = ["skill", "kind", "prompt", "gold", *out.keys()]
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
            row = {"skill": skill, "kind": kind,
                   "prompt": prompt, "gold": gold, **out}
            writer.writerow(row)
            return out

        def acc(kind, skill, prompt, gold, pair):
            nonlocal base_ex, cf_ex
            out = write(kind, skill, prompt, gold, pair)
            if kind == "base":
                agg_add(base_agg, out, topk_list); base_ex += 1
                agg_add(skills[skill]["base"], out, topk_list); skills[skill]["base_ex"] += 1
            else:
                agg_add(cf_agg, out, topk_list); cf_ex += 1
                agg_add(skills[skill]["cf"], out, topk_list); skills[skill]["cf_ex"] += 1

        for p in pairs:
            base, cf = p["base"], p["cf"]
            skill = base.get("skill", "unknown")

            bg = base.get("gold", base.get("completion", base.get("expected", ""))).strip()
            if bg:
                acc("base", skill, base["prompt"], bg, p.get("base"))

            if cf:
                cg = cf.get("gold", cf.get("completion", cf.get("expected", ""))).strip()
                if cg:
                    acc("cf", skill, cf["prompt"], cg, p.get("cf"))

    metrics = {"n_samples": len(pairs),
               "base_n_examples": float(base_ex), 
               "cf_n_examples": float(cf_ex)}
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

def main(model_name: str,
         input_path: str,
         output_dir: str,
         topk: str = None):
    
    output_dir = Path(output_dir)
    topk_list = sorted({int(x) for x in topk.split(",") if x.strip()}) if topk else []
    metrics_csv = "schoolbench_%s_metrics.csv" % model_name.split("/")[-1]
    samples_csv = "schoolbench_%s_samples.csv" % model_name.split("/")[-1]
    
    metrics_csv = output_dir / metrics_csv
    samples_csv = output_dir / samples_csv

    # Load data
    input_path = Path(input_path)    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = scored_rows_to_pairs(data)

    all_skills = sorted({p["base"].get("skill", "unknown") for p in pairs})
    fields = [
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

    m = evaluate_checkpoint(pairs, topk_list, samples_csv)  

    write_header = not os.path.exists(metrics_csv)
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON SkillBench output into required CSV format")
    parser.add_argument("--model_name", type=str, help="Target HuggingFace ID (e.g., 'swiss-ai/Apertus-70B-2509')")
    parser.add_argument("--input_path", type=str, help="Input path to the JSON dataset to load")
    parser.add_argument("--output_path", type=str, default="data", help="Input path to the JSON dataset to load")
    parser.add_argument("--topk", type=str, default="1,10,100", help="Comma-separated top-k list for accuracy (e.g., '1,5,10')")
    args = parser.parse_args()

    main(args.model_name,
         args.input_path,
         args.output_path,
         args.topk)