import json
from collections import defaultdict
import argparse
from pathlib import Path
import json
from typing import Dict, Any

def generate_statistics(
    input_path: str = "data/results.json",
    output_path: str = "data/statistics_summary.json",
) -> Dict[str, Any]:
    
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    n = len(results)
    if n == 0:
        raise ValueError(f"No results found in {input_path}")

    # ==============================================================================
    # OVERALL ACCURACY STATISTICS
    # ==============================================================================
    base_correct_exact = sum(1 for r in results if r["base_pass_exact"])
    cf_correct_exact   = sum(1 for r in results if r["cf_pass_exact"])
    base_correct_fuzzy = sum(1 for r in results if r["base_pass_fuzzy"])
    cf_correct_fuzzy   = sum(1 for r in results if r["cf_pass_fuzzy"])

    overall = {
        "n": n,
        "base_exact": {"correct": base_correct_exact, "accuracy": base_correct_exact / n},
        "cf_exact":   {"correct": cf_correct_exact,   "accuracy": cf_correct_exact / n},
        "base_fuzzy": {"correct": base_correct_fuzzy, "accuracy": base_correct_fuzzy / n},
        "cf_fuzzy":   {"correct": cf_correct_fuzzy,   "accuracy": cf_correct_fuzzy / n},
    }

    print("\n" + "="*80)
    print("OVERALL ACCURACY STATISTICS")
    print("="*80 + "\n")
    print(f"Base exact accuracy: {base_correct_exact}/{n} ({100*overall['base_exact']['accuracy']:.2f}%)")
    print(f"CF exact accuracy:   {cf_correct_exact}/{n} ({100*overall['cf_exact']['accuracy']:.2f}%)")
    print(f"Base fuzzy accuracy: {base_correct_fuzzy}/{n} ({100*overall['base_fuzzy']['accuracy']:.2f}%)")
    print(f"CF fuzzy accuracy:   {cf_correct_fuzzy}/{n} ({100*overall['cf_fuzzy']['accuracy']:.2f}%)")

    # ==============================================================================
    # Statistics grouped by skill and cf_edit
    # ==============================================================================
    skill_stats = defaultdict(lambda: {
        "total": 0,
        "base_exact": 0,
        "cf_exact": 0,
        "base_fuzzy": 0,
        "cf_fuzzy": 0
    })

    skill_edit_stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0,
        "base_exact": 0,
        "cf_exact": 0,
        "base_fuzzy": 0,
        "cf_fuzzy": 0
    }))

    for r in results:
        skill = r["skill"]
        cf_edit = r["cf_edit"]

        skill_stats[skill]["total"] += 1
        skill_stats[skill]["base_exact"] += int(r["base_pass_exact"])
        skill_stats[skill]["cf_exact"] += int(r["cf_pass_exact"])
        skill_stats[skill]["base_fuzzy"] += int(r["base_pass_fuzzy"])
        skill_stats[skill]["cf_fuzzy"] += int(r["cf_pass_fuzzy"])

        skill_edit_stats[skill][cf_edit]["total"] += 1
        skill_edit_stats[skill][cf_edit]["base_exact"] += int(r["base_pass_exact"])
        skill_edit_stats[skill][cf_edit]["cf_exact"] += int(r["cf_pass_exact"])
        skill_edit_stats[skill][cf_edit]["base_fuzzy"] += int(r["base_pass_fuzzy"])
        skill_edit_stats[skill][cf_edit]["cf_fuzzy"] += int(r["cf_pass_fuzzy"])

    # Build JSON-friendly output (regular dicts + computed accuracies)
    by_skill: Dict[str, Any] = {}
    for skill in sorted(skill_stats.keys()):
        stats = skill_stats[skill]
        total = stats["total"]

        skill_entry = {
            "total": total,
            "base_exact": {"correct": stats["base_exact"], "accuracy": stats["base_exact"] / total},
            "cf_exact":   {"correct": stats["cf_exact"],   "accuracy": stats["cf_exact"] / total},
            "base_fuzzy": {"correct": stats["base_fuzzy"], "accuracy": stats["base_fuzzy"] / total},
            "cf_fuzzy":   {"correct": stats["cf_fuzzy"],   "accuracy": stats["cf_fuzzy"] / total},
            "by_cf_edit": {}
        }

        for cf_edit, edit_stats in sorted(skill_edit_stats[skill].items()):
            edit_total = edit_stats["total"]
            skill_entry["by_cf_edit"][cf_edit] = {
                "total": edit_total,
                "base_exact": {"correct": edit_stats["base_exact"], "accuracy": edit_stats["base_exact"] / edit_total},
                "cf_exact":   {"correct": edit_stats["cf_exact"],   "accuracy": edit_stats["cf_exact"] / edit_total},
                "base_fuzzy": {"correct": edit_stats["base_fuzzy"], "accuracy": edit_stats["base_fuzzy"] / edit_total},
                "cf_fuzzy":   {"correct": edit_stats["cf_fuzzy"],   "accuracy": edit_stats["cf_fuzzy"] / edit_total},
            }

        by_skill[skill] = skill_entry

        # Keep your existing prints
        print("\n" + "="*80)
        print("STATISTICS GROUPED BY SKILL")
        print("="*80)
        break  # remove this break if you want the header once before printing all skills

    # If you want to keep the exact same printed per-skill section you had,
    # you can print using `by_skill` here; I'm leaving your print style intact below.
    for skill in sorted(skill_stats.keys()):
        stats = skill_stats[skill]
        total = stats["total"]

        print(f"\n{'='*40}")
        print(f"SKILL: {skill}")
        print(f"{'='*40}")

        print(f"\n  TOTAL (n={total}):")
        print(f"    Base Exact: {stats['base_exact']}/{total} ({100*stats['base_exact']/total:.2f}%)")
        print(f"    CF Exact:   {stats['cf_exact']}/{total} ({100*stats['cf_exact']/total:.2f}%)")
        print(f"    Base Fuzzy: {stats['base_fuzzy']}/{total} ({100*stats['base_fuzzy']/total:.2f}%)")
        print(f"    CF Fuzzy:   {stats['cf_fuzzy']}/{total} ({100*stats['cf_fuzzy']/total:.2f}%)")

        for cf_edit, edit_stats in sorted(skill_edit_stats[skill].items()):
            edit_total = edit_stats["total"]
            print(f"\n  {cf_edit} (n={edit_total}):")
            print(f"    Base Exact: {edit_stats['base_exact']}/{edit_total} ({100*edit_stats['base_exact']/edit_total:.2f}%)")
            print(f"    CF Exact:   {edit_stats['cf_exact']}/{edit_total} ({100*edit_stats['cf_exact']/edit_total:.2f}%)")
            print(f"    Base Fuzzy: {edit_stats['base_fuzzy']}/{edit_total} ({100*edit_stats['base_fuzzy']/edit_total:.2f}%)")
            print(f"    CF Fuzzy:   {edit_stats['cf_fuzzy']}/{edit_total} ({100*edit_stats['cf_fuzzy']/edit_total:.2f}%)")

    summary = {
        "input_path": str(input_path),
        "overall": overall,
        "by_skill": by_skill,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved statistics JSON to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates accuracy statistics")
    parser.add_argument("--input_path", help="Path to the results data")
    parser.add_argument("--output_path", help="Path to the output statistics")

    args = parser.parse_args()

    generate_statistics(args.input_path,
                        args.output_path)