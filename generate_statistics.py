import json
from collections import defaultdict
import argparse
from pathlib import Path

def generate_statistics(input_path: str = "data/results.json"):
    # Load JSON

    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # ==============================================================================
    # OVERALL ACCURACY STATISTICS
    # ==============================================================================
    print("\n" + "="*80)
    print("OVERALL ACCURACY STATISTICS")
    print("="*80 + "\n")

    # Calculate overall accuracy stats
    base_correct_exact = sum(1 for r in results if r["base_pass_exact"])
    cf_correct_exact = sum(1 for r in results if r["cf_pass_exact"])

    print(f"Base exact accuracy: {base_correct_exact}/{len(results)} ({100*base_correct_exact/len(results):.2f}%)")
    print(f"CF exact accuracy: {cf_correct_exact}/{len(results)} ({100*cf_correct_exact/len(results):.2f}%)")
    
    base_correct_fuzzy = sum(1 for r in results if r["base_pass_fuzzy"])
    cf_correct_fuzzy = sum(1 for r in results if r["cf_pass_fuzzy"])

    print(f"Base fuzzy accuracy: {base_correct_fuzzy}/{len(results)} ({100*base_correct_fuzzy/len(results):.2f}%)")
    print(f"CF fuzzy accuracy: {cf_correct_fuzzy}/{len(results)} ({100*cf_correct_fuzzy/len(results):.2f}%)")

    # ==============================================================================
    # Statistics grouped by skill
    # ==============================================================================
    print("\n" + "="*80)
    print("STATISTICS GROUPED BY SKILL")
    print("="*80)

    # Collect stats for skill totals
    skill_stats = defaultdict(lambda: {
        "total": 0,
        "base_exact": 0,
        "cf_exact": 0,
        "base_fuzzy": 0,
        "cf_fuzzy": 0
    })

    # Collect stats for skill + cf_edit breakdown
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
        
        # Skill totals
        skill_stats[skill]["total"] += 1
        skill_stats[skill]["base_exact"] += int(r["base_pass_exact"])
        skill_stats[skill]["cf_exact"] += int(r["cf_pass_exact"])
        skill_stats[skill]["base_fuzzy"] += int(r["base_pass_fuzzy"])
        skill_stats[skill]["cf_fuzzy"] += int(r["cf_pass_fuzzy"])
        
        # Skill + cf_edit breakdown
        skill_edit_stats[skill][cf_edit]["total"] += 1
        skill_edit_stats[skill][cf_edit]["base_exact"] += int(r["base_pass_exact"])
        skill_edit_stats[skill][cf_edit]["cf_exact"] += int(r["cf_pass_exact"])
        skill_edit_stats[skill][cf_edit]["base_fuzzy"] += int(r["base_pass_fuzzy"])
        skill_edit_stats[skill][cf_edit]["cf_fuzzy"] += int(r["cf_pass_fuzzy"])

    for skill in sorted(skill_stats.keys()):
        stats = skill_stats[skill]
        total = stats["total"]
        
        print(f"\n{'='*40}")
        print(f"SKILL: {skill}")
        print(f"{'='*40}")
        
        # Print skill totals
        print(f"\n  TOTAL (n={total}):")
        print(f"    Base Exact: {stats['base_exact']}/{total} ({100*stats['base_exact']/total:.2f}%)")
        print(f"    CF Exact:   {stats['cf_exact']}/{total} ({100*stats['cf_exact']/total:.2f}%)")
        print(f"    Base Fuzzy: {stats['base_fuzzy']}/{total} ({100*stats['base_fuzzy']/total:.2f}%)")
        print(f"    CF Fuzzy:   {stats['cf_fuzzy']}/{total} ({100*stats['cf_fuzzy']/total:.2f}%)")
        
        # Print cf_edit breakdown
        for cf_edit, edit_stats in sorted(skill_edit_stats[skill].items()):
            edit_total = edit_stats["total"]
            print(f"\n  {cf_edit} (n={edit_total}):")
            print(f"    Base Exact: {edit_stats['base_exact']}/{edit_total} ({100*edit_stats['base_exact']/edit_total:.2f}%)")
            print(f"    CF Exact:   {edit_stats['cf_exact']}/{edit_total} ({100*edit_stats['cf_exact']/edit_total:.2f}%)")
            print(f"    Base Fuzzy: {edit_stats['base_fuzzy']}/{edit_total} ({100*edit_stats['base_fuzzy']/edit_total:.2f}%)")
            print(f"    CF Fuzzy:   {edit_stats['cf_fuzzy']}/{edit_total} ({100*edit_stats['cf_fuzzy']/edit_total:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates accuracy statistics")
    parser.add_argument("--input_path", help="Path to the results data")

    args = parser.parse_args()

    generate_statistics(args.input_path)