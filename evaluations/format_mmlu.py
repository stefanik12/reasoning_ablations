import argparse
import csv
import logging
import os
import json
from pathlib import Path


def extract_metrics(results_dict: dict) -> dict:
    """
    Flattens lm_eval results into a single dictionary mapping Task -> Score.
    Prioritizes 'acc,none', then 'acc', then 'acc_norm'.
    """
    flat_metrics = {}
    for task_name, metrics in results_dict.items():
        # Handle different metric names depending on the task
        # MMLU typically uses 'acc' or 'acc,none'
        score = (metrics.get("acc,none")
                 or metrics.get("acc")
                 or metrics.get("acc_norm,none")
                 or metrics.get("acc_norm")
                 or 0.0)
        flat_metrics[task_name] = score
    return flat_metrics

def main(model_name, input_path, output_dir):
    logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = "lmeval_%s.csv" % model_name.split("/")[-1]
    results_csv = output_dir / results_csv

    # We do NOT write the header yet. We wait for the first result to determine
    # the full list of decomposed subtasks (e.g. mmlu_abstract_algebra, etc).


    try:
        
        # Run LM Eval
        # Note: When 'mmlu' is passed, lm_eval runs all subtasks.
        with open(input_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # --- Process Results ---
        # 'results["results"]' contains keys for the group AND all subtasks
        res_dict = results["results"]
        metric_row = extract_metrics(res_dict)

        # Prepare the full row data
        row_data = {}
        row_data.update(metric_row)

        # --- CSV Handling ---
        file_exists = os.path.exists(results_csv)

        # Determine fieldnames (columns)
        # If file exists, use existing columns. If new, generate from this first run.
        if file_exists:
            with open(results_csv, "r", encoding="utf-8") as f:
                fieldnames = csv.DictReader(f).fieldnames
        else:
            # Create columns: step, branch, then sorted task names
            task_cols = sorted(metric_row.keys())
            fieldnames = task_cols

        # Write to CSV
        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

        # Calculate average for logging purposes (simple mean of all columns found)
        avg_score = sum(metric_row.values()) / len(metric_row) if metric_row else 0
        logger.info(f"Approx Avg: {avg_score:.4f}")

    except Exception as e:
        logger.error(f"Failed to evaluate: {e}")
        # Optional: Log the full traceback if needed
        # import traceback; traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert saved MMLU JSON to CSV")
    parser.add_argument("--model_name", type=str, required=True, help="Target HuggingFace repository ID")
    parser.add_argument("--input_path", type=str, required=True, help="Path to JSON file to convert to CSV")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to output the CSV")

    args = parser.parse_args()
    main(args.model_name,
         args.input_path,
         args.output_dir)