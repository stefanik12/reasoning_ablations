import csv
import logging
import re
from typing import List, Dict, Any, Union
from huggingface_hub import list_repo_refs
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

def get_target_branches(repo_id: str, interval: int, only_final_model_eval: bool = False) -> List[Dict[str, Any]]:
    """Returns sorted list of branches matching step intervals."""
    logger.info(f"Fetching branches from {repo_id}...")
    try:
        refs = list_repo_refs(repo_id)
    except (OSError, ValueError) as e:
        logger.error(f"Error listing refs: {e}")
        return []

    branches = []
    pattern = re.compile(r"(?:.*)?step(\d+)(?:.*)?$")

    for b in refs.branches:
        match = pattern.match(b.name)
        if match:
            step = int(match.group(1))
            if step % interval == 0:
                branches.append({"step": step, "name": b.name})
    if only_final_model_eval:
        return [{"step": 0, "name": "main"}]
    else:
        sorted_branches = sorted(branches, key=lambda x: x["step"], reverse=True)
        logger.info(f"Identified {len(sorted_branches)} target branches.")
        return sorted_branches


def get_processed_steps(csv_path: Union[str, Path]) -> set:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    with open(csv_path, "r", encoding="utf-8") as f:
        return {int(r["step"]) for r in csv.DictReader(f) if r.get("step")}
    
def configure_logging(level=logging.INFO) -> None:
    # Root: keep dependencies quiet unless they warn/error
    root = logging.getLogger()
    if not root.handlers:  # avoid double handlers if called twice
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.WARNING)

    logging.getLogger("evaluations").setLevel(level)
    logging.getLogger("__main__").setLevel(level)
