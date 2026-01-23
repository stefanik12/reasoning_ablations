import csv
import logging
import re
from typing import List, Dict, Any, Union, Tuple
from huggingface_hub import list_repo_refs
from pathlib import Path
import math
from dataclasses import dataclass
import sys

logger = logging.getLogger(__name__)

def get_target_branches(repo_id: str, interval: int, only_final_model_eval: bool = False) -> List[Dict[str, Any]]:
    """Returns sorted list of branches matching step intervals."""
    if only_final_model_eval:
        logger.info("Selected final model in repo (main)")
        return [{"step": 0, "name": "main"}]
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



@dataclass
class _Agg:
    nll: float = 0.0
    n_tokens: int = 0
    topk: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        if self.topk is None:
            self.topk = {}

def agg_new(topk_list: List[int]) -> _Agg:
    return _Agg(nll=0.0, n_tokens=0, topk={str(k): (0, 0) for k in topk_list})

def agg_add(agg: _Agg, out: Dict[str, Any], topk_list: List[int]) -> None:
    agg.nll += float(out.get("nll", 0.0))
    agg.n_tokens += int(out.get("n_tokens", 0))

    topk_hits = out.get("topk_hits", {}) or {}
    topk_total = out.get("topk_total", {}) or {}

    for k in topk_list:
        h = int(topk_hits.get(str(k), 0))
        t = int(topk_total.get(str(k), 0))
        hh, tt = agg.topk.get(str(k), (0, 0))
        agg.topk[str(k)] = (hh + h, tt + t)

def agg_finalize(agg: _Agg, prefix: str, topk_list: List[int]) -> Dict[str, float]:
    tok = max(agg.n_tokens, 1)
    out: Dict[str, float] = {f"{prefix}_ppl": float(math.exp(agg.nll / tok)),
                             f"{prefix}_n_tokens": float(agg.n_tokens)}
    for k in topk_list:
        h, t = agg.topk.get(str(k), (0, 0))
        out[f"{prefix}_top{k}_acc"] = float(h / max(t, 1))
    return out