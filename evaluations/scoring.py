# plugins/schoolbench_plugin.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F


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


@torch.inference_mode()
def score_one(
    model,
    tokenizer,
    prompt: str,
    gold: str,
    topk_list: List[int],
    device: torch.device,
    mask_chars_only: str = " ,()",
) -> Dict[str, Any]:
    """
    Gold-conditioned scoring:
      - input = prompt + " " + gold
      - labels masked on prompt tokens
      - NLL + top-k computed only on gold tokens (ALL gold tokens)
      - optionally drop label-token positions whose decoded expected token consists only of mask_chars_only
    """
    full = prompt.strip() + " " + gold.strip()

    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc_full = tokenizer(full, return_tensors="pt", add_special_tokens=False)

    input_ids = enc_full["input_ids"].to(device)
    attn_mask = enc_full.get("attention_mask", torch.ones_like(input_ids)).to(device)

    prompt_len = enc_prompt["input_ids"].shape[1]
    T = input_ids.shape[1]
    if T <= prompt_len:
        out: Dict[str, Any] = {"nll": 0.0,
                               "n_tokens": 0,
                               "expected": gold.strip(),
                               "most_likely": "",
                               "topn": {},
                               "topk_hits": {str(k): 0 for k in topk_list},
                               "topk_total": {str(k): 0 for k in topk_list},}

        return out

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    # labels[:, prompt_len+1:] = -100  # aggregating over all label tokens

    # Use Autocast for compatibility with Apertus/Olmo layers (xIELU etc)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())

    mask = (shift_labels != -100)
    idx = mask[0].nonzero(as_tuple=False).squeeze(-1)  # scored positions in shift space

    if idx.numel() == 0:
        out: Dict[str, Any] = {"nll": 0.0, "n_tokens": 0,
                               "expected": gold.strip(), "most_likely": "", "topn": {}}
        for k in topk_list:
            out[f"top{k}_hits"] = 0
            out[f"top{k}_total"] = 0
        return out

    allowed = set(mask_chars_only)

    # Filter out non-informative expected tokens by decoded text
    keep = []
    for p in idx.tolist():
        tid = int(shift_labels[0, p].item())
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s and (set(s) <= allowed):
            continue
        keep.append(p)

    if not keep:
        out: Dict[str, Any] = {"nll": 0.0,
                               "n_tokens": 0,
                               "expected": gold.strip(),
                               "most_likely": "",
                               "topn": {},
                               "topk_hits": {str(k): 0 for k in topk_list},
                               "topk_total": {str(k): 0 for k in topk_list}}
        return out

    keep_idx = torch.tensor(keep, device=shift_labels.device, dtype=torch.long)
    keep_mask = torch.zeros_like(mask)
    keep_mask[0, keep_idx] = True

    nll = float((loss * keep_mask).sum().item())
    n_tokens = int(keep_mask.sum().item())

    # Most-likely sequence across kept label positions
    ml_ids = shift_logits[0, keep_idx].argmax(dim=-1).tolist()
    most_likely_seq = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(i) for i in ml_ids]).strip()

    topn: Dict[str, List[List[str]]] = {}
    topk_hits: Dict[str, int] = {}
    topk_total: Dict[str, int] = {}
    out: Dict[str, Any] = {"nll": nll,
                           "n_tokens": n_tokens,
                           "expected": gold.strip(),
                           "most_likely": most_likely_seq,
                           "topn": topn,
                           "topk_hits": topk_hits,
                           "topk_total": topk_total}
    for k in topk_list:
        if n_tokens == 0:
            topk_hits[str(k)] = 0
            topk_total[str(k)] = 0
            topn[str(k)] = []
            continue

        topk_ids = torch.topk(shift_logits, k=k, dim=-1).indices  # [1,T-1,k]
        hits = ((topk_ids == shift_labels.unsqueeze(-1)) & mask.unsqueeze(-1)).any(dim=-1).sum().item()

        topk_hits[str(k)] = int(hits)
        topk_total[str(k)] = int(n_tokens)

        # Aggregate top-n across ALL scored (gold) positions: shape [n_tokens, k]
        ids = torch.topk(shift_logits[0, idx], k=k, dim=-1).indices.tolist()
        topn[str(k)] = [[tokenizer.convert_ids_to_tokens(i) for i in row] for row in ids]

    return out
