# plugins/schoolbench_plugin.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F


@dataclass
class _Agg:
    nll_sum: float = 0.0
    tok: int = 0
    topk: Dict[int, Tuple[int, int]] = None

    def __post_init__(self):
        if self.topk is None:
            self.topk = {}

def _finalize(agg: _Agg) -> Dict[str, float]:
    tok = max(agg.tok, 1)
    ppl = float(math.exp(agg.nll_sum / tok))
    out = {"ppl": ppl, "n_tokens": float(agg.tok)}
    for k, (h, t) in agg.topk.items():
        out[f"top{k}_acc"] = float(h / max(t, 1))
    return out

def _add(aggs: Dict[str, _Agg], key: str, nll_sum: float, tok: int, topk_hits: Dict[int, Tuple[int, int]]):
    a = aggs.setdefault(key, _Agg())
    a.nll_sum += nll_sum
    a.tok += tok
    for k, (h, t) in topk_hits.items():
        hh, tt = a.topk.get(k, (0, 0))
        a.topk[k] = (hh + h, tt + t)


@torch.inference_mode()
def score_one(
    model,
    tokenizer,
    prompt: str,
    gold: str,
    topk_list: List[int],
    device: torch.device,
) -> Dict[str, float]:
    """
    Gold-conditioned scoring:
      - input = prompt + " " + gold
      - labels masked on prompt tokens
      - NLL + top-k computed only on gold tokens
    """
    full = prompt.strip() + " " + gold.strip()

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
    labels[:, prompt_len+1:] = -100

    # Use Autocast for compatibility with Apertus/Olmo layers (xIELU etc)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits

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

    pos = mask.nonzero(as_tuple=False)[0]
    tpos = int(pos[1].item())

    expected_id = int(shift_labels[0, tpos].item())
    expected_tok = tokenizer.convert_ids_to_tokens(expected_id)

    most_likely_id = int(shift_logits[0, tpos].argmax(dim=-1).item())
    most_likely_tok = tokenizer.convert_ids_to_tokens(most_likely_id)

    topn: Dict[int, List[str]] = {}
    for n in topk_list:
        if n <= 0:
            continue
        ids = torch.topk(shift_logits[0, tpos], k=n, dim=-1).indices.tolist()
        topn[n] = [tokenizer.convert_ids_to_tokens(i) for i in ids]

    agg = _Agg(nll_sum=nll_sum, tok=tok, topk=topk_hits)
    out: Dict[str, Any] = {"nll_sum": nll_sum, "tok": tok, "topk": topk_hits, "expected": expected_tok, "most_likely": most_likely_tok, "topn": topn}
    out.update(_finalize(agg))
    return out
