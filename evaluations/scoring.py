# plugins/schoolbench_plugin.py
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@torch.inference_mode()
def score_one(
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
