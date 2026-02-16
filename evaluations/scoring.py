from __future__ import annotations
from typing import Dict, List, Any

import torch
import torch.nn.functional as F

@torch.inference_mode()
def score_one(
    model,
    tokenizer,
    prompt: str,
    gold: str,
    device: torch.device,
    label_tokens: List[str],
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
        raise ValueError("No label tokens found (prompt length >= total input length)")

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    # labels[:, prompt_len+1:] = -100  # aggregating over all label tokens

    # Use Autocast for compatibility with Apertus/Olmo layers (xIELU etc)
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    else:
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
                               "expected": gold.strip(), "most_likely": "",
                               "top1_hits": 0,
                               "top1_total": 0,
                               "top1_correct": False
                               }
        return out

    allowed = set(mask_chars_only)

    # Filter out non-informative expected tokens by decoded text
    keep = []
    for p in idx:
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
                               "top1_hits": 0,
                               "top1_total": 0,
                               "top1_correct": False}
        return out

    keep_idx = torch.tensor(keep, device=shift_labels.device, dtype=torch.long)
    keep_mask = torch.zeros_like(mask)
    keep_mask[0, keep_idx] = True

    nll = float((loss * keep_mask).sum().item())
    n_tokens = int(keep_mask.sum().item())

    # Most-likely sequence across kept label positions (top-1 / argmax)
    pred_ids = shift_logits[0, keep_idx].argmax(dim=-1)

    gold_ids = shift_labels[0, keep_idx]
    expected_seq = tokenizer.batch_decode(gold_ids, clean_up_tokenization_spaces=False)
    true_seq = tokenizer.batch_decode(pred_ids, clean_up_tokenization_spaces=False)

    # Top-1 aggregation on kept positions only
    top1_total = int(n_tokens)

    assert top1_total > 0, "Less than 1 expected (label) token in sample with gold '%s' prompt '%s'." % (gold, prompt)
    # assess the match token-elementwise with stripped spaces
    top1_hits = sum(exp_one.strip() == pred_one.strip() for exp_one, pred_one in zip(expected_seq, true_seq))
    top1_correct = top1_hits == top1_total

    out: Dict[str, Any] = {"nll": nll,
                           "n_tokens": n_tokens,
                           "expected": expected_seq,
                           "most_likely": true_seq,
                           "top1_hits": top1_hits,
                           "top1_total": top1_total,
                           "top1_correct": top1_correct}
    return out
