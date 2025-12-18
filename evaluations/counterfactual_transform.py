from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Callable, Optional
import random
import re
import hashlib




# -------------------------
# Shortcut identifiers
# -------------------------

class Shortcut:
    FIRST_OPTION_BIAS = "first_option_bias"
    LABEL_RENAMING_INVARIANCE = "label_renaming_invariance"
    RECENCY_BIAS = "recency_bias"
    LAST_TOKEN_HEURISTIC = "last_token_heuristic"
    LENGTH_HEURISTIC = "length_heuristic"
    FORMAT_KEYWORD_TRIGGER = "format_keyword_trigger"


SKILL_TO_SHORTCUTS: Dict[str, List[str]] = {
    "relational_reasoning": [
        Shortcut.RECENCY_BIAS, Shortcut.LAST_TOKEN_HEURISTIC, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "rule_induction": [
        Shortcut.RECENCY_BIAS, Shortcut.LAST_TOKEN_HEURISTIC, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "working_memory_maintenance": [
        Shortcut.LAST_TOKEN_HEURISTIC, Shortcut.RECENCY_BIAS, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "working_memory_manipulation": [
        Shortcut.RECENCY_BIAS, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "quantitative_reasoning": [
        Shortcut.LENGTH_HEURISTIC
    ],
    "cognitive_control_inhibition": [
        Shortcut.RECENCY_BIAS, Shortcut.LAST_TOKEN_HEURISTIC, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "symbol_recognition": [
        Shortcut.LABEL_RENAMING_INVARIANCE, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "vocabulary": [
        Shortcut.FIRST_OPTION_BIAS, Shortcut.LABEL_RENAMING_INVARIANCE
    ],
    "phonological_awareness": [
        Shortcut.FIRST_OPTION_BIAS, Shortcut.LABEL_RENAMING_INVARIANCE, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "instruction_comprehension": [
        Shortcut.LABEL_RENAMING_INVARIANCE, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "fine_motor_proxy": [
        Shortcut.RECENCY_BIAS, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "social_emotional_awareness": [
        Shortcut.LABEL_RENAMING_INVARIANCE, Shortcut.RECENCY_BIAS
    ],
    "metacognitive_self_estimation": [
        Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
}


# -------------------------
# Utility helpers
# -------------------------

def _stable_cf_id(from_id: str, shortcut: str, prompt: str, completion: str) -> str:
    """Deterministic 32-hex id for a counterfactual item."""
    h = hashlib.sha1((from_id + "|" + shortcut + "\n" + prompt + "\n" + completion).encode("utf-8")).hexdigest()
    return h[:32]

def _clone_item(it: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": it.get("id",""),
        "skill": it["skill"],
        "prompt": it["prompt"],
        "completion": it["completion"],
        "meta": dict(it.get("meta", {})),
    }

def _gold(it: Dict[str, Any]) -> str:
    return it["completion"].lstrip()

def _set_gold(it: Dict[str, Any], new_gold: str) -> None:
    it["completion"] = " " + new_gold

def _assert_prompt_format(prompt: str) -> None:
    if not prompt.endswith("A:"):
        raise ValueError("Expected prompt to end with 'A:'")

def _swap_tokens(prompt: str, a: str, b: str) -> str:
    # safe swap with placeholders
    ph = "__SWAP__"
    prompt = prompt.replace(a, ph)
    prompt = prompt.replace(b, a)
    prompt = prompt.replace(ph, b)
    return prompt


# -------------------------
# Core transformer
# -------------------------

@dataclass
class CounterfactualTransformer:
    """
    Transforms an existing benchmark sample into a counterfactual form by applying
    ONE chosen shortcut transformation that is applicable to that sample's skill.

    - Keeps the *original prompt format* (no QA-shell, no additional sections).
    - Tries hard to preserve the underlying task semantics.
    - Updates the gold completion when the transformation renames labels/options.
    """

    def applicable_shortcuts(self, item: Dict[str, Any]) -> List[str]:
        return SKILL_TO_SHORTCUTS.get(item["skill"], [])

    def transform(self, item: Dict[str, Any], shortcut: str, rng: random.Random) -> Dict[str, Any]:
        _assert_prompt_format(item["prompt"])
        if shortcut not in self.applicable_shortcuts(item):
            raise ValueError(f"Shortcut '{shortcut}' not applicable to skill '{item['skill']}'")

        fn = self._dispatch(item["skill"], shortcut)
        if fn is None:
            raise ValueError(f"No implementation for {item['skill']} × {shortcut}")

        out = _clone_item(item)
        old_id = item["id"]
        out = fn(out, rng)

        # Deterministic counterfactual id (stable across runs)
        out["id"] = _stable_cf_id(old_id, shortcut, out["prompt"], out["completion"])

        out["meta"].update({
            "counterfactual": True,
            "cf_from_id": old_id,
            "cf_shortcut": shortcut,
        })
        return out

    # -------------------------
    # Dispatch
    # -------------------------

    def _dispatch(self, skill: str, shortcut: str) -> Optional[Callable[[Dict[str, Any], random.Random], Dict[str, Any]]]:
        key = (skill, shortcut)
        return {
            # relational_reasoning
            ("relational_reasoning", Shortcut.RECENCY_BIAS): self._rr_reorder_facts,
            ("relational_reasoning", Shortcut.LAST_TOKEN_HEURISTIC): self._rr_add_trailing_distractor_symbol,
            ("relational_reasoning", Shortcut.FORMAT_KEYWORD_TRIGGER): self._rr_rename_minmax_token,

            # rule_induction
            ("rule_induction", Shortcut.RECENCY_BIAS): self._ri_shuffle_examples,
            ("rule_induction", Shortcut.LAST_TOKEN_HEURISTIC): self._ri_add_trailing_equation_noise,
            ("rule_induction", Shortcut.FORMAT_KEYWORD_TRIGGER): self._ri_rename_ex_q_tokens,

            # WM maint/manip
            ("working_memory_maintenance", Shortcut.LAST_TOKEN_HEURISTIC): self._wm_maint_make_last_token_wrong,
            ("working_memory_maintenance", Shortcut.RECENCY_BIAS): self._wm_maint_rotate_sequence,
            ("working_memory_maintenance", Shortcut.FORMAT_KEYWORD_TRIGGER): self._wm_maint_rename_kth_token,

            ("working_memory_manipulation", Shortcut.RECENCY_BIAS): self._wm_manip_insert_irrelevant_prefix,
            ("working_memory_manipulation", Shortcut.FORMAT_KEYWORD_TRIGGER): self._wm_manip_rename_rev_token,

            # quantitative
            ("quantitative_reasoning", Shortcut.LENGTH_HEURISTIC): self._quant_break_length_proxy,

            # inhibition
            ("cognitive_control_inhibition", Shortcut.RECENCY_BIAS): self._inh_shuffle_mapping_order,
            ("cognitive_control_inhibition", Shortcut.LAST_TOKEN_HEURISTIC): self._inh_append_query_distractor,
            ("cognitive_control_inhibition", Shortcut.FORMAT_KEYWORD_TRIGGER): self._inh_rename_map_token,

            # symbol recognition
            ("symbol_recognition", Shortcut.LABEL_RENAMING_INVARIANCE): self._symrec_swap_yes_no,
            ("symbol_recognition", Shortcut.FORMAT_KEYWORD_TRIGGER): self._symrec_rename_member_token,

            # vocabulary (MC)
            ("vocabulary", Shortcut.FIRST_OPTION_BIAS): self._mc_shuffle_options_letters,
            ("vocabulary", Shortcut.LABEL_RENAMING_INVARIANCE): self._mc_permute_option_letters_only,

            # phonological (MC)
            ("phonological_awareness", Shortcut.FIRST_OPTION_BIAS): self._mc_shuffle_options_letters,
            ("phonological_awareness", Shortcut.LABEL_RENAMING_INVARIANCE): self._mc_permute_option_letters_only,
            ("phonological_awareness", Shortcut.FORMAT_KEYWORD_TRIGGER): self._phon_rename_rhyme_token,

            # instruction comprehension
            ("instruction_comprehension", Shortcut.LABEL_RENAMING_INVARIANCE): self._instr_swap_yes_no,
            ("instruction_comprehension", Shortcut.FORMAT_KEYWORD_TRIGGER): self._instr_rename_rule_token,

            # fine motor proxy
            ("fine_motor_proxy", Shortcut.RECENCY_BIAS): self._motor_shuffle_moves_in_neutral_way,
            ("fine_motor_proxy", Shortcut.FORMAT_KEYWORD_TRIGGER): self._motor_rename_move_token,

            # social-emotional
            ("social_emotional_awareness", Shortcut.LABEL_RENAMING_INVARIANCE): self._soc_rename_scale_labels,
            ("social_emotional_awareness", Shortcut.RECENCY_BIAS): self._soc_shuffle_scale_def_order,

            # metacognition
            ("metacognitive_self_estimation", Shortcut.FORMAT_KEYWORD_TRIGGER): self._meta_rename_deviation_token,
        }.get(key)

    # -------------------------
    # Transform implementations
    # -------------------------

    # --- Relational reasoning ---

    def _rr_reorder_facts(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes recency bias: reorder comparison facts; semantics unchanged.
        lines = it["prompt"].splitlines()
        # facts are lines until the "Q:" line
        q_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Q:"))
        facts = lines[:q_idx]
        rest = lines[q_idx:]
        if len(facts) >= 2:
            facts = facts[::-1]
        it["prompt"] = "\n".join(facts + rest)
        it["meta"]["cf_edit"] = "reversed_fact_order"
        return it

    def _rr_add_trailing_distractor_symbol(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes last-token heuristic: append an irrelevant trailing token near the query.
        # Keep query meaning the same.
        prompt = it["prompt"]
        gold = _gold(it)
        distract = "Z" if gold != "Z" else "Y"
        it["prompt"] = prompt.replace("A:", f"NOTE:{distract}\nA:")
        it["meta"]["cf_edit"] = "added_trailing_distractor_token"
        return it

    def _rr_rename_minmax_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes keyword-trigger reliance: rename 'min'/'max' token while preserving meaning.
        # We keep same structure but change the keyword.
        prompt = it["prompt"]
        if "min(" in prompt:
            it["prompt"] = prompt.replace("min(", "smallest(")
            it["meta"]["cf_edit"] = "renamed_min_to_smallest"
        elif "max(" in prompt:
            it["prompt"] = prompt.replace("max(", "largest(")
            it["meta"]["cf_edit"] = "renamed_max_to_largest"
        else:
            # If not present, leave unchanged (shouldn't happen if format is consistent).
            it["meta"]["cf_edit"] = "noop_no_minmax"
        return it

    # --- Rule induction ---

    def _ri_shuffle_examples(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes recency bias: shuffle EX example lines; the rule is invariant to example order.
        lines = it["prompt"].splitlines()
        # Find the block after "EX:" until "Q:"
        ex_start = next(i for i, ln in enumerate(lines) if ln.strip() == "EX:")
        q_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Q:"))
        ex_lines = lines[ex_start + 1:q_idx]
        rng.shuffle(ex_lines)
        it["prompt"] = "\n".join(lines[:ex_start + 1] + ex_lines + lines[q_idx:])
        it["meta"]["cf_edit"] = "shuffled_example_order"
        return it

    def _ri_add_trailing_equation_noise(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes last-token heuristic: append a distractor equation after examples but before Q.
        lines = it["prompt"].splitlines()
        q_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Q:"))
        noise = "NOISE: P+Q=R"
        it["prompt"] = "\n".join(lines[:q_idx] + [noise] + lines[q_idx:])
        it["meta"]["cf_edit"] = "inserted_noise_equation"
        return it

    def _ri_rename_ex_q_tokens(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes keyword-trigger: rename 'EX:'/'Q:' markers while keeping structure.
        it["prompt"] = it["prompt"].replace("EX:", "SAMPLES:").replace("\nQ:", "\nQUERY:")
        it["meta"]["cf_edit"] = "renamed_markers_EX_Q"
        return it

    # --- WM maintenance ---

    def _wm_maint_make_last_token_wrong(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes last-token heuristic: ensure the last token in SEQ is NOT the answer,
        # by swapping last token with another token if needed (keeping asked index same).
        prompt = it["prompt"]
        gold = _gold(it)

        m = re.search(r"^SEQ:\s*(.+)\nQ:\s*(\d+)th=", prompt, flags=re.M)
        if not m:
            return it
        seq_str, k_str = m.group(1), m.group(2)
        seq = seq_str.split()
        k = int(k_str) - 1
        if not (0 <= k < len(seq)):
            return it

        # If last token equals gold, swap last with some other non-gold token.
        if seq[-1] == gold:
            # find an index != k whose token != gold to swap with last
            candidates = [i for i, t in enumerate(seq[:-1]) if t != gold]
            if candidates:
                i = candidates[0]
                seq[-1], seq[i] = seq[i], seq[-1]
                # update prompt only; gold unchanged
                new_seq_str = " ".join(seq)
                it["prompt"] = re.sub(r"^SEQ:\s*.+$", f"SEQ: {new_seq_str}", prompt, flags=re.M)
                it["meta"]["cf_edit"] = "swapped_last_token_away_from_gold"
        return it

    def _wm_maint_rotate_sequence(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes recency/position: rotate the displayed sequence but keep the asked index referring
        # to the *same symbol* by adjusting ask_index in the prompt and gold accordingly.
        prompt = it["prompt"]
        m = re.search(r"^SEQ:\s*(.+)\nQ:\s*(\d+)th=", prompt, flags=re.M)
        if not m:
            return it
        seq = m.group(1).split()
        k = int(m.group(2))  # 1-indexed
        gold = _gold(it)

        # Rotate by r
        r = rng.randint(1, len(seq) - 1) if len(seq) > 1 else 0
        rotated = seq[r:] + seq[:r]

        # Find where the original gold ended up; set new k so answer is still gold.
        # (This keeps the underlying memory retrieval semantics constant.)
        new_k = rotated.index(gold) + 1

        it["prompt"] = re.sub(r"^SEQ:\s*.+$", f"SEQ: {' '.join(rotated)}", prompt, flags=re.M)
        it["prompt"] = re.sub(r"\nQ:\s*\d+th=", f"\nQ: {new_k}th=", it["prompt"])
        _set_gold(it, gold)
        it["meta"]["cf_edit"] = "rotated_seq_and_adjusted_k"
        return it

    def _wm_maint_rename_kth_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes keyword-trigger on "kth": rename the query marker while preserving meaning.
        it["prompt"] = it["prompt"].replace("th=", "pos=")
        it["meta"]["cf_edit"] = "renamed_kth_to_pos"
        return it

    # --- WM manipulation ---

    def _wm_manip_insert_irrelevant_prefix(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes recency: add an irrelevant prefix token that should be ignored
        # (still in same format: stays on SEQ line).
        prompt = it["prompt"]
        m = re.search(r"^SEQ:\s*(.+)\nQ:\s*REV=", prompt, flags=re.M)
        if not m:
            return it
        seq = m.group(1).split()
        # Add a neutral marker at the beginning that is explicitly "IGNORE"
        seq2 = ["IGNORE"] + seq
        # Gold should remain reverse of original seq (not including IGNORE).
        gold = " ".join(reversed(seq))
        it["prompt"] = re.sub(r"^SEQ:\s*.+$", f"SEQ: {' '.join(seq2)}", prompt, flags=re.M)
        _set_gold(it, gold)
        it["meta"]["cf_edit"] = "added_ignore_prefix_not_part_of_task"
        return it

    def _wm_manip_rename_rev_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        it["prompt"] = it["prompt"].replace("Q: REV=", "Q: BACK=")
        it["meta"]["cf_edit"] = "renamed_REV_to_BACK"
        return it

    # --- Quantitative reasoning ---

    def _quant_break_length_proxy(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes length_heuristic: keep counts identical but render each dot as a 2-character token
        # separated by spaces, breaking “string length” heuristics and pushing toward counting.
        #
        # Original format:
        # X: ●●●
        # Y: ●●
        # We change to:
        # X: ● ● ●
        # Y: ● ●
        prompt = it["prompt"]
        lines = prompt.splitlines()
        if len(lines) < 3:
            return it

        def explode(line: str) -> str:
            m = re.match(r"^(\w+):\s*(.+)$", line)
            if not m:
                return line
            label, dots = m.group(1), m.group(2).strip()
            # If it's already spaced, do nothing.
            if " " in dots:
                return line
            # Split into characters (works for "●" repeated, or any single-char token)
            chars = list(dots)
            return f"{label}: " + " ".join(chars)

        lines[0] = explode(lines[0])
        lines[1] = explode(lines[1])
        it["prompt"] = "\n".join(lines)
        it["meta"]["cf_edit"] = "spaced_quantity_tokens_to_break_length_proxy"
        return it

    # --- Inhibition ---

    def _inh_shuffle_mapping_order(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes recency: reorder MAP entries; the mapping is set-valued, order-irrelevant.
        prompt = it["prompt"]
        m = re.search(r"^MAP:\s*(.+)\nQ:\s*(\S+)\nA:", prompt, flags=re.M)
        if not m:
            return it
        pairs_str = m.group(1)
        pairs = [p.strip() for p in pairs_str.split(",")]
        rng.shuffle(pairs)
        new_pairs_str = ", ".join(pairs)
        it["prompt"] = prompt.replace(pairs_str, new_pairs_str)
        it["meta"]["cf_edit"] = "shuffled_map_pairs"
        return it

    def _inh_append_query_distractor(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes last-token heuristic: append an irrelevant token after Q line but before A:.
        # Same format (still lines), no extra sections.
        prompt = it["prompt"].splitlines()
        for i, ln in enumerate(prompt):
            if ln.startswith("Q:"):
                prompt[i] = ln + " / IGNORE"
        it["prompt"] = "\n".join(prompt)
        it["meta"]["cf_edit"] = "added_ignore_token_on_query_line"
        return it

    def _inh_rename_map_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        it["prompt"] = it["prompt"].replace("MAP:", "RULEMAP:")
        it["meta"]["cf_edit"] = "renamed_MAP_marker"
        return it

    # --- Symbol recognition ---

    def _symrec_swap_yes_no(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes label-meaning attachment: swap surface tokens YES/NO everywhere and flip gold.
        gold = _gold(it)
        if gold not in ("YES", "NO"):
            return it
        it["prompt"] = _swap_tokens(it["prompt"], "YES", "NO")
        _set_gold(it, "NO" if gold == "YES" else "YES")
        it["meta"]["cf_edit"] = "swapped_yes_no_tokens_and_flipped_gold"
        return it

    def _symrec_rename_member_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        it["prompt"] = it["prompt"].replace("member(", "inset(")
        it["meta"]["cf_edit"] = "renamed_member_to_inset"
        return it

    # --- Multiple choice (Vocabulary, Phonology) ---

    def _parse_mc_options(self, prompt: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Parses options of the form 'A:foo B:bar C:baz ...' on a single line.
        Returns (letters_in_order, letter->option_text).
        """
        # Find the first line containing 'A:' ... pattern.
        lines = prompt.splitlines()
        opt_line_idx = None
        for i, ln in enumerate(lines):
            if re.search(r"\bA:\S+", ln) and re.search(r"\bB:\S+", ln):
                opt_line_idx = i
                break
        if opt_line_idx is None:
            raise ValueError("No MC option line found")

        ln = lines[opt_line_idx]
        pairs = re.findall(r"\b([A-D]):([^\s]+)", ln)
        letters = [p[0] for p in pairs]
        mapping = {L: txt for (L, txt) in pairs}
        return letters, mapping

    def _rewrite_mc_line(self, prompt: str, new_pairs: List[Tuple[str, str]]) -> str:
        lines = prompt.splitlines()
        for i, ln in enumerate(lines):
            if re.search(r"\bA:\S+", ln) and re.search(r"\bB:\S+", ln):
                lines[i] = " ".join([f"{L}:{t}" for L, t in new_pairs])
                return "\n".join(lines)
        return prompt

    def _mc_shuffle_options_letters(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes first-option bias: shuffle the option order and update gold letter accordingly.
        prompt = it["prompt"]
        gold = _gold(it)
        if gold not in ("A", "B", "C", "D"):
            return it

        letters, mapping = self._parse_mc_options(prompt)
        # Build list in current order, then shuffle
        opts = [(L, mapping[L]) for L in letters]
        rng.shuffle(opts)

        # Find which option text was originally correct.
        gold_text = mapping[gold]
        new_gold = next(L for (L, txt) in opts if txt == gold_text)

        it["prompt"] = self._rewrite_mc_line(prompt, opts)
        _set_gold(it, new_gold)
        it["meta"]["cf_edit"] = "shuffled_option_order_updated_gold"
        return it

    def _mc_permute_option_letters_only(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes label renaming: keep option order and texts, but rename the letters (A/B/C/D) consistently.
        prompt = it["prompt"]
        gold = _gold(it)
        if gold not in ("A", "B", "C", "D"):
            return it

        letters, mapping = self._parse_mc_options(prompt)
        perm = letters[:]
        rng.shuffle(perm)
        rename = {old: new for old, new in zip(letters, perm)}  # A->C etc.

        new_pairs = [(rename[L], mapping[L]) for L in letters]  # keep same order/text, only letter changes
        new_gold = rename[gold]

        it["prompt"] = self._rewrite_mc_line(prompt, new_pairs)
        _set_gold(it, new_gold)
        it["meta"]["cf_edit"] = "renamed_option_letters_updated_gold"
        return it

    # --- Phonology keyword trigger ---

    def _phon_rename_rhyme_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        it["prompt"] = it["prompt"].replace("RHYME=", "MATCH_SUFFIX=")
        it["meta"]["cf_edit"] = "renamed_RHYME_token"
        return it

    # --- Instruction comprehension ---

    def _instr_swap_yes_no(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Swap surface tokens YES/NO and flip gold.
        gold = _gold(it)
        if gold not in ("YES", "NO"):
            return it
        it["prompt"] = _swap_tokens(it["prompt"], "YES", "NO")
        _set_gold(it, "NO" if gold == "YES" else "YES")
        it["meta"]["cf_edit"] = "swapped_yes_no_tokens_and_flipped_gold"
        return it

    def _instr_rename_rule_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        it["prompt"] = it["prompt"].replace("RULE:", "INSTR:")
        it["meta"]["cf_edit"] = "renamed_RULE_marker"
        return it

    # --- Fine motor proxy ---

    def _motor_shuffle_moves_in_neutral_way(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes recency bias *without changing semantics* is hard here because order defines the endpoint.
        # So we do a safe counterfactual: insert explicit neutral no-op moves "N" that do nothing, near the end,
        # and keep gold the same by defining N as no-op in prompt (still same format, just adds a token).
        prompt = it["prompt"]
        gold = _gold(it)
        if not prompt.startswith("MOVE:"):
            return it

        # Add explicit no-op definition in the MOVE line (still "MOVE: ...")
        # e.g., "MOVE: U R N N" where N is no-op.
        m = re.search(r"^MOVE:\s*(.+)\nQ:\s*end=", prompt, flags=re.M)
        if not m:
            return it
        seq = m.group(1).split()
        seq2 = seq + ["N", "N"]

        it["prompt"] = re.sub(r"^MOVE:\s*.+$", f"MOVE: {' '.join(seq2)}  (N=no-op)", prompt, flags=re.M)
        _set_gold(it, gold)  # unchanged endpoint
        it["meta"]["cf_edit"] = "added_noop_moves_to_dispute_recency"
        return it

    def _motor_rename_move_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        it["prompt"] = it["prompt"].replace("MOVE:", "PATH:")
        it["meta"]["cf_edit"] = "renamed_MOVE_marker"
        return it

    # --- Social-emotional ---

    def _soc_rename_scale_labels(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes label attachment: rename scale labels consistently and update gold.
        # Expected labels: EMO/CON/HYP/PEER/PRO in prompt and completion.
        gold = _gold(it)
        labels = ["EMO", "CON", "HYP", "PEER", "PRO"]
        if gold not in labels:
            return it

        perm = labels[:]
        rng.shuffle(perm)
        ren = {a: b for a, b in zip(labels, perm)}

        new_prompt = it["prompt"]
        for a, b in ren.items():
            new_prompt = _swap_tokens(new_prompt, a, f"__{b}__")  # two-step to avoid collisions
        # unwrap placeholders
        new_prompt = re.sub(r"__([A-Z]+)__", r"\1", new_prompt)

        it["prompt"] = new_prompt
        _set_gold(it, ren[gold])
        it["meta"]["cf_edit"] = "renamed_scale_labels_and_updated_gold"
        it["meta"]["cf_label_map"] = ren
        return it

    def _soc_shuffle_scale_def_order(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes recency: shuffle the order of SCALES definitions; membership unchanged.
        prompt = it["prompt"]
        m = re.search(r"^SCALES:\s*(.+)\nITEM:", prompt, flags=re.M)
        if not m:
            return it
        defs = m.group(1)
        parts = [p.strip() for p in defs.split(";")]
        rng.shuffle(parts)
        new_defs = " ; ".join(parts)
        it["prompt"] = prompt.replace(defs, new_defs)
        it["meta"]["cf_edit"] = "shuffled_scales_definition_order"
        return it

    # --- Metacognition ---

    def _meta_rename_deviation_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes keyword trigger: rename 'deviation_d' to 'delta_d' in the query line.
        it["prompt"] = it["prompt"].replace("deviation_d", "delta_d")
        it["meta"]["cf_edit"] = "renamed_deviation_token"
        return it


if __name__ == "__main__":
    from evaluations.developmental_skills import BenchmarkBuilder, BenchmarkSpec
    rng = random.Random(0)

    # 1) Build a small dataset (one sample per skill) using your existing BenchmarkBuilder
    spec = BenchmarkSpec(
        seed=42,
        n_per_skill={
            "relational_reasoning": 1,
            "rule_induction": 1,
            "working_memory_maintenance": 1,
            "working_memory_manipulation": 1,
            "quantitative_reasoning": 1,
            "cognitive_control_inhibition": 1,
            "symbol_recognition": 1,
            "vocabulary": 1,
            "phonological_awareness": 1,
            "instruction_comprehension": 1,
            "fine_motor_proxy": 1,
            "social_emotional_awareness": 1,
            "metacognitive_self_estimation": 1,
        },
        shuffle=False,
    )

    builder = BenchmarkBuilder(spec)
    base_items = builder.generate()

    # 2) Create the counterfactual transformer
    tfm = CounterfactualTransformer()

    # 3) For each base sample, generate ONE counterfactual by applying a randomly chosen applicable shortcut
    counterfactual_items = []
    for item in base_items:
        skill = item["skill"]
        shortcuts = tfm.applicable_shortcuts(item)
        if not shortcuts:
            # No applicable shortcut for this skill (shouldn't happen with the provided map).
            continue

        chosen = rng.choice(shortcuts)
        cf_item = tfm.transform(item, shortcut=chosen, rng=rng)
        counterfactual_items.append(cf_item)

    # 4) Print base vs counterfactual side-by-side (first few)
    for base, cf in zip(base_items, counterfactual_items):
        print("\n==============================")
        print("SKILL:", base["skill"])
        print("--- BASE PROMPT ---")
        print(base["prompt"])
        print("BASE GOLD:", base["completion"])
        print("--- COUNTERFACTUAL PROMPT ---")
        print(cf["prompt"])
        print("CF GOLD:", cf["completion"])
        print("CF META:", {k: cf["meta"].get(k) for k in ["cf_shortcut", "cf_edit", "cf_from_id"]})
