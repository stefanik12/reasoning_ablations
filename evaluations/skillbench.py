"""
Unified text-only CLM benchmark generator, with *implementation logic* traced back (inline)
to what the original school readiness / entry-assessment resources describe.

All skills from the prior COMPLETENESS CHECK (except social-emotional awareness) are covered:

1. Relational reasoning
2. Rule induction
3. Working memory
4. Quantitative reasoning
5. Cognitive control / inhibition
6. Symbol recognition
7. Vocabulary
8. Phonological awareness
9. Instruction comprehension
10. Fine motor (proxy only, stated)
11. Metacognitive self-estimation

Note on social-emotional awareness:
- We choose to preclude this as for children aged 11 and under, the test is to be completed by a parent or teacher (not the child themselves).

Important note on “verification”:
- Some readiness tests are visual/motor or rely on oral administration (e.g., BPVS picture choice).
  Here, we generate **text-only proxies** that preserve the *cognitive demand* described in the
  resources while minimizing confounds (closed vocab, fixed templates, small answer spaces).
- Where a text-only proxy is not equivalent (motor; true phonology; visual symbol recognition),
  the code explicitly states that in comments.

Resource anchors used in comments (primary ones):
- BSRA/Bracken subtests (letters, numbers/counting, sizes/comparisons, shapes, colours):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6596936/  (Camacho et al., describes BSRA subtests)
  https://bristoluniversitypressdigital.com/view/journals/llcs/16/1/article-p45.xml (Fitzsimons 2025; item counts/subtests)
- NEPS procedural metacognition: “estimate how many tasks correct” + deviation score:
  https://www.neps-data.de/Portals/0/NEPS/Datenzentrum/Forschungsdaten/SC2/2-0-0/Proc_Meta_2.pdf
- NEPS SC2 (general): task inventories across domains (incl. kindergarten adaptations):
  https://www.neps-data.de/Portals/0/NEPS/Datenzentrum/Forschungsdaten/SC2/11-0-0/NEPS_SC2_DataManual_11-0-0_en.pdf
- NLSY79 Children: “Memory for Locations” (2–6 cups; 1–15s delay):
  https://www.nlsinfo.org/content/cohorts/nlsy79-children/topical-guide/assessments/memory-locations
- BiB “Starting School”: literacy/communication, fine motor (CKAT), social-emotional (SDQ), BPVS:
  https://wellcomeopenresearch.org/articles/5-47/v1/pdf
- SDQ structure (5 scales: emotional, conduct, hyperactivity/inattention, peer, prosocial):
  https://www.sdqinfo.org/a0.html
- TADI battery: direct tasks include “follow an instruction / execute a pre-set task”:
  https://tadi.cl/bateria-de-evaluacion-2/
  (Also describes cognition components in the slide deck):
  https://adipa.cl/content/uploads/2025/03/webinar-presentacion-tadi-2.pdf
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Callable, Optional
import random
import re
import hashlib
from typing import Dict, List, Any, Optional, Protocol, Tuple, Callable
from collections import defaultdict
from pathlib import Path
import json
from collections import Counter
from tqdm import tqdm

# -------------------------
# Common interface & helpers
# -------------------------

class Skill(Protocol):
    name: str
    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]: ...

def _choice(rng: random.Random, xs: List[Any]) -> Any:
    return xs[rng.randrange(len(xs))]

def _shuffle(rng: random.Random, xs: List[Any]) -> List[Any]:
    ys = list(xs)
    rng.shuffle(ys)
    return ys

def _stable_id(skill: str, prompt: str, completion: str) -> str:
    """Deterministic 32-hex id derived from content (stable across runs)."""
    h = hashlib.sha1((skill + "\n" + prompt + "\n" + completion).encode("utf-8")).hexdigest()
    return h[:32]

def _fmt_item(skill: str, prompt: str, completion: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    # Dataset is CLM/next-token style: prompt ends right before the answer.
    if not prompt.endswith("A:"):
        raise ValueError("Prompt must end with 'A:' (next-token completion format).")
    # Leading space tends to be friendlier for tokenizers/CLM continuations.
    if not completion.startswith(" "):
        completion = " " + completion
    return {"id": _stable_id(skill, prompt, completion), "skill": skill, "prompt": prompt, "completion": completion, "meta": meta}


# =========================
# 1) Relational reasoning
# =========================

class RelationalReasoningSkill:
    """
    Resource alignment:
    - BSRA/Bracken includes “Sizes/Comparisons” and “Shapes/Comparisons” among basic concept subtests
      (Camacho et al. describe BSRA as having subtests for ... sizes, comparisons, shapes, etc.):
      https://pmc.ncbi.nlm.nih.gov/articles/PMC6596936/
    - These subtests require applying comparison relations (bigger/smaller; same/different; matching).

    Implementation alignment:
    - We model “comparison concepts” with purely symbolic relational facts (A > B, B > C) and ask for min/max.
    - This preserves the *relational comparison* demand while minimizing language/visual confounds.

    Refinement (vs earlier versions):
    - We keep the chain short to reduce working-memory confounds and focus on relational inference.
    - We ask for comparison of specific relation with YES/NO output rather than just min/max.
    """

    name = "relational_reasoning"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        relation: str = "random",    # ">", "<", or "random"
        query: str = "random",       # "min", "max", "compare", or "random"
        chain_len: str = "random",   # 2, 3, or "random"
    ):
        self.symbols = symbols or ["A", "B", "C", "D", "E"]  # 5 symbols
        self.relation = relation
        self.query = query
        self.chain_len = chain_len

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            # Choose a total order so that "comparison" is well-defined and unambiguous.
            order = _shuffle(rng, self.symbols)
            
            # Randomize chain_len if set to "random"
            chain_len = rng.choice([2, 3]) if self.chain_len == "random" else self.chain_len
            k = min(chain_len, len(order) - 1)

            # Randomize relation if set to "random"
            relation = rng.choice([">", "<"]) if self.relation == "random" else self.relation

            # Facts mirror "comparison relationships" (e.g., bigger-than) in a stripped symbolic form.
            involved = order[: k + 1]
            facts = [f"{involved[i]} {relation} {involved[i+1]}" for i in range(k)]

            # Determine ordering semantics based on relation
            # For ">": involved[0] is largest, involved[-1] is smallest
            # For "<": involved[0] is smallest, involved[-1] is largest
            relation_means_greater = (relation == ">")

            # Randomize query if set to "random", otherwise use fixed query
            query = rng.choice(["min", "max", "compare"]) if self.query == "random" else self.query

            # Query asks to apply the inferred ordering (min/max), or transitive comparison
            if query == "min":
                # min = smallest element in the ordering
                gold = involved[-1] if relation_means_greater else involved[0]
                q = f"Q: min({','.join(involved)})="
            elif query == "max":
                # max = largest element in the ordering
                gold = involved[0] if relation_means_greater else involved[-1]
                q = f"Q: max({','.join(involved)})="
            elif query == "compare":
                # Transitive comparison: ask if X rel Y for non-adjacent elements
                if len(involved) >= 3:
                    # Pick first and last (guaranteed to require transitive inference)
                    first, last = involved[0], involved[-1]
                else:
                    # With only 2 elements, just use them (direct from facts)
                    first, last = involved[0], involved[1]
                
                # Randomly decide whether to ask true or false question
                # By construction: first rel first+1 rel ... rel last
                # So "first rel last" is TRUE (transitive closure)
                # And "last rel first" is FALSE
                if rng.random() < 0.5:
                    # True case: first rel last (follows from chain by construction)
                    q = f"Q: {first}{relation}{last}=YES/NO?"
                    gold = "YES"
                else:
                    # False case: last rel first (inverse of the chain)
                    q = f"Q: {last}{relation}{first}=YES/NO?"
                    gold = "NO"
            else:
                raise ValueError("query must be 'min', 'max', 'compare', or 'random'")

            prompt = "\n".join(facts + [q, "A:"])
            meta = {"order": order, "facts": facts, "involved": involved, "query": query}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# =========================
# 2) Rule induction
# =========================

class RuleInductionSkill:
    """
    Resource alignment:
    - NEPS includes non-verbal/problem-solving style components (e.g., matrices/pattern reasoning) across cohorts;
      SC2 manual provides the broader testing context:
      https://www.neps-data.de/Portals/0/NEPS/Datenzentrum/Forschungsdaten/SC2/11-0-0/NEPS_SC2_DataManual_11-0-0_en.pdf
    - TADI cognition explicitly includes “razonamiento lógico-matemático” and “resolución de problemas”
      (slide deck lists these cognitive processes):
      https://adipa.cl/content/uploads/2025/03/webinar-presentacion-tadi-2.pdf

    Implementation alignment:
    - Present multiple example input-output pairs for a hidden operator and ask the model to infer/apply it.
    - This matches the *induction from examples* / *pattern rule inference* aspect of those tasks.

    Refinement:
    - Keep symbols closed; avoid any semantic “math meaning” of '+'—it’s explicitly an arbitrary operator.
    """

    name = "rule_induction"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        op: str = "+",
        n_examples: int = 3,
    ):
        self.symbols = symbols or ["X", "Y", "Z"]
        self.op = op
        self.n_examples = n_examples

    def _make_hidden_op(self, rng: random.Random):
        # Hidden operator defined as a cyclic table under a random permutation:
        # This ensures learnable regularity (like a “pattern”) without importing arithmetic semantics.
        syms = self.symbols
        perm = _shuffle(rng, syms)
        idx = {s: i for i, s in enumerate(perm)}
        k = len(syms)

        def hidden(a: str, b: str) -> str:
            return perm[(idx[a] + idx[b]) % k]

        return perm, hidden

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        syms = self.symbols
        for _ in range(n):
            perm, hidden = self._make_hidden_op(rng)

            # Examples correspond to “items ordered by difficulty / induction from small evidence” (TADI),
            # and to “infer rule from structure” (NEPS-like reasoning).
            pairs: List[Tuple[str, str]] = []
            while len(pairs) < self.n_examples:
                a, b = _choice(rng, syms), _choice(rng, syms)
                if (a, b) not in pairs:
                    pairs.append((a, b))
            ex_lines = [f"{a}{self.op}{b}={hidden(a,b)}" for a, b in pairs]

            # Query is a held-out pair, enforcing generalization beyond rote copying.
            while True:
                qa, qb = _choice(rng, syms), _choice(rng, syms)
                if (qa, qb) not in pairs:
                    break
            gold = hidden(qa, qb)

            prompt = "EX:\n" + "\n".join(ex_lines) + f"\nQ: {qa}{self.op}{qb}=\nA:"
            meta = {"perm": perm, "examples": ex_lines, "query": f"{qa}{self.op}{qb}", "gold": gold}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# ===========================================
# 3) Working memory (maintenance) — explicit
# ===========================================

class WorkingMemoryMaintenanceSkill:
    """
    References:
    - NLSY79 Children “Memory for Locations” describes a short-term memory task:
      child watches a figure hidden under 2–6 cups; cups are screened 1–15 seconds; child finds location.
      https://www.nlsinfo.org/content/cohorts/nlsy79-children/topical-guide/assessments/memory-locations

    Mapping to implementation:
    - Maintenance here = keep a sequence "in mind" and retrieve a specific element.
    - This captures the *storage* component of short-term memory (without spatial/visual modality).
    """

    name = "working_memory_maintenance"

    def __init__(self, alphabet: Optional[List[str]] = None, seq_len: int = 5, ask_index: int = 3):
        self.alphabet = alphabet or list("0123456789")
        self.seq_len = seq_len
        self.ask_index = ask_index

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            seq = [_choice(rng, self.alphabet) for _ in range(self.seq_len)]

            # Retrieve kth element: a direct analog of “remember which cup/position” but abstracted.
            k = self.ask_index
            if not (1 <= k <= len(seq)):
                raise ValueError("ask_index out of range")
            gold = seq[k - 1]

            prompt = f"SEQ: {' '.join(seq)}\nQ: {k}th=\nA:"
            meta = {"seq": seq, "k": k, "note": "maintenance-only WM proxy"}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# ===========================================
# 4) Working memory (manipulation) — explicit
# ===========================================

class WorkingMemoryManipulationSkill:
    """
    References:
    - NEPS early cohorts include working-memory style tasks; backward-span is a standard manipulation form.
      (SC2 manual provides context for domain-general cognition tasks and competence measures)
      https://www.neps-data.de/Portals/0/NEPS/Datenzentrum/Forschungsdaten/SC2/11-0-0/NEPS_SC2_DataManual_11-0-0_en.pdf

    Mapping to implementation:
    - Manipulation here = transform stored sequence by reversing it (REV), akin to backward span.
    - This isolates the *manipulation* component from mere storage.
    """

    name = "working_memory_manipulation"

    def __init__(self, alphabet: Optional[List[str]] = None, seq_len: int = 4):
        self.alphabet = alphabet or list("0123456789")
        self.seq_len = seq_len

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            seq = [_choice(rng, self.alphabet) for _ in range(self.seq_len)]

            # Reverse requires holding the whole sequence and producing it in transformed order.
            gold = " ".join(reversed(seq))
            prompt = f"SEQ: {' '.join(seq)}\nQ: REV=\nA:"
            meta = {"seq": seq, "note": "manipulation-only WM proxy (backward-span-like)"}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# =========================
# 5) Quantitative reasoning
# =========================

class QuantitativeReasoningSkill:
    """
    Resource alignment:
    - BSRA includes “Numbers/Counting” as a readiness subtest (Camacho et al. list it explicitly):
      https://pmc.ncbi.nlm.nih.gov/articles/PMC6596936/
    - Bracken/BSRA is about “knowledge and understanding of basic concepts” for readiness;
      number/counting is a canonical such concept.

    Implementation alignment:
    - Compare two small sets represented as repeated neutral tokens (*** vs **).
    - This keeps the cognitive demand “which set is larger?” aligned with basic numeracy concepts,
      without adding language or world knowledge.

    Refinement:
    - Keep numbers tiny to reduce arithmetic-algorithm confounds and focus on cardinality comparison.
    """

    name = "quantitative_reasoning"

    def __init__(
        self,
        dot: str = "*",
        min_n: int = 1,
        max_n: int = 5,
        label_pool: Optional[List[str]] = None,
        allow_equal: bool = False,
    ):
        self.dot = dot
        self.min_n = min_n
        self.max_n = max_n
        self.label_pool = label_pool or ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.allow_equal = allow_equal

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            # Randomly select two distinct labels
            labels = _shuffle(rng, self.label_pool)[:2]
            X, Y = labels[0], labels[1]
            
            a = rng.randint(self.min_n, self.max_n)
            b = rng.randint(self.min_n, self.max_n)
            if not self.allow_equal:
                while b == a:
                    b = rng.randint(self.min_n, self.max_n)

            gold = X if a > b else (Y if b > a else "EQ")
            # The prompt is essentially a “count/compare quantities” micro-item.
            prompt = f"{X}: {self.dot * a}\n{Y}: {self.dot * b}\nQ: more({X},{Y})=\nA:"
            meta = {"counts": {X: a, Y: b}, "allow_equal": self.allow_equal}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# ==================================
# 6) Cognitive control / inhibition
# ==================================

class CognitiveControlInhibitionSkill:
    """
    Resource alignment:
    - TADI cognition includes attention/self-regulation related processes (attention explicitly listed)
      and is applied via direct tasks (follow instruction / execute task):
      https://adipa.cl/content/uploads/2025/03/webinar-presentacion-tadi-2.pdf
      https://tadi.cl/bateria-de-evaluacion-2/
    - NEPS includes metacompetencies and self-regulation adjacent constructs; while not all are
      “inhibition tasks”, cognitive control is commonly measured via tasks requiring suppression of a
      prepotent response (SC2 manual provides context of domains/measures):
      https://www.neps-data.de/Portals/0/NEPS/Datenzentrum/Forschungsdaten/SC2/11-0-0/NEPS_SC2_DataManual_11-0-0_en.pdf

    Implementation alignment:
    - We create an explicit mapping “MAP: A->B ...” then query with “Q: A”.
    - The *prepotent* (easy) response is to copy A; the correct response is to output mapped token.
    - This is a minimal “override default response” proxy (inhibition/control) in text-only form.

    Refinement:
    - Enforce derangement (no fixed points) so copying is *always* wrong.
    """

    name = "cognitive_control_inhibition"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        force_derangement = True,  # True, False, or "random"
    ):
        self.symbols = symbols or ["A", "B", "C", "D", "E", "F"]
        self.force_derangement = force_derangement

    def _mapping(self, rng: random.Random, force_derangement: bool) -> Dict[str, str]:
        syms = self.symbols
        perm = _shuffle(rng, syms)
        if force_derangement and len(syms) > 1:
            # Ensure “copy input” never matches the correct output.
            for _ in range(50):
                if any(a == b for a, b in zip(syms, perm)):
                    perm = _shuffle(rng, syms)
                else:
                    break
        return {a: b for a, b in zip(syms, perm)}

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            if isinstance(self.force_derangement, str):
                if self.force_derangement == "random":
                    force_derangement = rng.choice([True, False])
                else:
                    raise ValueError("force_derangement must be True, False, or 'random'")
            elif isinstance(self.force_derangement, bool):
                force_derangement = self.force_derangement
            else:
                raise ValueError("force_derangement must be True, False, or 'random'")
            
            m = self._mapping(rng, force_derangement)
            x = _choice(rng, self.symbols)
            gold = m[x]
            # “Given instruction/rule → execute task” matches TADI’s “tarea directa” framing.
            map_str = ", ".join([f"{k}->{v}" for k, v in m.items()])
            prompt = f"MAP: {map_str}\nQ: {x}\nA:"
            meta = {"mapping": m, "x": x, "note": "inhibition proxy: copying x is wrong"}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# =========================
# 7) Symbol recognition
# =========================

class SymbolRecognitionSkill:
    """
    Resource alignment:
    - BSRA includes “Letters” as a basic concept subtest (Camacho et al. list letters among BSRA subtests):
      https://pmc.ncbi.nlm.nih.gov/articles/PMC6596936/
    - BiB Starting School uses early literacy measures (incl. letter identification tests):
      https://wellcomeopenresearch.org/articles/5-47/v1/pdf

    IMPORTANT LIMITATION:
    - These assessments are typically visual (recognize printed letters).
    - Text-only cannot evaluate visual discrimination. So we implement a *formal symbol category*
      membership proxy, which still tests “recognize/identify membership”, but not vision.

    Implementation alignment:
    - Define a set ALPH and ask whether a queried symbol is a member.
    - This keeps the decision structure “is this one of the target symbols?”.
    """

    name = "symbol_recognition"

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        alph_size: int = 3,
        query_pool_size: int = 6,
    ):
        self.universe = universe or list("ABCDEFGHJKLMNPQRSTUVWXYZ")
        self.alph_size = alph_size
        self.query_pool_size = query_pool_size

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            pool = _shuffle(rng, self.universe)[: self.query_pool_size]
            alph = pool[: self.alph_size]
            q = _choice(rng, pool)
            # “recognize” = decide membership; this is the formal proxy for letter-ID.
            gold = "YES" if q in alph else "NO"
            prompt = f"ALPH: {{{','.join(alph)}}}\nQ: member({q})=YES/NO?\nA:"
            meta = {"alph": alph, "q": q, "note": "proxy for visual letter recognition"}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# =========================
# 8) Vocabulary (controlled)
# =========================

class VocabularySkill:
    """
    Resource alignment:
    - BiB Starting School includes BPVS (British Picture Vocabulary Scale) as a literacy/communication measure:
      https://wellcomeopenresearch.org/articles/5-47/v1/pdf
      (BPVS is typically: spoken word → choose matching picture among options.)

    IMPORTANT LIMITATION:
    - BPVS is orally administered with picture choices. Text-only cannot reproduce the perceptual component.
    - But the *decision structure* can be mimicked: “target token” → choose correct option among distractors.

    Implementation alignment:
    - Provide an explicit artificial lexicon DICT (removes world knowledge).
    - Ask a BPVS-like 4-option multiple-choice: choose which option is the correct “meaning(word)”.
    """

    name = "vocabulary"

    def __init__(
        self,
        words: Optional[List[str]] = None,
        meanings: Optional[List[str]] = None,
        n_options: int = 4,
    ):
        self.words = words or ["dax", "wug", "blicket", "toma"]
        self.meanings = meanings or ["RED", "BLUE", "GREEN", "YELLOW"]
        self.n_options = n_options

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        if len(self.words) != len(self.meanings):
            raise ValueError("words and meanings must have same length")
        if self.n_options < 2:
            raise ValueError("n_options must be >= 2")

        out: List[Dict[str, Any]] = []
        pairs = list(zip(self.words, self.meanings))
        for _ in range(n):
            # Show full dictionary so the task is “receptive mapping” (not induction).
            dict_pairs = _shuffle(rng, pairs)
            dict_str = ", ".join([f"{w}={m}" for w, m in dict_pairs])
            q_word, gold_meaning = _choice(rng, dict_pairs)

            # BPVS-like forced choice among distractors (still controlled, not world knowledge).
            distractors = [m for (_, m) in dict_pairs if m != gold_meaning]
            opts = [gold_meaning] + [_choice(rng, distractors) for _ in range(self.n_options - 1)]
            opts = _shuffle(rng, opts)
            letters = ["A", "B", "C", "D"][: len(opts)]
            gold_letter = letters[opts.index(gold_meaning)]

            opt_str = " ".join([f"{L}:{o}" for L, o in zip(letters, opts)])
            prompt = f"DICT: {dict_str}\nWORD: {q_word}\n{opt_str}\nQ: meaning(WORD)=\nA:"
            meta = {"dict": dict(dict_pairs), "q_word": q_word, "options": dict(zip(letters, opts))}
            out.append(_fmt_item(self.name, prompt, gold_letter, meta))
        return out


# ==============================
# 9) Phonological awareness proxy
# ==============================

class PhonologicalAwarenessSkill:
    """
    Resource alignment:
    - NEPS includes listening/language-related competence measures and kindergarten adaptations; phonological
      awareness tasks (e.g., rhyme) are typical precursors in early literacy batteries (SC2 manual context):
      https://www.neps-data.de/Portals/0/NEPS/Datenzentrum/Forschungsdaten/SC2/11-0-0/NEPS_SC2_DataManual_11-0-0_en.pdf

    IMPORTANT LIMITATION:
    - True phonological awareness is auditory. Text-only must implement an explicit proxy.

    Implementation alignment:
    - Define “RHYME = same last k letters” on nonsense syllables.
    - Ask for which option “rhymes” with the target under that explicit definition.
    """

    name = "phonological_awareness"

    def __init__(
        self,
        k_suffix: int = 2,
        syllables: Optional[List[str]] = None,
        n_options: int = 3,
    ):
        self.k_suffix = k_suffix
        self.syllables = syllables or ["mep", "lep", "dap", "mog", "teg", "pag", "nup", "siv", "rav"]
        self.n_options = n_options

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            target = _choice(rng, self.syllables)
            suffix = target[-self.k_suffix:]

            rhymers = [s for s in self.syllables if s != target and s.endswith(suffix)]
            nonrhymers = [s for s in self.syllables if not s.endswith(suffix)]

            # If the list is too small for k_suffix, degrade gracefully (still explicit rule).
            if not rhymers:
                suffix = target[-1:]
                rhymers = [s for s in self.syllables if s != target and s.endswith(suffix)]
                nonrhymers = [s for s in self.syllables if not s.endswith(suffix)]
                rule = "same last 1 letter"
            else:
                rule = f"same last {self.k_suffix} letters"

            right = _choice(rng, rhymers) if rhymers else _choice(rng, self.syllables)
            opts = [right] + [_choice(rng, nonrhymers) for _ in range(self.n_options - 1)]
            opts = _shuffle(rng, opts)
            letters = ["A", "B", "C", "D"][: len(opts)]
            gold = letters[opts.index(right)]

            opt_str = " ".join([f"{L}:{w}" for L, w in zip(letters, opts)])
            prompt = f"RHYME={rule}\nT:{target}\n{opt_str}\nQ: rhyme(T)=\nA:"
            meta = {"target": target, "rule": rule, "options": dict(zip(letters, opts))}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# =========================
# 10) Instruction comprehension
# =========================

class InstructionComprehensionSkill:
    """
    Resource alignment:
    - TADI explicitly distinguishes item types and includes direct tasks:
      “Tarea directa: evalúa si el niño logra seguir una instrucción y/o ejecutar una tarea preestablecida.”
      https://tadi.cl/bateria-de-evaluacion-2/

    Implementation alignment:
    - Provide a simple fixed RULE and ask the model to apply it (YES/NO).
    - We keep instruction wording constant across items to avoid “instruction novelty” confounds.
    """

    name = "instruction_comprehension"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        set_size: int = 5,
    ):
        self.symbols = symbols or ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.set_size = set_size

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            # Randomly select which symbols form set1
            shuffled = _shuffle(rng, self.symbols)
            set1 = sorted(shuffled[:self.set_size])
            set2 = sorted(shuffled[self.set_size:])
            
            x = _choice(rng, self.symbols)
            # "Follow instruction → execute" matches TADI direct-task framing, but simplified.
            gold = "YES" if x in set1 else "NO"
            prompt = f"RULE: YES if x in {{{','.join(set1)}}}, else NO.\nQ: x={x}\nA:"
            meta = {"set1": set1, "set2": set2, "x": x}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# =========================
# 11) Fine motor proxy
# =========================

class FineMotorProxySkill:
    """
    Resource alignment:
    - BiB Starting School assesses fine motor skills using CKAT-style tasks (tracking/aiming/tracing),
      i.e., motor execution + visuomotor integration:
      https://wellcomeopenresearch.org/articles/5-47/v1/pdf

    IMPORTANT LIMITATION:
    - Text-only cannot test motor execution/precision.

    Implementation alignment (proxy only):
    - “MOVE: U R ...” and ask for end coordinate is a proxy for tracking a path (sequencing + spatial integration).
    - This reflects *the cognitive part* of path tracking but not motor control.
    """

    name = "fine_motor_proxy"

    def __init__(
        self,
        steps: int = 6,
        moves: Optional[List[str]] = None,
    ):
        self.steps = steps
        self.moves = moves or ["U", "D", "L", "R"]

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            seq = [_choice(rng, self.moves) for _ in range(self.steps)]

            # Proxy for “tracking”: integrate step-by-step movement.
            x = y = 0
            for m in seq:
                if m == "U": y += 1
                elif m == "D": y -= 1
                elif m == "R": x += 1
                elif m == "L": x -= 1

            gold = f"({x},{y})"
            prompt = f"MOVE: {' '.join(seq)}\nQ: end=(x,y)=\nA:"
            meta = {"moves": seq, "end": (x, y), "note": "proxy only (not motor precision)"}
            out.append(_fmt_item(self.name, prompt, gold, meta))
        return out


# =========================
# 12) Metacognitive self-estimation
# =========================

class MetacognitiveSelfEstimationSkill:
    """
    Resource alignment (NEPS procedural metacognition doc):
    - NEPS does NOT directly measure planning/monitoring; instead it uses “metacognitive judgments of performance”.
    - After completing items in a domain, participants estimate how many they got correct.
    - Reported scores include:
        (a) estimated proportion correct  = N_estimated / N_items
        (b) deviation score d = (N_estimated/N_items) - (N_correct/N_items)
      with interpretation: d>0 overestimation, d<0 underestimation, d=0 perfect estimation.
      Source:
      https://www.neps-data.de/Portals/0/NEPS/Datenzentrum/Forschungsdaten/SC2/2-0-0/Proc_Meta_2.pdf

    Implementation alignment (refined to *match the doc exactly*):
    - We generate (N_items, N_correct, N_estimated) and ask for deviation score d.
    - Output is a fixed-format decimal rounded to 2 places to keep answers short and comparable.
    """

    name = "metacognitive_self_estimation"

    def __init__(
        self,
        list_of_skills: list = ["relational_reasoning",
                        "rule_induction",
                        "working_memory_maintenance",
                        "working_memory_manipulation",
                        "quantitative_reasoning",
                        "cognitive_control_inhibition",
                        "symbol_recognition",
                        "vocabulary",
                        "phonological_awareness",
                        "instruction_comprehension",
                        "fine_motor_proxy",
                        # "social_emotional_awareness": 5,
                        "metacognitive_self_estimation"]
    ):
        self.totals = list_of_skills

    def sample_one_per_skill_and_concat(
        self,
        items: List[Dict[str, Any]],
        rng: random.Random,
    ) -> Tuple[str, int, int, List[Dict[str, Any]]]:
        # 1) Group by skill
        by_skill: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for it in items:
            by_skill[it["skill"]].append(it)

        # 2) Pick one per skill (sorted for deterministic skill order in concatenation)
        selected: List[Dict[str, Any]] = []
        for skill in sorted(by_skill.keys()):
            selected.append(_choice(rng, by_skill[skill]))  # or rng.choice(...)

        # 3) Concatenate prompts
        # Add a separator so boundaries are unambiguous
        concat_prompt = "\n\n---\n\n".join(it["prompt"] for it in selected)

        # 4) Count correct
        n_correct = sum(1 for it in selected if it.get("meta", {}).get("correct") is True)
        n_total = len(selected)

        return concat_prompt, n_correct, n_total
    
    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        spec = BenchmarkSpec(
        seed=rng.random(),
        n_per_skill={
            "relational_reasoning": n,
            "rule_induction": n,
            "working_memory_maintenance": n,
            "working_memory_manipulation": n,
            "quantitative_reasoning": n,
            "cognitive_control_inhibition": n,
            "symbol_recognition": n,
            "vocabulary": n,
            "phonological_awareness": n,
            "instruction_comprehension": n,
            "fine_motor_proxy": n,
            # "social_emotional_awareness": n,
        },
        shuffle=False,
        )

        builder = BenchmarkBuilder(spec)
        items = builder.generate()

        wrong_choices_letters = [" A", " B", " C", " D", " E", " F", " G", " H", " I", " J"]
        wrong_choices_xyz = [" X", " Y", " Z"]
        wrong_choices_seq = [" 1 2 3 4", " 4 5 6 7", "8 9 1 2"]
        wrong_choices_coord = [" (0,0)", " (1,1)", "(0, 1)", " (1,0)"]
        wrong_choices_yesno = [" YES", " NO"]
        wrong_choices_int = [" 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]

        # Map skill -> wrong-choice list
        wrong_pool_by_skill = {
            "fine_motor_proxy": wrong_choices_coord,                 # (x,y)
            "instruction_comprehension": wrong_choices_yesno,        # YES/NO
            "phonological_awareness": wrong_choices_letters,         # Letter
            "vocabulary": wrong_choices_letters,                     # Letter
            "symbol_recognition": wrong_choices_yesno,               # YES/NO
            "cognitive_control_inhibition": wrong_choices_letters,   # Letter
            "quantitative_reasoning": wrong_choices_letters,         # Letter
            "working_memory_manipulation": wrong_choices_seq,        # number seq with spaces
            "working_memory_maintenance": wrong_choices_int,         # int
            "rule_induction": wrong_choices_xyz,                     # X/Y/Z
            "relational_reasoning": wrong_choices_letters,           # Letter
        }

        # Build prompts with (sometimes wrong) answers appended
        for item in items:
            skill = item.get("skill")
            gold = item["completion"]

            # Default: keep correct unless we can confidently sample a wrong answer
            completion = gold

            # Decide whether to flip to a wrong answer (50/50)
            if rng.choice([True, False]):
                pool = wrong_pool_by_skill.get(skill)

                if pool is not None:
                    # Ensure the sampled wrong answer isn't accidentally the gold
                    candidates = [c for c in pool if c != gold]
                    if candidates:
                        completion = rng.choice(candidates)
                        item["meta"]["correct"] = False
                    else:
                        # Pool had only the gold; fall back to correct
                        item["meta"]["correct"] = True
            else:
                item["meta"]["correct"] = True

            # Concatenate prompt + (possibly modified) completion
            item["prompt"] = item["prompt"] + completion


        for _ in range(n):
            concat_prompt, n_correct, n_total = self.sample_one_per_skill_and_concat(items, rng)
            
            prompt = concat_prompt + "\n\n---\n\n\nQ: How many were correct?\nA:"
            gold = f" {n_correct}"
            meta = {"n_total": n_total}

            out.append(_fmt_item(self.name, prompt, gold, meta))

        return out



# -------------------------
# Benchmark wrapper
# -------------------------

@dataclass
class BenchmarkSpec:
    seed: int = 0
    n_per_skill: Optional[Dict[str, int]] = None
    shuffle: bool = True
    overrides: Optional[Dict[str, Dict[str, Any]]] = None  # kwargs per skill constructor


class BenchmarkBuilder:
    """
    Builds the benchmark under a single interface.
    All skills are separate classes.
    """

    def __init__(self, spec: BenchmarkSpec):
        self.spec = spec
        self.rng = random.Random(spec.seed)
        self.overrides = spec.overrides or {}

        self.skills: List[Skill] = [
            self._make(RelationalReasoningSkill, "relational_reasoning"),
            self._make(RuleInductionSkill, "rule_induction"),
            self._make(WorkingMemoryMaintenanceSkill, "working_memory_maintenance"),
            self._make(WorkingMemoryManipulationSkill, "working_memory_manipulation"),
            self._make(QuantitativeReasoningSkill, "quantitative_reasoning"),
            self._make(CognitiveControlInhibitionSkill, "cognitive_control_inhibition"),
            self._make(SymbolRecognitionSkill, "symbol_recognition"),
            self._make(VocabularySkill, "vocabulary"),
            self._make(PhonologicalAwarenessSkill, "phonological_awareness"),
            self._make(InstructionComprehensionSkill, "instruction_comprehension"),
            self._make(FineMotorProxySkill, "fine_motor_proxy"),
            self._make(MetacognitiveSelfEstimationSkill, "metacognitive_self_estimation"),
        ]

        default_counts = {s.name: 100 for s in self.skills}
        self.n_per_skill = spec.n_per_skill or default_counts

    def _make(self, cls, key: str):
        kwargs = self.overrides.get(key, {})
        return cls(**kwargs)

    def generate(self, progress: bool = False) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for sk in tqdm(self.skills, desc="Generating base prompts", disable=not progress):
            n = self.n_per_skill.get(sk.name, 0)
            if n <= 0:
                continue
            items.extend(sk.generate(n=n, rng=self.rng))
        if self.spec.shuffle:
            self.rng.shuffle(items)
        return items


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
    COPY_BIAS = "copy_bias"
    TRAILING_SEQ_DISTRACTOR = "trailing_seq_distractor"


SKILL_TO_SHORTCUTS: Dict[str, List[str]] = {
    "relational_reasoning": [
        Shortcut.LAST_TOKEN_HEURISTIC, Shortcut.FORMAT_KEYWORD_TRIGGER 
    ],
    "rule_induction": [
        Shortcut.LAST_TOKEN_HEURISTIC, Shortcut.FORMAT_KEYWORD_TRIGGER 
    ],
    "working_memory_maintenance": [
        Shortcut.FORMAT_KEYWORD_TRIGGER, Shortcut.COPY_BIAS, Shortcut.TRAILING_SEQ_DISTRACTOR 
    ],
    "working_memory_manipulation": [
        Shortcut.RECENCY_BIAS, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "quantitative_reasoning": [
        Shortcut.LENGTH_HEURISTIC
    ],
    "cognitive_control_inhibition": [
        Shortcut.LAST_TOKEN_HEURISTIC, Shortcut.FORMAT_KEYWORD_TRIGGER, Shortcut.COPY_BIAS
    ],
    "symbol_recognition": [
        Shortcut.FORMAT_KEYWORD_TRIGGER 
    ],
    "vocabulary": [
        Shortcut.COPY_BIAS, Shortcut.LENGTH_HEURISTIC 
    ],
    "phonological_awareness": [
        Shortcut.FORMAT_KEYWORD_TRIGGER 
    ],
    "instruction_comprehension": [
        Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "fine_motor_proxy": [
        Shortcut.RECENCY_BIAS, Shortcut.FORMAT_KEYWORD_TRIGGER
    ],
    "metacognitive_self_estimation": [
        Shortcut.COPY_BIAS 
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
            ("relational_reasoning", Shortcut.LAST_TOKEN_HEURISTIC): self._rr_add_trailing_distractor_symbol,
            ("relational_reasoning", Shortcut.FORMAT_KEYWORD_TRIGGER): self._rr_rename_query_token,

            # rule_induction
            ("rule_induction", Shortcut.LAST_TOKEN_HEURISTIC): self._ri_add_trailing_equation_noise,
            ("rule_induction", Shortcut.FORMAT_KEYWORD_TRIGGER): self._ri_rename_ex_q_tokens,

            # WM maintenance
            ("working_memory_maintenance", Shortcut.TRAILING_SEQ_DISTRACTOR): self._wm_maint_add_trailing_noise,
            ("working_memory_maintenance", Shortcut.FORMAT_KEYWORD_TRIGGER): self._wm_maint_rename_markers,
            ("working_memory_maintenance", Shortcut.COPY_BIAS): self._wm_maint_add_conflicting_hint,

            # WM manipulation
            ("working_memory_manipulation", Shortcut.RECENCY_BIAS): self._wm_manip_insert_irrelevant_prefix,
            ("working_memory_manipulation", Shortcut.FORMAT_KEYWORD_TRIGGER): self._wm_manip_rename_rev_token,

            # quantitative
            ("quantitative_reasoning", Shortcut.LENGTH_HEURISTIC): self._quant_break_length_proxy,

            # inhibition
            ("cognitive_control_inhibition", Shortcut.LAST_TOKEN_HEURISTIC): self._inh_append_query_distractor,
            ("cognitive_control_inhibition", Shortcut.FORMAT_KEYWORD_TRIGGER): self._inh_rename_map_token,
            ("cognitive_control_inhibition", Shortcut.COPY_BIAS): self._inh_add_conflicting_hint,

            # symbol recognition
            ("symbol_recognition", Shortcut.FORMAT_KEYWORD_TRIGGER): self._symrec_rename_member_token,

            # vocabulary (MC)
            ("vocabulary", Shortcut.COPY_BIAS): self._mc_add_conflicting_hint,
            ("vocabulary", Shortcut.LENGTH_HEURISTIC): self._mc_add_irrelevant_dict_entry,   

            # phonological (MC)
            ("phonological_awareness", Shortcut.FORMAT_KEYWORD_TRIGGER): self._phon_rename_rhyme_token,

            # instruction comprehension
            ("instruction_comprehension", Shortcut.FORMAT_KEYWORD_TRIGGER): self._instr_rename_rule_token,

            # fine motor proxy
            ("fine_motor_proxy", Shortcut.RECENCY_BIAS): self._motor_shuffle_moves_in_neutral_way,
            ("fine_motor_proxy", Shortcut.FORMAT_KEYWORD_TRIGGER): self._motor_rename_move_token,

            # metacognition
            ("metacognitive_self_estimation", Shortcut.COPY_BIAS): self._meta_add_noisy_illustration,
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

    def _rr_rename_query_token(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Disputes keyword-trigger reliance: rename 'min'/'max' token while preserving meaning.
        # We keep same structure but change the keyword.
        # For compare queries, rename YES/NO to TRUE/FALSE.
        prompt = it["prompt"]
        if "min(" in prompt:
            it["prompt"] = prompt.replace("min(", "smallest(")
            it["meta"]["cf_edit"] = "renamed_min_to_smallest"
        elif "max(" in prompt:
            it["prompt"] = prompt.replace("max(", "largest(")
            it["meta"]["cf_edit"] = "renamed_max_to_largest"
        elif "=YES/NO?" in prompt:
            # Compare query: rename YES/NO tokens to TRUE/FALSE
            it["prompt"] = prompt.replace("=YES/NO?", "=TRUE/FALSE?")
            gold = _gold(it)
            if gold == "YES":
                _set_gold(it, "TRUE")
            elif gold == "NO":
                _set_gold(it, "FALSE")
            it["meta"]["cf_edit"] = "renamed_yesno_to_truefalse"
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

    def _wm_maint_rename_markers(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        """
        Disputes FORMAT_KEYWORD_TRIGGER: rename SEQ: -> LIST: and th= -> pos=
        to test if model relies on specific keywords rather than task understanding.
        """
        prompt = it["prompt"]
        prompt = prompt.replace("SEQ:", "LIST:")
        prompt = prompt.replace("th=", "pos=")
        it["prompt"] = prompt
        it["meta"]["cf_edit"] = "renamed_seq_and_th_markers"
        return it

    def _wm_maint_add_conflicting_hint(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        """
        Disputes COPY_BIAS: add HINT: <wrong_element> before the Q: line.
        The model must ignore this salient distractor and retrieve from memory.
        """
        prompt = it["prompt"]
        gold = _gold(it)
        
        # Parse the sequence to find wrong elements
        m = re.search(r"^SEQ:\s*(.+)$", prompt, flags=re.M)
        if not m:
            return it
        
        seq = m.group(1).split()
        # Pick a wrong element (different from gold)
        wrong_elements = [e for e in seq if e != gold]
        if not wrong_elements:
            return it
        
        wrong_hint = rng.choice(wrong_elements)
        
        # Insert HINT: <wrong_element> before the Q: line
        it["prompt"] = re.sub(r"\nQ:", f"\nHINT: {wrong_hint}\nQ:", prompt)
        it["meta"]["cf_edit"] = "added_conflicting_hint_wrong_element"
        it["meta"]["cf_hint"] = wrong_hint
        return it

    def _wm_maint_add_trailing_noise(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        """
        Disputes LAST_TOKEN_HEURISTIC: add NOTE: <distractor> after SEQ line.
        The model might copy this trailing token instead of retrieving from memory.
        """
        prompt = it["prompt"]
        gold = _gold(it)
        
        # Pick a distractor that is NOT the gold
        distractors = ["X", "Y", "Z", "W"]
        distractor = next((d for d in distractors if d != gold), "X")
        
        # Insert NOTE: <distractor> after the SEQ line
        it["prompt"] = re.sub(r"^(SEQ:\s*.+)$", f"\\1\nNOTE: {distractor}", prompt, flags=re.M)
        it["meta"]["cf_edit"] = "added_trailing_noise_token"
        it["meta"]["cf_distractor"] = distractor
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

    def _inh_add_conflicting_hint(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Adds a HINT line showing the query symbol (wrong answer) to test copy-bias inhibition.
        # The model must suppress this explicit distractor cue and apply the mapping.
        prompt = it["prompt"]
        m = re.search(r"^Q:\s*(\S+)", prompt, flags=re.M)
        if not m:
            return it
        query_sym = m.group(1)
        # Insert HINT: <query_sym> before the Q: line
        it["prompt"] = prompt.replace(f"Q: {query_sym}", f"HINT: {query_sym}\nQ: {query_sym}")
        it["meta"]["cf_edit"] = "added_conflicting_hint"
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
    
    def _mc_add_conflicting_hint(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        """
        Adds a HINT line showing a WRONG meaning to test copy-bias inhibition.
        For vocabulary format:
            DICT: wug=BLUE, toma=YELLOW, dax=RED, blicket=GREEN
            WORD: dax
            A:YELLOW B:GREEN C:BLUE D:RED
            Q: meaning(WORD)=
            A:
        We insert HINT: <wrong_meaning> before the Q: line.
        The model must suppress this explicit distractor cue and apply the mapping.
        """
        prompt = it["prompt"]
        gold = _gold(it)  # e.g., "D" for the correct option letter
        
        # Parse the options to find wrong meanings
        try:
            letters, mapping = self._parse_mc_options(prompt)
        except ValueError:
            return it
        
        if gold not in mapping:
            return it
        
        # Pick a wrong letter (different from the correct one)
        wrong_letters = [L for L in letters if L != gold]
        if not wrong_letters:
            return it
        
        wrong_hint = rng.choice(wrong_letters)
        
        # Insert HINT: <wrong_letter> before the Q: line
        it["prompt"] = prompt.replace("Q: meaning(WORD)=", f"HINT: {wrong_hint}\nQ: meaning(WORD)=")
        it["meta"]["cf_edit"] = "added_conflicting_hint_wrong_letter"
        it["meta"]["cf_hint"] = wrong_hint
        return it

    def _mc_add_irrelevant_dict_entry(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        """
        Adds an irrelevant DICT entry to test length heuristic.
        For vocabulary format, we add an extra word→meaning pair that is never queried.
        This increases prompt length without changing the answer.
        """
        prompt = it["prompt"]
        
        # Parse existing DICT entries
        m = re.search(r"^DICT:\s*(.+)$", prompt, flags=re.M)
        if not m:
            return it
        
        dict_str = m.group(1)
        # Extract existing words to avoid collision
        existing_words = re.findall(r"(\w+)=", dict_str)
        existing_meanings = re.findall(r"=(\w+)", dict_str)
        
        # Generate novel nonsense words that don't collide
        novel_words = ["zorp", "flink", "glorp", "snib", "plonk", "quib"]
        novel_meanings = ["PURPLE", "ORANGE", "PINK", "CYAN", "GRAY", "WHITE"]
        
        # Pick one that doesn't exist
        new_word = next((w for w in novel_words if w not in existing_words), None)
        new_meaning = next((m for m in novel_meanings if m not in existing_meanings), None)
        
        if not new_word or not new_meaning:
            return it
        
        # Add the new entry to the DICT line
        new_dict_str = f"{dict_str}, {new_word}={new_meaning}"
        it["prompt"] = prompt.replace(f"DICT: {dict_str}", f"DICT: {new_dict_str}")
        it["meta"]["cf_edit"] = "added_irrelevant_dict_entry"
        it["meta"]["cf_extra_entry"] = f"{new_word}={new_meaning}"
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
    
    def _meta_add_noisy_illustration(self, it: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        """
        Disputes anchoring / boundary discipline:
        Insert a misleading HINT line between the final Q and A.
        """
        prompt = it["prompt"]

        # Match the final metacognitive question block:
        # group(1): the Q line (with trailing newline)
        # group(2): the A: line
        m = re.search(
            r"(Q:\s*How many were correct\?\s*\n)(A:)",
            prompt
        )
        if not m:
            return it

        hint = "HINT: 11\n"

        # Reconstruct prompt with hint inserted between Q and A
        it["prompt"] = (
            prompt[: m.start(2)] +
            hint +
            prompt[m.start(2):]
        )

        it["meta"]["cf_edit"] = "added_noisy_hint_between_q_and_a"
        return it


# -------------------------
# Dataset generation
# -------------------------

def generate_dataset(n_samples_per_skill: int = 2500, output_path: str = None, seed: int = 42, shuffle: bool = False):
    rng = random.Random(seed)

    # 1) Build a dataset using BenchmarkBuilder
    spec = BenchmarkSpec(
        seed=seed,       
        n_per_skill={
            "relational_reasoning": n_samples_per_skill,
            "rule_induction": n_samples_per_skill,
            "working_memory_maintenance": n_samples_per_skill,
            "working_memory_manipulation": n_samples_per_skill,
            "quantitative_reasoning": n_samples_per_skill,
            "cognitive_control_inhibition": n_samples_per_skill,
            "symbol_recognition": n_samples_per_skill,
            "vocabulary": n_samples_per_skill,
            "phonological_awareness": n_samples_per_skill,
            "instruction_comprehension": n_samples_per_skill,
            "fine_motor_proxy": n_samples_per_skill,
            "metacognitive_self_estimation": n_samples_per_skill,
        },
        shuffle=shuffle,
    )

    builder = BenchmarkBuilder(spec)
    base_items = builder.generate(progress=True)

    unique_bases = {}
    for it in base_items:
        unique_bases.setdefault(it["prompt"], it)
    base_items = list(unique_bases.values())

    # 2) Create the counterfactual transformer
    tfm = CounterfactualTransformer()

    # 3) For each base sample, generate ONE counterfactual by applying a randomly chosen applicable shortcut
    pairs = []
    for item in tqdm(base_items, desc="Performing counterfactual transformations"):
        shortcuts = tfm.applicable_shortcuts(item)
        if not shortcuts:
            continue

        chosen = rng.choice(shortcuts)
        cf_item = tfm.transform(item, shortcut=chosen, rng=rng)

        pair = {"base": item, "cf": cf_item}
        if cf_item["meta"].get("cf_edit") is not None and pair not in pairs:
            pairs.append(pair)

    
    # 4) (Optional) Save output
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(pairs, f, indent=2)

    # Print counts per skill
    skill_counts = Counter(item["base"]["skill"] for item in pairs)
    print("\n=== Samples per skill ===")
    for skill, count in sorted(skill_counts.items()):
        print(f"  {skill}: {count}")

    print(f"Returned {len(pairs)} entries")

    return pairs

# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":

    # Generate the dataset with the specified seed (if output path is specified, we save the dataset)
    generate_dataset(n_samples_per_skill = 2500, output_path = "data/skillbench_2500.json", seed=42)