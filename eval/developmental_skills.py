"""
Unified text-only CLM benchmark generator, with *implementation logic* traced back (inline)
to what the original school readiness / entry-assessment resources describe.

All skills from the prior COMPLETENESS CHECK are covered:

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
11. Social-emotional awareness
12. Metacognitive self-estimation

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
from typing import Dict, List, Any, Optional, Protocol, Tuple
import random
import hashlib



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
    """

    name = "relational_reasoning"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        relation: str = ">",
        query: str = "min",      # "min" or "max"
        chain_len: int = 2,      # short chain = closer to “basic concept” comparisons, less WM-heavy
    ):
        self.symbols = symbols or ["A", "B", "C"]
        self.relation = relation
        self.query = query
        self.chain_len = chain_len

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            # Choose a total order so that "comparison" is well-defined and unambiguous.
            order = _shuffle(rng, self.symbols)
            k = min(self.chain_len, len(order) - 1)

            # Facts mirror “comparison relationships” (e.g., bigger-than) in a stripped symbolic form.
            involved = order[: k + 1]
            facts = [f"{involved[i]} {self.relation} {involved[i+1]}" for i in range(k)]

            # Query asks to apply the inferred ordering (min/max), analogous to choosing the “smaller/larger”.
            if self.query == "min":
                gold = involved[-1]
                q = f"Q: min({','.join(involved)})="
            elif self.query == "max":
                gold = involved[0]
                q = f"Q: max({','.join(involved)})="
            else:
                raise ValueError("query must be 'min' or 'max'")

            prompt = "\n".join(facts + [q, "A:"])
            meta = {"order": order, "facts": facts, "involved": involved, "query": self.query}
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
    - Compare two small sets represented as repeated neutral tokens (●●● vs ●●).
    - This keeps the cognitive demand “which set is larger?” aligned with basic numeracy concepts,
      without adding language or world knowledge.

    Refinement:
    - Keep numbers tiny to reduce arithmetic-algorithm confounds and focus on cardinality comparison.
    """

    name = "quantitative_reasoning"

    def __init__(
        self,
        dot: str = "●",
        min_n: int = 1,
        max_n: int = 5,
        labels: Optional[List[str]] = None,
        allow_equal: bool = False,
    ):
        self.dot = dot
        self.min_n = min_n
        self.max_n = max_n
        self.labels = labels or ["X", "Y"]
        self.allow_equal = allow_equal

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        X, Y = self.labels
        for _ in range(n):
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
        force_derangement: bool = True,
    ):
        self.symbols = symbols or ["A", "B", "C"]
        self.force_derangement = force_derangement

    def _mapping(self, rng: random.Random) -> Dict[str, str]:
        syms = self.symbols
        perm = _shuffle(rng, syms)
        if self.force_derangement and len(syms) > 1:
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
            m = self._mapping(rng)
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
            prompt = f"ALPH: {{{','.join(alph)}}}\nQ: member({q})=YES/NO\nA:"
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
        set1: Optional[List[str]] = None,
        set2: Optional[List[str]] = None,
    ):
        self.set1 = set1 or ["A", "C", "E"]
        self.set2 = set2 or ["B", "D", "F"]

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        pool = self.set1 + self.set2
        for _ in range(n):
            x = _choice(rng, pool)
            # "Follow instruction → execute" matches TADI direct-task framing, but simplified.
            gold = "YES" if x in self.set1 else "NO"
            prompt = f"RULE: YES if x in {{{','.join(self.set1)}}}, else NO.\nQ: x={x}\nA:"
            meta = {"set1": self.set1, "set2": self.set2, "x": x}
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
# 12) Social-emotional awareness
# =========================

class SocialEmotionalAwarenessSkill:
    """
    Resource alignment:
    - SDQ: 25 items divided into 5 scales:
      emotional symptoms, conduct problems, hyperactivity/inattention, peer problems, prosocial behaviour
      https://www.sdqinfo.org/a0.html
    - BiB Starting School includes SDQ for social/emotional health:
      https://wellcomeopenresearch.org/articles/5-47/v1/pdf

    Implementation alignment (refined to better match SDQ’s structure):
    - SDQ is not a “single correct answer about norms”; it’s a structured *categorization* into scales.
    - We therefore implement a controlled proxy: define 5 SDQ-like scale labels as sets of action tokens,
      then ask which scale a behavior belongs to.
    - This mirrors the SDQ’s “items belong to one of 5 scales” structure, but avoids culture/language nuance.
    """

    name = "social_emotional_awareness"

    def __init__(
        self,
        # SDQ has 5 subscales; we mirror that with 5 controlled token sets.
        emotional: Optional[List[str]] = None,
        conduct: Optional[List[str]] = None,
        hyperactivity: Optional[List[str]] = None,
        peer: Optional[List[str]] = None,
        prosocial: Optional[List[str]] = None,
    ):
        self.emotional = emotional or ["WORRY", "FEAR", "SAD"]
        self.conduct = conduct or ["FIGHT", "LIE", "STEAL"]
        self.hyperactivity = hyperactivity or ["RESTLESS", "IMPULSIVE", "DISTRACT"]
        self.peer = peer or ["LONELY", "BULLIED", "ISOLATE"]
        self.prosocial = prosocial or ["SHARE", "HELP", "KIND"]

        self.scale_map: Dict[str, List[str]] = {
            "EMO": self.emotional,
            "CON": self.conduct,
            "HYP": self.hyperactivity,
            "PEER": self.peer,
            "PRO": self.prosocial,
        }

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        all_actions = [(scale, act) for scale, acts in self.scale_map.items() for act in acts]

        for _ in range(n):
            gold_scale, act = _choice(rng, all_actions)

            # Make the mapping explicit (like SDQ scale definitions), then query one "item".
            # This keeps the task about *scale assignment structure* rather than real-world norms.
            defs = " ; ".join([f"{k}={{{','.join(v)}}}" for k, v in self.scale_map.items()])
            prompt = f"SCALES: {defs}\nITEM: {act}\nQ: scale(ITEM)=\nA:"
            meta = {"item": act, "gold_scale": gold_scale, "note": "SDQ-like 5-scale categorization proxy"}
            out.append(_fmt_item(self.name, prompt, gold_scale, meta))
        return out


# =========================
# 13) Metacognitive self-estimation
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
        totals: Optional[List[int]] = None,
        round_places: int = 2,
        avoid_trivial: bool = True,   # avoid always d=0
    ):
        self.totals = totals or [20, 25, 30, 33, 40]  # includes NEPS-like example counts (e.g., 33)
        self.round_places = round_places
        self.avoid_trivial = avoid_trivial

    def _fmt_d(self, d: float) -> str:
        # Fixed rounding -> stable target format.
        return f"{d:.{self.round_places}f}"

    def generate(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            total = _choice(rng, self.totals)
            correct = rng.randint(0, total)

            # estimated correct is the metacognitive judgment described in NEPS (post-diction)
            est = rng.randint(0, total)

            if self.avoid_trivial:
                # Encourage non-zero deviations some fraction of time.
                # (NEPS deviation score interpretation is most meaningful when d != 0.)
                for _try in range(20):
                    if est != correct:
                        break
                    est = rng.randint(0, total)

            # NEPS definition: d = (N_estimated/N_items) - (N_correct/N_items)
            d = (est / total) - (correct / total)
            gold = self._fmt_d(d)

            prompt = (
                f"TEST: items={total}\n"
                f"ACTUAL: correct={correct}\n"
                f"SELF-EST: estimated_correct={est}\n"
                f"Q: deviation_d=(estimated/items)-(correct/items)=\n"
                f"A:"
            )
            meta = {
                "total": total,
                "correct": correct,
                "estimated": est,
                "d": d,
                "interpretation": "d>0 overestimate; d<0 underestimate; d=0 perfect",
                "source": "NEPS procedural metacognition deviation score definition",
            }
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
    All skills are separate classes (including WM maintenance vs manipulation).
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
            self._make(SocialEmotionalAwarenessSkill, "social_emotional_awareness"),
            self._make(MetacognitiveSelfEstimationSkill, "metacognitive_self_estimation"),
        ]

        default_counts = {s.name: 100 for s in self.skills}
        self.n_per_skill = spec.n_per_skill or default_counts

    def _make(self, cls, key: str):
        kwargs = self.overrides.get(key, {})
        return cls(**kwargs)

    def generate(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for sk in self.skills:
            n = self.n_per_skill.get(sk.name, 0)
            if n <= 0:
                continue
            items.extend(sk.generate(n=n, rng=self.rng))
        if self.spec.shuffle:
            self.rng.shuffle(items)
        return items


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    spec = BenchmarkSpec(
        seed=42,
        n_per_skill={
            "relational_reasoning": 5,
            "rule_induction": 5,
            "working_memory_maintenance": 5,
            "working_memory_manipulation": 5,
            "quantitative_reasoning": 5,
            "cognitive_control_inhibition": 5,
            "symbol_recognition": 5,
            "vocabulary": 5,
            "phonological_awareness": 5,
            "instruction_comprehension": 5,
            "fine_motor_proxy": 5,
            "social_emotional_awareness": 5,
            "metacognitive_self_estimation": 5,
        },
        overrides={
            "relational_reasoning": {"symbols": ["A", "B", "C", "D"], "chain_len": 3, "query": "min"},
            "working_memory_maintenance": {"seq_len": 6, "ask_index": 4},
            "working_memory_manipulation": {"seq_len": 5},
            "vocabulary": {"n_options": 4},
            "phonological_awareness": {"k_suffix": 2, "n_options": 3},
            "fine_motor_proxy": {"steps": 8},
            "metacognitive_self_estimation": {"totals": [20, 33, 40], "round_places": 2},
        },
        shuffle=True,
    )

    ds = BenchmarkBuilder(spec).generate()
    for ex in ds[:10]:
        print("\n---")
        print("SKILL:", ex["skill"])
        print(ex["prompt"], ex["completion"])
