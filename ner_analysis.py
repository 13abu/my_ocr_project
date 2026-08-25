"""
ner_analysis.py

IRP Instrument 2: "a named entity analysis mapping the grammar of
personhood versus provenance."

Off-the-shelf spaCy NER confirmed unreliable on this material
(documented finding: "Suwahil" tagged PERSON in "a slave of Suwahil
named Almas", "Khalifah" flip-flopping PERSON/GPE for the same
referent, "Khazrah" tagged PERSON, bare "Shaikh" tagged PERSON).

This script does two things:
  1. Runs vanilla spaCy NER and records every hit as a baseline
     (ner_baseline.csv) -- so the raw failure stays documented and
     citable exactly as before.
  2. Applies a lightweight RULE LAYER on top of spaCy's output --
     no retraining, no fine-tuning, just proximity and morphology --
     to test whether the personhood/provenance distinction is
     recoverable from surface patterns alone (ner_corrected.csv).

This is deliberately NOT the NYU Abu Dhabi approach (Kapan,
Kirmizialtin & Wrisley 2022), which fine-tunes spaCy's NER component
on a hand-annotated training set built from BPR ledgers + Lorimer's
Gazetteer. That's a stronger method but needs real training data this
project doesn't have. What's here is a much cheaper test: can 3-4
simple surface rules recover most of what fine-tuning would fix? If
yes, that's evidence the personhood/provenance grammar is not just a
finding about NER's failure -- it's a legible, rule-describable
pattern in the documents themselves. If the rules only fix a little,
that's also a real, citable result (motivates the harder fine-tuning
path as future work, with a concrete number for how much it would
need to close).

SETUP (run once):
    pip install spacy
    python -m spacy download en_core_web_sm

Run from the same folder as your other analysis scripts, next to
transcriptions/.

Outputs (written to the current folder):
  ner_corrected.csv  -- every entity spaCy found, with spaCy's own
                        label PLUS a derived_role column showing what
                        the rule layer thinks it actually is, and
                        WHY (which rule fired, or none). This is the
                        only output file -- it already contains
                        spaCy's raw label (spacy_label) alongside the
                        correction, so a separate baseline file would
                        just be a redundant subset of these columns.
  Console summary: how many entities each rule reclassified, and a
  KWIC-style listing of every reclassification so each one can be
  checked against the real document before being trusted.
"""

import re
import csv
from pathlib import Path
from collections import defaultdict, Counter

import spacy

TRANSCRIPTIONS_DIR = Path("./transcriptions")
CORRECTED_CSV = Path("./ner_corrected.csv")

# Restrict to a manageable, checkable set of files by default -- NER
# at full corpus scale (110 files) is a bigger run than the pilot
# needs right now. Edit this list, or set to None to run everything.
FILE_ALLOWLIST = None  # e.g. ["IOR_R_15_1_208_0029_7r", "IOR_R_15_1_208_0031_8r"]


# ==========================================
# SHARED: transcript loading (same logic as the other scripts, so all
# instruments see identical body text for identical files)
# ==========================================

BODY_HEADER_PATTERN = re.compile(
    r"^\s*#{0,3}\s*\*{0,2}\s*"
    r"(TRANSCRIPTION|TRANSCRIPT|MAIN\s+BODY\s+TEXT|MAIN\s+TEXT|MAIN\s+BODY|BODY\s+TEXT)"
    r"\s*(\([^)]*\))?\s*:?\s*\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
MARGINALIA_HEADER_PATTERN = re.compile(
    r"^\s*#{0,3}\s*\*{0,2}\s*MARGINALIA\s*:?\s*\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
BLANK_LINE_PATTERN = re.compile(r"\n\s*\n")
MIN_BODY_BEFORE_MARGINALIA = 50
MIN_BODY_AFTER_MARGINALIA = 50


def extract_body_text(raw: str):
    body_match = BODY_HEADER_PATTERN.search(raw)
    if body_match:
        return raw[body_match.end():].strip(), "body_header"
    marg_match = MARGINALIA_HEADER_PATTERN.search(raw)
    if marg_match:
        blank_match = BLANK_LINE_PATTERN.search(raw, marg_match.end())
        if blank_match:
            remainder = raw[blank_match.end():].strip()
            if len(remainder) >= MIN_BODY_AFTER_MARGINALIA:
                return remainder, "marginalia_structural"
        text_before = raw[:marg_match.start()].strip()
        if len(text_before) >= MIN_BODY_BEFORE_MARGINALIA:
            return text_before, "body_before_marginalia"
        return None, "ambiguous"
    if raw.strip():
        return raw.strip(), "no_marginalia_found"
    return None, "empty_file"


def load_transcripts(directory: Path, allowlist=None):
    texts = {}
    for path in sorted(directory.glob("*.txt")):
        if allowlist is not None and path.stem not in allowlist:
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        body, _method = extract_body_text(raw)
        if body is not None:
            texts[path.stem] = body
    return texts


# ==========================================
# RULE LAYER -- surface patterns, not retraining
# ==========================================

# Words that signal "what follows is provenance/origin, not a new
# person" when they sit directly before an entity that comes right
# after a name-like token. Kept short and literal on purpose -- these
# are exactly the connective words this corpus's naming grammar uses
# ("a slave OF Suwahil", "Almas BIN Husain", "Khazur BINT Abdullah").
ORIGIN_MARKERS = {"of", "from", "native", "at"}
PATRONYMIC_MARKERS = {"bin", "ibn", "bint", "binti", "walad"}

# Bare titles that spaCy sometimes tags PERSON on their own, with no
# actual name attached -- a title alone is not a person.
TITLE_ONLY_WORDS = {"shaikh", "sheikh", "sheik", "sir", "mr", "mr.", "dr", "dr.", "miss", "mrs", "mrs."}

# Kinship/status nouns and biographical-action verbs this corpus's
# testimony/manumission genre uses right before a name-plus-origin
# construction ("a SLAVE of Suwahil", "a MAN of Khazrah", "KIDNAPPED
# from Zanzibar", "SOLD to a man of..."). None of these are
# capitalized and none get their own PERSON tag from spaCy, which is
# exactly why the origin-marker rule needs this list rather than
# relying on capitalization/PERSON-adjacency alone.
BIOGRAPHICAL_MARKERS = {
    "slave", "man", "woman", "native", "subject", "wife", "daughter",
    "son", "master", "servant", "named", "employee", "labourer",
    "kidnapped", "sold", "born", "purchased", "imported", "taken",
}


def token_before(doc, ent, n=1):
    """Returns up to n tokens immediately preceding an entity span."""
    start_i = ent.start
    lo = max(0, start_i - n)
    return doc[lo:start_i]


def classify_entity(doc, ent):
    """
    Returns (derived_role, rule_fired). derived_role is spaCy's own
    label if no rule overrides it.
    """
    preceding = token_before(doc, ent, n=1)
    preceding_text = preceding.text.lower().strip() if len(preceding) else ""

    # Rule 1: bare title, no name attached (single-token PERSON entity
    # that IS a title word) -- reclassify as title-only, not a person.
    if ent.label_ == "PERSON" and ent.text.strip().lower() in TITLE_ONLY_WORDS:
        return "title_only (not a person)", "rule_title_only"

    # Rule 2: origin marker -- entity immediately preceded by "of" /
    # "from" / "native", in a BIOGRAPHICAL context. Catches "a slave of
    # Suwahil named Almas", "kidnapped from Zanzibar", "a man of
    # Khazrah". Does NOT fire on ordinary possessive/prepositional "of"
    # ("copy of the Death Certificate", "the Ruler of Ajman", "Ministry
    # of Foreign Affairs") -- that's most of what "of" does in English,
    # and firing on all of it (an earlier bug here) destroyed precision.
    # Gate: fire only if a token in the 3 words before the entity is
    # either already PERSON-tagged by spaCy, or one of a small set of
    # kinship/status/biographical-action words this corpus's testimony
    # genre actually uses around a name-plus-origin construction.
    if preceding_text in ORIGIN_MARKERS:
        two_before = token_before(doc, ent, n=3)
        two_before_lower = {t.text.lower() for t in two_before}
        looks_biographical = (
            any(t.ent_type_ == "PERSON" for t in two_before)
            or bool(two_before_lower & BIOGRAPHICAL_MARKERS)
        )
        if looks_biographical:
            return "origin_marker (place, not a new person)", "rule_origin_marker"

    # Rule 3: patronymic marker -- "bin"/"bint"/etc immediately before
    # the entity means this token is part of ONE person's own name
    # chain, not a second, separate person.
    if preceding_text in PATRONYMIC_MARKERS:
        return "patronymic_component (same person's name)", "rule_patronymic"

    return ent.label_, "none"


def find_inconsistent_tags(all_hits):
    """Surface forms that got different spaCy labels in different
    mentions across the corpus -- e.g. Khalifah as PERSON in one
    sentence, GPE in another. Returns {surface_text: {labels_seen}}."""
    by_text = defaultdict(set)
    for hit in all_hits:
        by_text[hit["entity_text"]].add(hit["spacy_label"])
    return {text: labels for text, labels in by_text.items() if len(labels) > 1}


# ==========================================
# MAIN
# ==========================================

def run(texts: dict, nlp):
    corrected_rows = []
    rule_counts = Counter()

    for name, text in texts.items():
        doc = nlp(text)
        for ent in doc.ents:
            snip_start = max(0, ent.start_char - 40)
            snip_end = min(len(text), ent.end_char + 40)
            snippet = text[snip_start:snip_end].replace("\n", " ")

            derived_role, rule_fired = classify_entity(doc, ent)
            rule_counts[rule_fired] += 1
            corrected_rows.append({
                "document": name,
                "entity_text": ent.text,
                "spacy_label": ent.label_,
                "derived_role": derived_role,
                "rule_fired": rule_fired,
                "changed": derived_role != ent.label_,
                "snippet": snippet,
            })

    with open(CORRECTED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["document", "entity_text", "spacy_label", "derived_role",
                           "rule_fired", "changed", "snippet"]
        )
        writer.writeheader()
        writer.writerows(corrected_rows)
    print(f"Wrote {len(corrected_rows)} rows to {CORRECTED_CSV}")

    n_changed = sum(1 for r in corrected_rows if r["changed"])
    print()
    print("=" * 70)
    print("RULE-LAYER SUMMARY")
    print("=" * 70)
    print(f"  Total entities (spaCy):       {len(corrected_rows)}")
    print(f"  Reclassified by a rule:       {n_changed} ({n_changed / len(corrected_rows) * 100:.0f}%)" if corrected_rows else "  (no entities found)")
    for rule, count in rule_counts.most_common():
        print(f"    {rule:<25} {count}")

    print()
    print("=" * 70)
    print("EVERY RECLASSIFICATION (check each against the real document)")
    print("=" * 70)
    for r in corrected_rows:
        if r["changed"]:
            print(f"  [{r['document']}] '{r['entity_text']}'  spaCy={r['spacy_label']} -> {r['derived_role']}  ({r['rule_fired']})")
            print(f"      ...{r['snippet']}...")

    inconsistent = find_inconsistent_tags(corrected_rows)
    print()
    print("=" * 70)
    print(f"SURFACE FORMS SPACY TAGGED INCONSISTENTLY ({len(inconsistent)} found)")
    print("=" * 70)
    print("These are the same word getting different labels in different")
    print("mentions -- e.g. Khalifah as PERSON once, GPE elsewhere. The rule")
    print("layer above does NOT resolve these; they need a manual check.")
    for text, labels in inconsistent.items():
        print(f"  '{text}': tagged as {sorted(labels)}")


if __name__ == "__main__":
    if not TRANSCRIPTIONS_DIR.exists():
        raise SystemExit(f"Could not find {TRANSCRIPTIONS_DIR}. Run this from your project folder.")

    print("Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")

    texts = load_transcripts(TRANSCRIPTIONS_DIR, allowlist=FILE_ALLOWLIST)
    print(f"Loaded {len(texts)} transcript(s)"
          + (f" (allowlist active: {len(FILE_ALLOWLIST)} file(s))" if FILE_ALLOWLIST else " (full corpus)"))
    print()

    run(texts, nlp)