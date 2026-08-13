"""
register_analysis.py

Second-pass computational tool for the affective lexicon instrument --
tests two things plain word-frequency counting can't distinguish:

  1. INTENSITY: is an affect word bare ("regret to inform") or
     intensified ("profound regret", "deeply regret")? A small
     intensifier lexicon plus a proximity check -- same level of
     tooling as lexicon_analysis.py, no classifier involved.

  2. FORMULA vs SINGULARITY: is the sentence containing an affect/death
     term boilerplate repeated near-verbatim across multiple documents
     (an administrative template), or phrasing found nowhere else in
     the corpus? Measured with plain string similarity (difflib,
     stdlib only) -- not semantic similarity, not a language model. A
     high max-similarity-to-any-OTHER-document's-sentence score means
     the document is reusing a repeated template; a low score means
     the phrasing is singular.

NEITHER OF THESE IS SENTIMENT ANALYSIS. Off-the-shelf sentiment tools
(VADER, TextBlob) score polarity on modern conversational text and
would not reliably separate "profound regret" from "we regret to
inform you" -- both register as mild negative sentiment on the same
underlying word, and a corporate death-notice letter (with words like
"deceased", "dysentery") could easily score MORE negative than an
elaborated mourning notice, which would actively invert the finding
this project is trying to test. These two measures instead
operationalize the specific distinction under discussion directly:
intensity (an intensifier lexicon, checkable against real hits) and
repetition (literal string overlap, not a judgment about feeling).

Treat every output as a starting point to read against the real
document, not a finished finding -- same discipline as
lexicon_analysis.py.

Run this from the same folder as lexicon_analysis.py, next to your
transcriptions/ directory.
"""

import re
import csv
import difflib
from pathlib import Path
from collections import defaultdict

TRANSCRIPTIONS_DIR = Path("./transcriptions")
INTENSIFIER_CSV = Path("./intensifier_results.csv")
FORMULA_CSV = Path("./formula_results.csv")


# ==========================================
# Transcript loading -- identical logic to lexicon_analysis.py, so
# both scripts see exactly the same body text for the same files.
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


def load_transcripts(directory: Path):
    texts = {}
    for path in sorted(directory.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        body, _method = extract_body_text(raw)
        if body is not None:
            texts[path.stem] = body
    return texts


# ==========================================
# PART 1: INTENSITY
# ==========================================

# Deliberately the same "mourning" category as lexicon_analysis.py,
# minus "profound" -- which moves to the intensifier list below, since
# it's exactly the word from your f.116r anchor phrase ("profound
# regret") that started this whole question.
AFFECT_TERMS = [
    "regret", "sorrow", "condolence", "condolences", "mourning",
    "grief", "lamented", "melancholy", "bereavement", "sympathy",
    "shocked", "tragic",
]

INTENSIFIERS = [
    "profound", "profoundly", "deeply", "deep", "sincere", "sincerest",
    "heartfelt", "genuine", "great", "utmost", "greatest", "extreme",
    "grievous", "unfeigned",
]

WINDOW = 3  # words before the affect term to check for an intensifier


def find_intensified_hits(texts: dict):
    results = []
    for name, text in texts.items():
        words = text.split()
        lower_words = [w.lower().strip(".,;:") for w in words]
        for i, w in enumerate(lower_words):
            if w in AFFECT_TERMS:
                start = max(0, i - WINDOW)
                preceding = lower_words[start:i]
                intensifier_found = next((t for t in INTENSIFIERS if t in preceding), None)
                snip_start = max(0, i - WINDOW)
                snip_end = min(len(words), i + WINDOW + 1)
                snippet = " ".join(words[snip_start:snip_end])
                results.append({
                    "document": name,
                    "term": w,
                    "intensified": bool(intensifier_found),
                    "intensifier": intensifier_found or "",
                    "snippet": snippet,
                })
    return results


def export_intensifier_csv(results, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "term", "intensified", "intensifier", "snippet"])
        for r in results:
            writer.writerow([r["document"], r["term"], r["intensified"], r["intensifier"], r["snippet"]])


def print_intensity_summary(results):
    by_doc = defaultdict(lambda: {"total": 0, "intensified": 0})
    for r in results:
        by_doc[r["document"]]["total"] += 1
        if r["intensified"]:
            by_doc[r["document"]]["intensified"] += 1

    print("=" * 70)
    print("INTENSITY SUMMARY (per document)")
    print("=" * 70)
    if not by_doc:
        print("  No affect-term hits found.")
    for name, counts in sorted(by_doc.items()):
        ratio = counts["intensified"] / counts["total"] if counts["total"] else 0
        print(f"  {name:<45} {counts['intensified']}/{counts['total']} intensified ({ratio:.0%})")


# ==========================================
# PART 2: FORMULA vs SINGULARITY
# ==========================================

TARGET_TERMS = ["regret", "death", "died", "deceased", "condolence", "wages", "manumit"]

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?;])\s+")


def extract_relevant_sentences(texts: dict):
    """Pull every sentence containing at least one TARGET_TERMS word,
    tagged by which document it came from."""
    sentences = []
    for name, text in texts.items():
        clean = re.sub(r"\s+", " ", text)
        for sent in SENTENCE_SPLIT_PATTERN.split(clean):
            sent_lower = sent.lower()
            if any(term in sent_lower for term in TARGET_TERMS):
                sent_stripped = sent.strip()
                if len(sent_stripped) > 15:  # skip stray fragments
                    sentences.append({"document": name, "sentence": sent_stripped})
    return sentences


def score_formula_vs_singular(sentences, similarity_threshold: float = 0.75):
    """
    For every extracted sentence, find its single most similar OTHER
    sentence FROM A DIFFERENT DOCUMENT (comparing a document against
    itself doesn't tell you anything about template reuse across the
    corpus), using difflib's plain string-overlap ratio. A high score
    means this phrasing is repeated near-verbatim elsewhere in the
    corpus (a template); a low score means it's singular/unique
    phrasing found nowhere else.

    O(n^2) in the number of extracted sentences -- fine at pilot scale
    (dozens to low hundreds of matches); would need a smarter approach
    (shingling/minhash) if this were ever run at thousand-folio scale.
    """
    n = len(sentences)
    for i in range(n):
        best_score = 0.0
        best_doc = None
        best_sentence = None
        for j in range(n):
            if i == j or sentences[i]["document"] == sentences[j]["document"]:
                continue
            ratio = difflib.SequenceMatcher(
                None, sentences[i]["sentence"].lower(), sentences[j]["sentence"].lower()
            ).ratio()
            if ratio > best_score:
                best_score = ratio
                best_doc = sentences[j]["document"]
                best_sentence = sentences[j]["sentence"]
        sentences[i]["max_similarity"] = round(best_score, 3)
        sentences[i]["most_similar_document"] = best_doc or ""
        sentences[i]["most_similar_sentence"] = best_sentence or ""
        sentences[i]["classification"] = (
            "formula/template" if best_score >= similarity_threshold else "singular/unique"
        )
    return sentences


def export_formula_csv(sentences, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "document", "sentence", "max_similarity", "classification",
            "most_similar_document", "most_similar_sentence",
        ])
        for s in sentences:
            writer.writerow([
                s["document"], s["sentence"], s["max_similarity"], s["classification"],
                s["most_similar_document"], s["most_similar_sentence"],
            ])


def print_formula_summary(sentences):
    formula_count = sum(1 for s in sentences if s["classification"] == "formula/template")
    unique_count = len(sentences) - formula_count

    print()
    print("=" * 70)
    print("FORMULA vs SINGULARITY SUMMARY")
    print("=" * 70)
    print(f"  Total relevant sentences: {len(sentences)}")
    print(f"  Formula/template (similarity >= threshold): {formula_count}")
    print(f"  Singular/unique: {unique_count}")

    if sentences:
        print()
        print("  Top 10 highest-similarity pairs (most likely templates):")
        top = sorted(sentences, key=lambda s: -s["max_similarity"])[:10]
        for s in top:
            print(f"    [{s['max_similarity']:.2f}] {s['document']}: \"{s['sentence'][:80]}\"")
            print(f"           <-> {s['most_similar_document']}: \"{s['most_similar_sentence'][:80]}\"")

        print()
        print("  Bottom 10 lowest-similarity (most singular/unique phrasing):")
        bottom = sorted(sentences, key=lambda s: s["max_similarity"])[:10]
        for s in bottom:
            print(f"    [{s['max_similarity']:.2f}] {s['document']}: \"{s['sentence'][:80]}\"")


if __name__ == "__main__":
    if not TRANSCRIPTIONS_DIR.exists():
        raise SystemExit(
            f"Could not find {TRANSCRIPTIONS_DIR}. "
            f"Run this from the same folder as lexicon_analysis.py."
        )

    texts = load_transcripts(TRANSCRIPTIONS_DIR)
    print(f"Loaded {len(texts)} transcripts from {TRANSCRIPTIONS_DIR}\n")

    # Part 1: intensity
    intensity_results = find_intensified_hits(texts)
    export_intensifier_csv(intensity_results, INTENSIFIER_CSV)
    print_intensity_summary(intensity_results)
    print(f"\nExported {len(intensity_results)} rows to {INTENSIFIER_CSV}")

    # Part 2: formula vs singularity
    sentences = extract_relevant_sentences(texts)
    sentences = score_formula_vs_singular(sentences)
    export_formula_csv(sentences, FORMULA_CSV)
    print_formula_summary(sentences)
    print(f"\nExported {len(sentences)} rows to {FORMULA_CSV}")