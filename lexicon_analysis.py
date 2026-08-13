"""
lexicon_analysis.py

First-pass computational tool for the affective lexicon instrument
(IRP instrument 1: "a custom affective lexicon mapping the
administrative vocabulary that permits or forecloses grief").

Run this from inside your project folder, next to transcribe_bulk.py.
It reads every .txt file in TRANSCRIPTIONS_DIR and produces:

  1. Overall word frequency (for building/refining your lexicon categories)
  2. A KWIC (keyword-in-context) concordance for any term you search
  3. A per-document lexicon score, tagging each transcript by which
     vocabulary cluster it leans toward

Nothing here uses machine learning — it's plain word counting and
string matching, which is the right level of tooling for a pilot-scale
corpus. Treat every output as a starting point to read against the
real document, not a finished finding.
"""

import re
from pathlib import Path
from collections import Counter

TRANSCRIPTIONS_DIR = Path("./transcriptions")

# ==========================================
# LEXICON CATEGORIES — edit these to match your actual close-reading
# ==========================================
# Start with a small, deliberately chosen set of terms drawn from
# documents you've already read (Victoria mourning vs. anti-trafficking
# vs. administrative-neutral). Expand only after checking real hits.

LEXICON = {
    "mourning": [
        "regret", "sorrow", "condolence", "condolences", "mourning",
        "grief", "lamented", "melancholy", "bereavement", "sympathy",
        "shocked", "tragic", "profound",  # "profound" — the f.116r anchor phrase
    ],
    "administrative_neutral": [
        "claim", "settlement", "instructed", "acknowledge", "forwarded",
        "requested", "report", "action", "case", "reference",
        # Deliberately NOT adding "letter", "beg", "copy", "compliments" —
        # these are near-universal epistolary formulas present in almost
        # every document regardless of category, so they'd inflate every
        # score roughly equally rather than actually discriminating
        # between categories.
    ],
    "trafficking_enforcement": [
        "slave", "slaves", "proclamation", "trade", "traffic",
        "prohibiting", "forbid", "forbidden", "suppress",
        "manumission", "manumitted", "kidnapped", "diving", "certificate",
        "re-enslave", "re-enslavement", "fear", "beaten",
        # "beaten" sits here for now, but really belongs to a distinct
        # ill-treatment/violence-testimony category if that becomes
        # worth separating out later — it's about conditions of
        # enslavement, not the trafficking/enforcement apparatus itself.
    ],
    "monetary_valuation": [
        "rupees", "rs", "price", "advance", "debt", "wages", "compensation",
        "settlement", "paid", "unpaid", "owed", "sold", "purchased", "purchase",
    ],
}

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "is", "was", "were",
    "be", "on", "for", "with", "as", "at", "by", "that", "this",
    "it", "his", "her", "he", "she", "i", "you", "your", "from",
    "or", "not", "has", "have", "had", "will", "would", "which",
}


# The model's header wording after MARGINALIA is essentially
# unpredictable — observed so far: TRANSCRIPTION, TRANSCRIPT, MAIN TEXT,
# MAIN BODY TEXT, MAIN BODY, BODY TEXT, TRANSLATION, TRANSCRIBED
# DOCUMENT, BODY, BODY TRANSCRIPTION, and a bare "---" rule with no
# header at all. Rather than keep enumerating header phrases, we detect
# the MARGINALIA block STRUCTURALLY: it ends at the first blank line,
# and everything past that blank line is body — whatever the next
# header happens to say. A stray one-line header sometimes ends up as
# the first line of the returned body text; that's a few words of
# harmless noise, not worth chasing further, and far better than
# losing the whole document to a "no match" every time a new phrasing
# turns up.

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

# How much text has to sit before a MARGINALIA header for us to treat
# it as "body came first" (continuation page) rather than "nothing here".
# Kept low and matched to MIN_BODY_AFTER_MARGINALIA — short notes (a
# few lines) are still real documents and shouldn't be excluded just
# for brevity.
MIN_BODY_BEFORE_MARGINALIA = 50

# How much text has to sit after the marginalia block for it to count
# as real body content, rather than "marginalia was the whole file"
MIN_BODY_AFTER_MARGINALIA = 50


def extract_body_text(raw: str) -> tuple[str | None, str]:
    """
    Isolate the real transcribed document text, dropping the MARGINALIA
    block. Returns (body_text_or_None, method) where `method` records
    which branch matched, for auditing.

    Strategy, in order:
      1. Fast path: a known, exactly-worded body header -> clean strip.
      2. Structural fallback: find where MARGINALIA ends (first blank
         line after it) and take everything past that point, whatever
         the next header says.
      3. Continuation-page check: if MARGINALIA has little/no content
         after it, but substantial text BEFORE it, that leading text
         is the real body (marginalia was appended at the end).
      4. No MARGINALIA found at all -> treat the whole file as body.
    """
    # 1. Fast path — known exact header phrasing
    body_match = BODY_HEADER_PATTERN.search(raw)
    if body_match:
        return raw[body_match.end():].strip(), "body_header"

    marg_match = MARGINALIA_HEADER_PATTERN.search(raw)
    if marg_match:
        # 2. Structural fallback — first blank line after MARGINALIA
        # marks the end of the marginalia block, whatever comes next
        blank_match = BLANK_LINE_PATTERN.search(raw, marg_match.end())
        if blank_match:
            remainder = raw[blank_match.end():].strip()
            if len(remainder) >= MIN_BODY_AFTER_MARGINALIA:
                return remainder, "marginalia_structural"

        # 3. Continuation page — substantial text BEFORE MARGINALIA
        text_before = raw[:marg_match.start()].strip()
        if len(text_before) >= MIN_BODY_BEFORE_MARGINALIA:
            return text_before, "body_before_marginalia"

        return None, "ambiguous"

    # 4. No MARGINALIA header anywhere -> pure continuation, nothing to strip
    if raw.strip():
        return raw.strip(), "no_marginalia_found"

    return None, "empty_file"


def load_transcripts(directory: Path) -> tuple[dict[str, str], list[str], Counter]:
    """
    Load every .txt file, keyed by filename stem, returning only the
    body text. Files that couldn't be parsed at all (genuinely
    ambiguous structure, or empty) are excluded and listed in
    `unmarked`. Also returns a Counter of which parsing method was
    used across the corpus, so you can sanity-check the split (e.g.
    "how many were continuation pages vs. normal headers").
    """
    texts = {}
    unmarked = []
    method_counts = Counter()
    for path in sorted(directory.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        body, method = extract_body_text(raw)
        method_counts[method] += 1
        if body is None:
            unmarked.append(path.name)
        else:
            texts[path.stem] = body
    return texts, unmarked, method_counts


def word_frequency(texts: dict[str, str], top_n: int = 50) -> list[tuple[str, int]]:
    """Overall word frequency across the whole corpus, stopwords removed."""
    counter = Counter()
    for text in texts.values():
        words = re.findall(r"[a-zA-Z']+", text.lower())
        counter.update(w for w in words if w not in STOPWORDS and len(w) > 2)
    return counter.most_common(top_n)


def kwic(texts: dict[str, str], term: str, window: int = 8) -> list[tuple[str, str]]:
    """
    Keyword-in-context: every instance of `term` with surrounding words,
    tagged by which file it came from. Use this to actually read each
    hit in context before trusting a frequency count.
    """
    results = []
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for name, text in texts.items():
        words = text.split()
        lower_words = [w.lower().strip(".,;:") for w in words]
        for i, w in enumerate(lower_words):
            if w == term.lower():
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                snippet = " ".join(words[start:end])
                results.append((name, snippet))
    return results


def score_document(text: str) -> dict[str, int]:
    """Count lexicon-category hits in a single document."""
    lower = text.lower()
    scores = {}
    for category, terms in LEXICON.items():
        count = 0
        for term in terms:
            count += len(re.findall(rf"\b{re.escape(term)}\b", lower))
        scores[category] = count
    return scores


def score_all_documents(texts: dict[str, str]) -> dict[str, dict[str, int]]:
    return {name: score_document(text) for name, text in texts.items()}


def print_report(texts: dict[str, str], unmarked: list[str], method_counts: Counter) -> None:
    print(f"Loaded {len(texts)} transcripts from {TRANSCRIPTIONS_DIR}")

    print("\n  Parsing method breakdown:")
    for method, count in method_counts.most_common():
        print(f"    {method:<25} {count}")

    if unmarked:
        print(
            f"\n  WARNING: {len(unmarked)} file(s) could not be parsed "
            f"(ambiguous structure or empty) and were EXCLUDED:"
        )
        for name in unmarked:
            print(f"    - {name}")
    print()

    print("=" * 60)
    print("TOP 50 WORDS ACROSS THE CORPUS")
    print("=" * 60)
    for word, count in word_frequency(texts):
        print(f"  {word:<20} {count}")

    print()
    print("=" * 60)
    print("PER-DOCUMENT LEXICON SCORES")
    print("=" * 60)
    scores = score_all_documents(texts)
    for name, doc_scores in scores.items():
        dominant = max(doc_scores, key=doc_scores.get)
        total = sum(doc_scores.values())
        if total == 0:
            print(f"  {name:<40} no lexicon hits")
            continue
        print(f"  {name:<40} {doc_scores}  -> leans: {dominant}")


if __name__ == "__main__":
    if not TRANSCRIPTIONS_DIR.exists():
        raise SystemExit(
            f"Could not find {TRANSCRIPTIONS_DIR}. "
            f"Run this from the same folder as transcribe_bulk.py."
        )

    texts, unmarked, method_counts = load_transcripts(TRANSCRIPTIONS_DIR)
    print_report(texts, unmarked, method_counts)

    # Example KWIC usage — uncomment and edit to search a specific term:
    # for name, snippet in kwic(texts, "regret"):
    #     print(f"[{name}] ...{snippet}...")

    print("================KWIC TIME================")
    print("~~~~~~~~~~TERM: REGRET~~~~~~~~~~")
    for name, snippet in kwic(texts, "regret"):
        print(f"[{name}] ...{snippet}...")

    print("~~~~~~~~~~TERM: SLAVE~~~~~~~~~~")
    for name, snippet in kwic(texts, "slave"):
        print(f"[{name}] ...{snippet}...")

    print("~~~~~~~~~~TERM: PROPERTY~~~~~~~~~~")
    for name, snippet in kwic(texts, "property"):
        print(f"[{name}] ...{snippet}...")

    print("~~~~~~~~~~TERM: WAGES~~~~~~~~~~")
    for name, snippet in kwic(texts, "wages"):
        print(f"[{name}] ...{snippet}...")

    print("~~~~~~~~~~TERM: DEATH~~~~~~~~~~")
    for name, snippet in kwic(texts, "death"):
        print(f"[{name}] ...{snippet}...")

    print("~~~~~~~~~~TERM: STRUCK~~~~~~~~~~")
    for name, snippet in kwic(texts, "struck"):
        print(f"[{name}] ...{snippet}...")

    print("~~~~~~~~~~TERM: BEG~~~~~~~~~~")
    for name, snippet in kwic(texts, "beg"):
        print(f"[{name}] ...{snippet}...")

    print("~~~~~~~~~~TERM: SOLD~~~~~~~~~~")
    for name, snippet in kwic(texts, "sold"):
        print(f"[{name}] ...{snippet}...")

    print("~~~~~~~~~~TERM: BORN~~~~~~~~~~")
    for name, snippet in kwic(texts, "born"):
        print(f"[{name}] ...{snippet}...")

import csv


def export_lexicon_scores(scores: dict[str, dict[str, int]], output_path: str = "lexicon_scores.csv") -> None:
    """Export per-document lexicon scores to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        categories = list(LEXICON.keys())
        writer.writerow(["filename"] + categories + ["dominant_category", "total_hits"])

        for filename, doc_scores in scores.items():
            total = sum(doc_scores.values())
            if total == 0:
                dominant = "none"
            else:
                dominant = max(doc_scores, key=doc_scores.get)
            row = [filename] + [doc_scores.get(cat, 0) for cat in categories] + [dominant, total]
            writer.writerow(row)

    print(f"Exported lexicon scores to {output_path}")


def export_kwic_results(texts: dict[str, str], terms: list[str], window: int = 8,
                        output_path: str = "kwic_results.csv") -> None:
    """Export KWIC results for a list of terms to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "filename", "snippet"])

        for term in terms:
            results = kwic(texts, term, window)
            for filename, snippet in results:
                writer.writerow([term, filename, snippet])

    print(f"Exported KWIC results for {len(terms)} terms to {output_path}")


# Inside your main block, add these lines after the print_report and KWIC loops:

if __name__ == "__main__":
    # ... (existing code: load texts, print report, KWIC loops) ...

    # EXPORT TO CSV
    scores = score_all_documents(texts)
    export_lexicon_scores(scores)

    # KWIC export for the terms you've been searching
    target_terms = ["regret", "slave", "property", "wages", "death", "beg", "sold", "born"]
    export_kwic_results(texts, target_terms)