"""
exhibition_comparison.py

Compares the language of exhibition.txt (curatorial text from the
Bin Jelmood House / slavery museum in Doha) against the full corpus of
IOR/R/15 transcriptions -- how does a museum's own narration of this
history compare to the colonial administration's own vocabulary for it?

Deliberately reuses the same 4-category lexicon and stopword list
already built for lexicon_analysis.py, so exhibition.txt's category
score is directly comparable to every transcript's score, not a new
one-off measure.

Three angles:
  1. Which of the 4 lexicon categories (mourning / administrative-
     neutral / trafficking-enforcement / monetary-valuation) does
     exhibition.txt lean toward, and how does that compare to the
     corpus average?
  2. Vocabulary overlap (Jaccard similarity) -- what fraction of
     distinct words are shared between the two, vs unique to each?
  3. Top words unique to exhibition.txt (not just rare in the corpus,
     genuinely its own vocabulary) and top words heavily used in the
     corpus but nearly absent from exhibition.txt -- the actual
     content of the divergence, not just a similarity number.

Run from your project folder, next to exhibition.txt and
transcriptions/.
"""

import re
from pathlib import Path
from collections import Counter

EXHIBITION_TXT = Path("./exhibition.txt")
TRANSCRIPTIONS_DIR = Path("./transcriptions")

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "is", "was", "were",
    "be", "on", "for", "with", "as", "at", "by", "that", "this",
    "it", "his", "her", "he", "she", "i", "you", "your", "from",
    "or", "not", "has", "have", "had", "will", "would", "which",
}

LEXICON = {
    "mourning": [
        "regret", "sorrow", "condolence", "condolences", "mourning",
        "grief", "lamented", "melancholy", "bereavement", "sympathy",
        "shocked", "tragic", "profound",
    ],
    "administrative_neutral": [
        "claim", "settlement", "instructed", "acknowledge", "forwarded",
        "requested", "report", "action", "case", "reference",
    ],
    "trafficking_enforcement": [
        "slave", "slaves", "proclamation", "trade", "traffic",
        "prohibiting", "forbid", "forbidden", "suppress",
        "manumission", "manumitted", "kidnapped", "diving", "certificate",
        "re-enslave", "re-enslavement", "fear", "beaten",
    ],
    "monetary_valuation": [
        "rupees", "rs", "price", "advance", "debt", "wages", "compensation",
        "settlement", "paid", "unpaid", "owed", "sold", "purchased", "purchase",
    ],
}

# Same body-extraction logic as the other scripts, for identical
# transcript loading behavior.
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
        return raw[body_match.end():].strip()
    marg_match = MARGINALIA_HEADER_PATTERN.search(raw)
    if marg_match:
        blank_match = BLANK_LINE_PATTERN.search(raw, marg_match.end())
        if blank_match:
            remainder = raw[blank_match.end():].strip()
            if len(remainder) >= MIN_BODY_AFTER_MARGINALIA:
                return remainder
        text_before = raw[:marg_match.start()].strip()
        if len(text_before) >= MIN_BODY_BEFORE_MARGINALIA:
            return text_before
        return None
    return raw.strip() if raw.strip() else None


def load_corpus_text(directory: Path) -> str:
    chunks = []
    for path in sorted(directory.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        body = extract_body_text(raw)
        if body:
            chunks.append(body)
    return "\n".join(chunks)


def tokenize(text: str) -> list:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def score_lexicon(text: str) -> dict:
    lower = text.lower()
    scores = {}
    for category, terms in LEXICON.items():
        count = 0
        for term in terms:
            count += len(re.findall(rf"\b{re.escape(term)}\b", lower))
        scores[category] = count
    return scores


def main():
    if not EXHIBITION_TXT.exists():
        raise SystemExit(f"Could not find {EXHIBITION_TXT}.")
    if not TRANSCRIPTIONS_DIR.exists():
        raise SystemExit(f"Could not find {TRANSCRIPTIONS_DIR}.")

    exhibition_text = EXHIBITION_TXT.read_text(encoding="utf-8", errors="ignore")
    corpus_text = load_corpus_text(TRANSCRIPTIONS_DIR)

    print(f"exhibition.txt: {len(exhibition_text)} characters")
    print(f"corpus (all transcripts combined): {len(corpus_text)} characters")
    print()

    # --- 1. Lexicon category comparison ---
    exhibition_scores = score_lexicon(exhibition_text)
    corpus_scores = score_lexicon(corpus_text)

    print("=" * 70)
    print("LEXICON CATEGORY COMPARISON")
    print("=" * 70)
    print(f"{'Category':<28}{'exhibition.txt':<18}{'corpus (raw count)':<20}")
    for cat in LEXICON:
        print(f"{cat:<28}{exhibition_scores[cat]:<18}{corpus_scores[cat]:<20}")
    exh_total = sum(exhibition_scores.values())
    exh_dominant = "none" if exh_total == 0 else max(exhibition_scores, key=exhibition_scores.get)
    print(f"\nexhibition.txt dominant category: {exh_dominant}")

    # --- 2. Vocabulary overlap (Jaccard similarity) ---
    exhibition_words = set(tokenize(exhibition_text))
    corpus_words = set(tokenize(corpus_text))
    shared = exhibition_words & corpus_words
    union = exhibition_words | corpus_words
    jaccard = len(shared) / len(union) if union else 0

    print()
    print("=" * 70)
    print("VOCABULARY OVERLAP")
    print("=" * 70)
    print(f"Distinct words in exhibition.txt: {len(exhibition_words)}")
    print(f"Distinct words in corpus: {len(corpus_words)}")
    print(f"Shared distinct words: {len(shared)}")
    print(f"Jaccard similarity (shared / union): {jaccard:.3f}")

    # --- 3. Divergent vocabulary, not just a number ---
    exhibition_counts = Counter(tokenize(exhibition_text))
    corpus_counts = Counter(tokenize(corpus_text))

    exhibition_only = [w for w in exhibition_counts if w not in corpus_counts]
    exhibition_only_top = sorted(exhibition_only, key=lambda w: -exhibition_counts[w])[:20]

    print()
    print("=" * 70)
    print("TOP 20 WORDS IN exhibition.txt NOT FOUND IN THE CORPUS AT ALL")
    print("=" * 70)
    for w in exhibition_only_top:
        print(f"  {w:<20} {exhibition_counts[w]}")

    corpus_heavy_not_exhibition = [
        w for w, c in corpus_counts.most_common(200) if w not in exhibition_counts
    ][:20]
    print()
    print("=" * 70)
    print("TOP 20 WORDS HEAVILY USED IN THE CORPUS, ABSENT FROM exhibition.txt")
    print("=" * 70)
    for w in corpus_heavy_not_exhibition:
        print(f"  {w:<20} {corpus_counts[w]}")

    print()
    print("=" * 70)
    print("TOP 20 SHARED WORDS (by combined frequency)")
    print("=" * 70)
    shared_ranked = sorted(shared, key=lambda w: -(exhibition_counts[w] + corpus_counts[w]))[:20]
    for w in shared_ranked:
        print(f"  {w:<20} exhibition:{exhibition_counts[w]:<6} corpus:{corpus_counts[w]}")


if __name__ == "__main__":
    main()