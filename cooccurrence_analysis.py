"""
cooccurrence_analysis.py

Tests co-occurrence of MOURNING language and DEFERENCE language
within the same document, across the whole corpus -- not restricted
to response_status cases. The question: when a document uses grief
vocabulary (regret, sorrow, condolence...), does it ALSO reassert
loyalty/subordination (friendship, faithful, obedient...) in the same
breath -- "we are deeply saddened by this loss, and in this moment we
reaffirm our great friendship"? Or is grief language, where it
appears, kept separate from loyalty language?

This runs on every document with >=1 mourning hit, regardless of
whether that document later generated a reply -- so an original death
ANNOUNCEMENT about a Gulf ruler gets checked, not just a reply to one.
Much larger, more representative sample than the response-status
approach (2-3 documents), which this replaces as the primary version
of the test.

If a corpus metadata TSV is found (same convention as
network_analysis.py / register_by_person.py), each row is also
tagged with its Subject, so you can see the pattern grouped by tier
afterward in a spreadsheet -- but the script runs fine without it too
(subject column just stays blank).

Run from the same folder as your other analysis scripts.
"""

import re
import csv
from pathlib import Path

TRANSCRIPTIONS_DIR = Path("./transcriptions")
INPUT_TSV = Path("./spreadsheet.tsv")  # optional
OUTPUT_CSV = Path("./cooccurrence_results.csv")

MOURNING_TERMS = [
    "regret", "sorrow", "condolence", "condolences", "mourning",
    "grief", "lamented", "melancholy", "bereavement", "sympathy",
    "shocked", "tragic", "profound", "sad", "sadness",
]

DEFERENCE_TERMS = [
    "friendship", "friend", "faithful", "faithfully", "loyal", "loyalty",
    "obedient", "obedience", "devoted", "devotion", "gracious",
    "humble", "humbly", "indebted", "gratitude", "grateful",
    "allegiance", "esteemed", "sincere attachment", "attachment",
    "honour to be", "dutiful",
]

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


def count_terms(text: str, term_list: list) -> dict:
    lower = text.lower()
    hits = {}
    for term in term_list:
        count = len(re.findall(rf"\b{re.escape(term)}\b", lower))
        if count:
            hits[term] = count
    return hits


def filename_to_ref_folio(stem: str):
    parts = stem.split("_")
    if len(parts) < 6:
        return None, None
    return "/".join(parts[:5]), parts[-1]


def load_metadata_if_available(path: Path):
    if not path.exists():
        return {}
    lookup = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ref = row.get("IOR Ref", "").strip()
            folio = row.get("Folio", "").strip()
            if ref and folio:
                lookup[(ref, folio)] = row.get("Subject", "").split(";")[0].strip()
    return lookup


if __name__ == "__main__":
    if not TRANSCRIPTIONS_DIR.exists():
        raise SystemExit(f"Could not find {TRANSCRIPTIONS_DIR}.")

    texts = load_transcripts(TRANSCRIPTIONS_DIR)
    subject_lookup = load_metadata_if_available(INPUT_TSV)
    if subject_lookup:
        print(f"Loaded subject metadata for {len(subject_lookup)} folios from {INPUT_TSV}")
    else:
        print(f"No metadata TSV found at {INPUT_TSV} -- subject column will be blank "
              f"(not required, just optional enrichment)")
    print(f"Loaded {len(texts)} transcripts\n")

    rows = []
    for name, text in texts.items():
        mourning_hits = count_terms(text, MOURNING_TERMS)
        deference_hits = count_terms(text, DEFERENCE_TERMS)
        m_total = sum(mourning_hits.values())
        d_total = sum(deference_hits.values())

        if m_total == 0 and d_total == 0:
            continue  # skip documents with neither -- not relevant to this test

        if m_total > 0 and d_total > 0:
            classification = "mourning + deference"
        elif m_total > 0:
            classification = "mourning only"
        else:
            classification = "deference only"

        ref, folio = filename_to_ref_folio(name)
        subject = subject_lookup.get((ref, folio), "")

        rows.append({
            "document": name,
            "subject": subject,
            "mourning_hits": m_total,
            "deference_hits": d_total,
            "classification": classification,
            "mourning_terms": "; ".join(f"{t}:{c}" for t, c in mourning_hits.items()),
            "deference_terms": "; ".join(f"{t}:{c}" for t, c in deference_hits.items()),
        })

    rows.sort(key=lambda r: (-r["mourning_hits"], -r["deference_hits"]))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "document", "subject", "mourning_hits", "deference_hits",
            "classification", "mourning_terms", "deference_terms",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 70)
    print("MOURNING x DEFERENCE CO-OCCURRENCE")
    print("=" * 70)

    mourning_docs = [r for r in rows if r["mourning_hits"] > 0]
    both = [r for r in mourning_docs if r["deference_hits"] > 0]
    mourning_only = [r for r in mourning_docs if r["deference_hits"] == 0]

    print(f"\nDocuments with mourning language: {len(mourning_docs)}")
    print(f"  ...of which ALSO contain deference language: {len(both)} "
          f"({len(both)/len(mourning_docs)*100:.0f}%)" if mourning_docs else "")
    print(f"  ...mourning language with NO deference language: {len(mourning_only)}")

    print(f"\nDocuments with mourning + deference co-occurring:")
    for r in both:
        print(f"  {r['document']:<40} subject: {r['subject']:<30} "
              f"mourning:{r['mourning_hits']} deference:{r['deference_hits']}")

    print(f"\nDocuments with mourning ONLY (no deference language):")
    for r in mourning_only:
        print(f"  {r['document']:<40} subject: {r['subject']:<30} mourning:{r['mourning_hits']}")

    print(f"\nExported {len(rows)} rows to {OUTPUT_CSV}")