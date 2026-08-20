"""
affective_register_analysis.py

Consolidated computational tool for IRP Instrument 1: the affective
lexicon mapping administrative vocabulary that permits or forecloses
grief -- plus everything built on top of it (intensity, formula vs
singularity, author/subject/recipient register-shift, mourning x
deference co-occurrence).

This merges four previously separate scripts (lexicon_analysis.py,
register_analysis.py, register_by_person.py, cooccurrence_analysis.py)
into one, since they all read the same transcripts and differ only in
what they count. deference_analysis.py is NOT included -- its
response-status-only test (2-3 documents) is superseded by the
corpus-wide co-occurrence test here (Part 4), per that script's own
docstring.

Run from your project folder, next to transcriptions/ and (for Part
3's author/subject/recipient breakdowns) your metadata spreadsheet.tsv.
Parts 1, 2, and 4 run fine without the TSV; Part 3 is skipped with a
clear message if it's absent.

Outputs (all written to the current folder):
  lexicon_scores.csv        -- per-document category scores (Part 1)
  kwic_results.csv          -- keyword-in-context for 8 anchor terms (Part 1)
  intensifier_results.csv   -- per-hit intensifier check (Part 2)
  formula_results.csv       -- per-sentence formula/singularity score (Part 2)
  register_by_author.csv    -- formula stats grouped by author (Part 3)
  register_by_subject.csv   -- formula stats grouped by subject (Part 3)
  register_by_recipient.csv -- formula stats grouped by recipient (Part 3)
  author_consistency.csv    -- the direct RQ2 test: same author, different subjects (Part 3)
  cooccurrence_results.csv  -- mourning x deference co-occurrence (Part 4)
"""

import re
import csv
import difflib
from pathlib import Path
from collections import Counter, defaultdict

TRANSCRIPTIONS_DIR = Path("./transcriptions")
INPUT_TSV = Path("./spreadsheet.tsv")

LEXICON_SCORES_CSV = Path("./lexicon_scores.csv")
KWIC_CSV = Path("./kwic_results.csv")
INTENSIFIER_CSV = Path("./intensifier_results.csv")
FORMULA_CSV = Path("./formula_results.csv")
BY_AUTHOR_CSV = Path("./register_by_author.csv")
BY_SUBJECT_CSV = Path("./register_by_subject.csv")
BY_RECIPIENT_CSV = Path("./register_by_recipient.csv")
AUTHOR_CONSISTENCY_CSV = Path("./author_consistency.csv")
COOCCURRENCE_CSV = Path("./cooccurrence_results.csv")


# ==========================================
# SHARED: transcript loading (identical logic across all four original
# scripts, kept as a single copy here)
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
    """Returns (texts, unmarked, method_counts)."""
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


def filename_to_ref_folio(stem: str):
    """IOR_R_15_1_208_0029_7r -> ('IOR/R/15/1/208', '7r')"""
    parts = stem.split("_")
    if len(parts) < 6:
        return None, None
    return "/".join(parts[:5]), parts[-1]


def load_metadata_if_available(path: Path):
    """Returns {} if the TSV doesn't exist -- callers handle the empty
    case gracefully rather than requiring it (Parts 1/2/4 don't need it)."""
    if not path.exists():
        return {}
    lookup = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ref = row.get("IOR Ref", "").strip()
            folio = row.get("Folio", "").strip()
            if ref and folio:
                lookup[(ref, folio)] = {
                    "author": row.get("Author/Officer", "").strip(),
                    "recipient": row.get("Recipient", "").strip(),
                    "subject": row.get("Subject", "").split(";")[0].strip(),
                }
    return lookup


STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "is", "was", "were",
    "be", "on", "for", "with", "as", "at", "by", "that", "this",
    "it", "his", "her", "he", "she", "i", "you", "your", "from",
    "or", "not", "has", "have", "had", "will", "would", "which",
}


# ==========================================
# PART 1: BASE LEXICON (was lexicon_analysis.py)
# ==========================================

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

KWIC_TARGET_TERMS = ["regret", "slave", "property", "wages", "death", "beg", "sold", "born"]


def word_frequency(texts: dict, top_n: int = 50):
    counter = Counter()
    for text in texts.values():
        words = re.findall(r"[a-zA-Z']+", text.lower())
        counter.update(w for w in words if w not in STOPWORDS and len(w) > 2)
    return counter.most_common(top_n)


def kwic(texts: dict, term: str, window: int = 8):
    results = []
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


def score_document_lexicon(text: str):
    lower = text.lower()
    scores = {}
    for category, terms in LEXICON.items():
        count = 0
        for term in terms:
            count += len(re.findall(rf"\b{re.escape(term)}\b", lower))
        scores[category] = count
    return scores


def run_lexicon(texts: dict):
    print("=" * 70)
    print("PART 1: BASE LEXICON")
    print("=" * 70)

    print("\nTop 20 words across the corpus:")
    for word, count in word_frequency(texts, top_n=20):
        print(f"  {word:<20} {count}")

    scores = {name: score_document_lexicon(text) for name, text in texts.items()}
    with open(LEXICON_SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        categories = list(LEXICON.keys())
        writer.writerow(["filename"] + categories + ["dominant_category", "total_hits"])
        for filename, doc_scores in scores.items():
            total = sum(doc_scores.values())
            dominant = "none" if total == 0 else max(doc_scores, key=doc_scores.get)
            writer.writerow([filename] + [doc_scores.get(c, 0) for c in categories] + [dominant, total])
    print(f"\nWrote {len(scores)} rows to {LEXICON_SCORES_CSV}")

    with open(KWIC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "filename", "snippet"])
        total_hits = 0
        for term in KWIC_TARGET_TERMS:
            for filename, snippet in kwic(texts, term):
                writer.writerow([term, filename, snippet])
                total_hits += 1
    print(f"Wrote {total_hits} rows to {KWIC_CSV}")

    return scores


# ==========================================
# PART 2: INTENSITY + FORMULA/SINGULARITY (was register_analysis.py)
# ==========================================

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

# FIX (validated against the my_ocr_project repo -- see ma-thesis-repo
# notes): was preceding-only ("profound regret", "deeply regret").
# Widened to check BOTH sides after f.1733/58r's "we regret deeply" --
# a postposed intensifier -- scored as NOT intensified under the old
# preceding-only check. English allows the intensifier on either side
# of the affect word, so the check needs to too. Confirmed this is the
# ONLY row that changes across the full 110-doc corpus -- a precise,
# not a noisy, fix.
WINDOW = 3

TARGET_TERMS = ["regret", "death", "died", "deceased", "condolence", "wages", "manumit"]
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?;])\s+")


def find_intensified_hits(texts: dict):
    results = []
    for name, text in texts.items():
        words = text.split()
        lower_words = [w.lower().strip(".,;:") for w in words]
        for i, w in enumerate(lower_words):
            if w in AFFECT_TERMS:
                start = max(0, i - WINDOW)
                end = min(len(lower_words), i + WINDOW + 1)
                surrounding = lower_words[start:i] + lower_words[i + 1:end]
                intensifier_found = next((t for t in INTENSIFIERS if t in surrounding), None)
                snip_start = max(0, i - WINDOW)
                snip_end = min(len(words), i + WINDOW + 1)
                snippet = " ".join(words[snip_start:snip_end])
                results.append({
                    "document": name, "term": w,
                    "intensified": bool(intensifier_found),
                    "intensifier": intensifier_found or "",
                    "snippet": snippet,
                })
    return results


def extract_relevant_sentences(texts: dict):
    sentences = []
    for name, text in texts.items():
        clean = re.sub(r"\s+", " ", text)
        for sent in SENTENCE_SPLIT_PATTERN.split(clean):
            sent_lower = sent.lower()
            if any(term in sent_lower for term in TARGET_TERMS):
                sent_stripped = sent.strip()
                if len(sent_stripped) > 15:
                    sentences.append({"document": name, "sentence": sent_stripped})
    return sentences


def score_formula_vs_singular(sentences, similarity_threshold: float = 0.75):
    n = len(sentences)
    for i in range(n):
        best_score, best_doc, best_sentence = 0.0, None, None
        for j in range(n):
            if i == j or sentences[i]["document"] == sentences[j]["document"]:
                continue
            ratio = difflib.SequenceMatcher(
                None, sentences[i]["sentence"].lower(), sentences[j]["sentence"].lower()
            ).ratio()
            if ratio > best_score:
                best_score, best_doc, best_sentence = ratio, sentences[j]["document"], sentences[j]["sentence"]
        sentences[i]["max_similarity"] = round(best_score, 3)
        sentences[i]["most_similar_document"] = best_doc or ""
        sentences[i]["most_similar_sentence"] = best_sentence or ""
        sentences[i]["classification"] = "formula/template" if best_score >= similarity_threshold else "singular/unique"
    return sentences


def run_register(texts: dict):
    print()
    print("=" * 70)
    print("PART 2: INTENSITY + FORMULA/SINGULARITY")
    print("=" * 70)

    intensity_results = find_intensified_hits(texts)
    with open(INTENSIFIER_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "term", "intensified", "intensifier", "snippet"])
        for r in intensity_results:
            writer.writerow([r["document"], r["term"], r["intensified"], r["intensifier"], r["snippet"]])
    n_intensified = sum(1 for r in intensity_results if r["intensified"])
    print(f"\nIntensity: {n_intensified}/{len(intensity_results)} affect-term hits intensified")
    print(f"Wrote {len(intensity_results)} rows to {INTENSIFIER_CSV}")

    sentences = extract_relevant_sentences(texts)
    sentences = score_formula_vs_singular(sentences)
    with open(FORMULA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "sentence", "max_similarity", "classification",
                          "most_similar_document", "most_similar_sentence"])
        for s in sentences:
            writer.writerow([s["document"], s["sentence"], s["max_similarity"], s["classification"],
                              s["most_similar_document"], s["most_similar_sentence"]])
    n_formula = sum(1 for s in sentences if s["classification"] == "formula/template")
    print(f"\nFormula/singularity: {n_formula}/{len(sentences)} sentences classified as formula/template")
    print(f"Wrote {len(sentences)} rows to {FORMULA_CSV}")

    return intensity_results, sentences


# ==========================================
# PART 3: AUTHOR/SUBJECT/RECIPIENT AGGREGATION (was register_by_person.py)
# Needs spreadsheet.tsv -- skipped with a clear message if absent.
# ==========================================

def enrich_with_metadata(rows, metadata):
    unmatched = set()
    for row in rows:
        ref, folio = filename_to_ref_folio(row["document"])
        meta = metadata.get((ref, folio), {})
        if not meta:
            unmatched.add(row["document"])
        row["author"] = meta.get("author", "")
        row["recipient"] = meta.get("recipient", "")
        row["subject"] = meta.get("subject", "")
    return unmatched


def aggregate_formula(rows, key_field):
    groups = defaultdict(list)
    for r in rows:
        key = r[key_field]
        if key:
            groups[key].append(r["max_similarity"])
    out = []
    for key, sims in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n_formula = sum(1 for s in sims if s >= 0.75)
        out.append({
            key_field: key, "n_sentences": len(sims),
            "avg_max_similarity": round(sum(sims) / len(sims), 3),
            "n_formula": n_formula, "n_singular": len(sims) - n_formula,
        })
    return out


def build_author_consistency(formula_rows):
    by_author_subject = defaultdict(list)
    for r in formula_rows:
        if r["author"] and r["subject"]:
            by_author_subject[(r["author"], r["subject"])].append(r["max_similarity"])
    by_author = defaultdict(dict)
    for (author, subject), sims in by_author_subject.items():
        by_author[author][subject] = round(sum(sims) / len(sims), 3)
    out = []
    for author, subject_scores in by_author.items():
        if len(subject_scores) < 2:
            continue
        values = list(subject_scores.values())
        spread = round(max(values) - min(values), 3)
        for subject, avg_sim in sorted(subject_scores.items(), key=lambda kv: kv[1]):
            out.append({
                "author": author, "n_subjects_covered": len(subject_scores),
                "spread_across_subjects": spread, "subject": subject,
                "avg_max_similarity_for_this_subject": avg_sim,
            })
    return out


def export_csv(rows, fieldnames, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_by_person(formula_sentences, metadata):
    print()
    print("=" * 70)
    print("PART 3: AUTHOR / SUBJECT / RECIPIENT AGGREGATION")
    print("=" * 70)

    if not metadata:
        print(f"\nSKIPPED -- no {INPUT_TSV} found. This part needs your corpus")
        print("metadata sheet exported as a TSV with IOR Ref/Folio/Author/")
        print("Recipient/Subject columns. Everything else above still ran.")
        return

    unmatched = enrich_with_metadata(formula_sentences, metadata)
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} document(s) had no metadata match:")
        for name in sorted(unmatched):
            print(f"  - {name}")

    for key_field, path in (
        ("author", BY_AUTHOR_CSV), ("subject", BY_SUBJECT_CSV), ("recipient", BY_RECIPIENT_CSV)
    ):
        agg = aggregate_formula(formula_sentences, key_field)
        export_csv(agg, [key_field, "n_sentences", "avg_max_similarity", "n_formula", "n_singular"], path)
        print(f"\nWrote {len(agg)} rows to {path}")

    consistency = build_author_consistency(formula_sentences)
    export_csv(consistency,
               ["author", "n_subjects_covered", "spread_across_subjects",
                "subject", "avg_max_similarity_for_this_subject"],
               AUTHOR_CONSISTENCY_CSV)
    print(f"\nWrote {len(consistency)} rows to {AUTHOR_CONSISTENCY_CSV}")

    print("\nAuthors covering multiple subjects (the direct RQ2 test):")
    seen = set()
    for row in sorted(consistency, key=lambda r: -r["spread_across_subjects"]):
        if row["author"] not in seen:
            seen.add(row["author"])
            print(f"\n  {row['author']}  (spread: {row['spread_across_subjects']})")
        print(f"    {row['subject']:<35} avg similarity: {row['avg_max_similarity_for_this_subject']}")


# ==========================================
# PART 4: MOURNING x DEFERENCE CO-OCCURRENCE (was cooccurrence_analysis.py)
# Supersedes deference_analysis.py's response-status-only test (n=2-3)
# with a corpus-wide version -- deference_analysis.py is retired, not
# included here.
# ==========================================

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


def count_terms(text: str, term_list: list):
    lower = text.lower()
    hits = {}
    for term in term_list:
        count = len(re.findall(rf"\b{re.escape(term)}\b", lower))
        if count:
            hits[term] = count
    return hits


def run_cooccurrence(texts: dict, metadata: dict):
    print()
    print("=" * 70)
    print("PART 4: MOURNING x DEFERENCE CO-OCCURRENCE")
    print("=" * 70)

    subject_lookup = {ref_folio: meta["subject"] for ref_folio, meta in metadata.items()} if metadata else {}

    rows = []
    for name, text in texts.items():
        mourning_hits = count_terms(text, MOURNING_TERMS)
        deference_hits = count_terms(text, DEFERENCE_TERMS)
        m_total, d_total = sum(mourning_hits.values()), sum(deference_hits.values())
        if m_total == 0 and d_total == 0:
            continue
        classification = (
            "mourning + deference" if m_total > 0 and d_total > 0
            else "mourning only" if m_total > 0
            else "deference only"
        )
        ref, folio = filename_to_ref_folio(name)
        rows.append({
            "document": name, "subject": subject_lookup.get((ref, folio), ""),
            "mourning_hits": m_total, "deference_hits": d_total,
            "classification": classification,
            "mourning_terms": "; ".join(f"{t}:{c}" for t, c in mourning_hits.items()),
            "deference_terms": "; ".join(f"{t}:{c}" for t, c in deference_hits.items()),
        })
    rows.sort(key=lambda r: (-r["mourning_hits"], -r["deference_hits"]))

    with open(COOCCURRENCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "document", "subject", "mourning_hits", "deference_hits",
            "classification", "mourning_terms", "deference_terms",
        ])
        writer.writeheader()
        writer.writerows(rows)

    mourning_docs = [r for r in rows if r["mourning_hits"] > 0]
    both = [r for r in mourning_docs if r["deference_hits"] > 0]
    print(f"\nDocuments with mourning language: {len(mourning_docs)}")
    if mourning_docs:
        print(f"  ...also containing deference language: {len(both)} "
              f"({len(both) / len(mourning_docs) * 100:.0f}%)")
    print(f"\nWrote {len(rows)} rows to {COOCCURRENCE_CSV}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    if not TRANSCRIPTIONS_DIR.exists():
        raise SystemExit(f"Could not find {TRANSCRIPTIONS_DIR}. Run this from your project folder.")

    texts, unmarked, method_counts = load_transcripts(TRANSCRIPTIONS_DIR)
    print(f"Loaded {len(texts)} transcripts from {TRANSCRIPTIONS_DIR}")
    if unmarked:
        print(f"WARNING: {len(unmarked)} file(s) excluded (ambiguous/empty): {', '.join(unmarked)}")
    print()

    metadata = load_metadata_if_available(INPUT_TSV)

    run_lexicon(texts)
    intensity_results, formula_sentences = run_register(texts)
    run_by_person(formula_sentences, metadata)
    run_cooccurrence(texts, metadata)

    print()
    print("=" * 70)
    print("DONE. Outputs written to the current folder:")
    print("=" * 70)
    for path in (LEXICON_SCORES_CSV, KWIC_CSV, INTENSIFIER_CSV, FORMULA_CSV,
                 BY_AUTHOR_CSV, BY_SUBJECT_CSV, BY_RECIPIENT_CSV,
                 AUTHOR_CONSISTENCY_CSV, COOCCURRENCE_CSV):
        print(f"  {path}")