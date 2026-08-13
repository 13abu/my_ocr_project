"""
register_by_person.py

Joins the outputs of register_analysis.py (formula_results.csv,
intensifier_results.csv) against your corpus metadata sheet, so the
register-shift measures can be viewed by AUTHOR, SUBJECT, and
RECIPIENT -- not just by document.

This is the direct test of RQ2: does the same officer write
formulaically about some subjects and singularly about others,
depending on the addressee's tier? If an author's own scores vary by
who the document is about, that's much stronger evidence for
addressee-driven register shift than corpus-wide genre clustering is
-- it isolates the author as the constant and the subject as the
variable.

Prerequisites (run these first, in this order):
  1. lexicon_analysis.py       (not required here, but the shared parser)
  2. register_analysis.py      -> produces formula_results.csv,
                                   intensifier_results.csv
  3. THIS SCRIPT

Also needs your corpus metadata exported as a tab-separated file
(same INPUT_TSV convention as network_analysis.py) with at least:
IOR Ref, Folio, Author/Officer, Recipient, Subject columns.

Run from your project folder, next to formula_results.csv,
intensifier_results.csv, and your metadata TSV.
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

INPUT_TSV = Path("./spreadsheet.tsv")
FORMULA_CSV = Path("./formula_results.csv")
INTENSIFIER_CSV = Path("./intensifier_results.csv")

BY_AUTHOR_CSV = Path("./register_by_author.csv")
BY_SUBJECT_CSV = Path("./register_by_subject.csv")
BY_RECIPIENT_CSV = Path("./register_by_recipient.csv")
AUTHOR_CONSISTENCY_CSV = Path("./author_consistency.csv")


# ==========================================
# Map a transcript filename stem to (IOR Ref, Folio), so it can be
# joined against the metadata sheet. Filenames look like:
#   IOR_R_15_1_208_0029_7r  ->  IOR Ref "IOR/R/15/1/208", Folio "7r"
# The pattern is consistently: first 5 underscore-separated tokens
# form the ref, the middle token(s) are an image index (ignored), the
# LAST token is the folio.
# ==========================================

def filename_to_ref_folio(stem: str):
    parts = stem.split("_")
    if len(parts) < 6:
        return None, None
    ref = "/".join(parts[:5])
    folio = parts[-1]
    return ref, folio


def load_metadata(path: Path):
    """Returns {(ior_ref, folio): {'author':..., 'recipient':..., 'subject':...}}"""
    lookup = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ref = row.get("IOR Ref", "").strip()
            folio = row.get("Folio", "").strip()
            if not ref or not folio:
                continue
            lookup[(ref, folio)] = {
                "author": row.get("Author/Officer", "").strip(),
                "recipient": row.get("Recipient", "").strip(),
                "subject": row.get("Subject", "").split(";")[0].strip(),
            }
    return lookup


def load_formula_results(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["max_similarity"] = float(row["max_similarity"])
            rows.append(row)
    return rows


def load_intensifier_results(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["intensified"] = row["intensified"].strip().lower() == "true"
            rows.append(row)
    return rows


def enrich_with_metadata(rows, metadata):
    """Adds author/recipient/subject fields to each row (formula or
    intensifier result), matched via the document filename. Rows with
    no metadata match get empty strings for all three -- printed as a
    warning count at the end rather than silently dropped, so unmapped
    documents are visible rather than hidden."""
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


# ==========================================
# Aggregation
# ==========================================

def aggregate_formula(rows, key_field):
    """Per key (author/subject/recipient): count, avg similarity,
    n_formula, n_singular."""
    groups = defaultdict(list)
    for r in rows:
        key = r[key_field]
        if key:
            groups[key].append(r["max_similarity"])

    out = []
    for key, sims in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n_formula = sum(1 for s in sims if s >= 0.75)
        out.append({
            key_field: key,
            "n_sentences": len(sims),
            "avg_max_similarity": round(sum(sims) / len(sims), 3),
            "n_formula": n_formula,
            "n_singular": len(sims) - n_formula,
        })
    return out


def aggregate_intensifier(rows, key_field):
    groups = defaultdict(lambda: {"total": 0, "intensified": 0})
    for r in rows:
        key = r[key_field]
        if key:
            groups[key]["total"] += 1
            if r["intensified"]:
                groups[key]["intensified"] += 1

    out = []
    for key, counts in sorted(groups.items(), key=lambda kv: -kv[1]["total"]):
        ratio = counts["intensified"] / counts["total"] if counts["total"] else 0
        out.append({
            key_field: key,
            "n_affect_hits": counts["total"],
            "n_intensified": counts["intensified"],
            "intensified_ratio": round(ratio, 3),
        })
    return out


def export_csv(rows, fieldnames, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_author_consistency(formula_rows):
    """For each author who appears with more than one SUBJECT, list
    their per-subject average similarity side by side -- this is the
    direct RQ2 view: does this specific person's language vary
    depending on who they're writing about?"""
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
            continue  # only interesting if this author covers 2+ subjects
        values = list(subject_scores.values())
        spread = round(max(values) - min(values), 3)
        for subject, avg_sim in sorted(subject_scores.items(), key=lambda kv: kv[1]):
            out.append({
                "author": author,
                "n_subjects_covered": len(subject_scores),
                "spread_across_subjects": spread,
                "subject": subject,
                "avg_max_similarity_for_this_subject": avg_sim,
            })
    return out


if __name__ == "__main__":
    for required in (INPUT_TSV, FORMULA_CSV, INTENSIFIER_CSV):
        if not required.exists():
            raise SystemExit(
                f"Could not find {required}. Run register_analysis.py first, "
                f"and make sure your metadata TSV is exported to {INPUT_TSV}."
            )

    metadata = load_metadata(INPUT_TSV)
    print(f"Loaded metadata for {len(metadata)} folios from {INPUT_TSV}\n")

    formula_rows = load_formula_results(FORMULA_CSV)
    intensifier_rows = load_intensifier_results(INTENSIFIER_CSV)

    unmatched_f = enrich_with_metadata(formula_rows, metadata)
    unmatched_i = enrich_with_metadata(intensifier_rows, metadata)
    all_unmatched = unmatched_f | unmatched_i
    if all_unmatched:
        print(f"WARNING: {len(all_unmatched)} document(s) had no metadata match "
              f"(check filename parsing / IOR Ref+Folio spelling in your TSV):")
        for name in sorted(all_unmatched):
            print(f"  - {name}")
        print()

    # By author
    author_formula = aggregate_formula(formula_rows, "author")
    export_csv(author_formula,
               ["author", "n_sentences", "avg_max_similarity", "n_formula", "n_singular"],
               BY_AUTHOR_CSV)
    print(f"Wrote {len(author_formula)} rows to {BY_AUTHOR_CSV}")

    # By subject
    subject_formula = aggregate_formula(formula_rows, "subject")
    export_csv(subject_formula,
               ["subject", "n_sentences", "avg_max_similarity", "n_formula", "n_singular"],
               BY_SUBJECT_CSV)
    print(f"Wrote {len(subject_formula)} rows to {BY_SUBJECT_CSV}")

    # By recipient
    recipient_formula = aggregate_formula(formula_rows, "recipient")
    export_csv(recipient_formula,
               ["recipient", "n_sentences", "avg_max_similarity", "n_formula", "n_singular"],
               BY_RECIPIENT_CSV)
    print(f"Wrote {len(recipient_formula)} rows to {BY_RECIPIENT_CSV}")

    # Author consistency -- the actual RQ2 test
    consistency = build_author_consistency(formula_rows)
    export_csv(consistency,
               ["author", "n_subjects_covered", "spread_across_subjects",
                "subject", "avg_max_similarity_for_this_subject"],
               AUTHOR_CONSISTENCY_CSV)
    print(f"Wrote {len(consistency)} rows to {AUTHOR_CONSISTENCY_CSV}")

    print()
    print("=" * 70)
    print("AUTHORS COVERING MULTIPLE SUBJECTS (the direct RQ2 test)")
    print("=" * 70)
    seen_authors = set()
    for row in sorted(consistency, key=lambda r: -r["spread_across_subjects"]):
        if row["author"] not in seen_authors:
            seen_authors.add(row["author"])
            print(f"\n  {row['author']}  (spread: {row['spread_across_subjects']})")
        print(f"    {row['subject']:<35} avg similarity: {row['avg_max_similarity_for_this_subject']}")