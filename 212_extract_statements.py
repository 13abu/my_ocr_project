"""Build a coding sheet of statements from plain-text transcriptions. No API calls.

Usage:  python extract_statements.py [transcriptions_folder] [output.csv]

Reads every .txt in the folder in page order, joins them (so a statement that runs
over a page break stays in one piece), finds each "Statement made by ..." heading,
and writes one CSV row per statement. Fields that follow a fixed form (name, bin/bint,
age, date, thumb impression, certificate number) are read with patterns. Everything
is copied from the text, nothing is inferred beyond bin/bint -> male/female.

Ground and 'counts as application' are left blank for you to code. The keyword_hints
column only helps you find passages quickly. It is not coding.

Assumes each transcription has its MARGINALIA block after the body text.
"""
import bisect
import csv
import re
import sys
from collections import Counter
from pathlib import Path

INPUT_FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./transcriptions")
OUTPUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("statements.csv")

# A statement starts at the beginning of a line.
HEADING = re.compile(
    r"^[ \t>*_#\"'\-]*(?:Translation of )?(?:the )?statement\s+(?:made\s+)?(?:by|of)\s+(?P<rest>.+)$",
    re.I | re.M,
)
# Other documents that end the previous statement.
CUT = re.compile(
    r"^[ \t]*Translation of (?:a |the )?(?:letter|telegram|memorandum|endorsement|application)\b",
    re.I | re.M,
)
MARGINALIA = re.compile(r"^\W*MARGINALIA\b.*$", re.I | re.M)

FATHER_UNKNOWN = re.compile(
    r"father'?s\s+name\s+(?:is\s+|was\s+)?(?:unknown|not\s+known)"
    r"|(?:do|did)\s*(?:not|n't)\s+know\s+(?:the\s+name\s+of\s+)?(?:my|his|her|the)\s+father"
    r"|unknown\s+father",
    re.I,
)
THUMB = re.compile(
    r"\b(?:L\.?\s?)?T\.?\s?I\.?\s*of\b|\bL\.?\s?T\.?\s?of\b|\bL\.?\s?T\.?\s?I\b"
    r"|left[\s-]+thumb|thumb[\s-]*impression|\bthumb\b", re.I)   # also catches "T.I.of X" and "L.T.of X"
SIGNED = re.compile(r"signature|\bsigned\b|\bseal\b", re.I)
CERT = re.compile(r"certificates?\s+(?:nos?\.?\s*)?(\d+)", re.I)

HINTS = {
    "sold": r"\bsold\b|\bsell\b|\bsale\b",
    "violence": r"\bbeat|\bbeaten|ill[\s-]?treat|cruel|imprison|\btied\b|\bflog",
    "kidnap": r"kidnap|abduct",
    "earnings": r"earning|wages|advance|\bdebt|salary",
    "food_clothing": r"\bfood\b|clothing|clothes",
    "family": r"\bmarried\b|husband|\bwife\b|\bson\b|daughter|mother|sister|brother",
    "age_infirmity": r"old age|old woman|old man|unable to work|infirm",
    "refuge_release": r"refuge|released from slavery|manumission|freedom",
}


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]


def tidy(s: str) -> str:
    return " ".join(s.split())


def split_page(text: str):
    """Return (body, marginalia).

    Same structural rule as the lexicon script. If real text comes before the
    MARGINALIA heading (continuation pages), the block is at the end and the body is
    what precedes it. Otherwise the block is first and the body starts after the
    first blank line that follows the heading, whatever the next line says
    (TRANSCRIPTION:, MAIN TEXT, a bare rule, or nothing).
    """
    m = MARGINALIA.search(text)
    if not m:
        return text, ""
    before = text[:m.start()]
    if len(before.strip()) > 40:
        return before, text[m.start():]
    rest = text[m.end():]
    b = re.search(r"\n[ \t]*\n", rest)
    if not b:
        return "", text[m.start():]
    return rest[b.end():], text[m.start():m.end() + b.start()]


def load_pages(folder: Path):
    pages = []
    for p in sorted(folder.glob("*.txt"), key=natural_key):
        if p.name.startswith("_"):   # e.g. _failures.txt
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if text.lstrip().startswith("[NO TEXT RETURNED"):   # blank-page marker from the transcriber
            pages.append((p.stem, "\n", "", text))
            continue
        main, marg = split_page(text)
        pages.append((p.stem, main.rstrip() + "\n", tidy(marg), text))
    return pages


def parse_head(body: str) -> dict:
    head = tidy(body[:400])
    out = {}

    name_raw = re.split(r",|\baged?\b|\bborn\b| on \d| on the |\bRecorded\b|\.", head, maxsplit=1)[0]
    out["name_raw"] = name_raw.strip()

    if re.search(r"\b(?:bint|binti)\b", name_raw, re.I):
        out["sex_auto"], out["sex_marker"] = "female", "bint"
    elif re.search(r"\b(?:bin|ibn)\b", name_raw, re.I):
        out["sex_auto"], out["sex_marker"] = "male", "bin"
    elif re.search(r"\(\s*his\b", name_raw, re.I):    # e.g. "Mubarak (his father's name unknown)"
        out["sex_auto"], out["sex_marker"] = "male", "(his"
    elif re.search(r"\(\s*her\b", name_raw, re.I):
        out["sex_auto"], out["sex_marker"] = "female", "(her"
    else:
        out["sex_auto"], out["sex_marker"] = "", ""

    m = re.search(r"\bage[d]?\s*(?:of|is)?\s*(?:about\s+)?(\d{1,3})", head, re.I) or \
        re.search(r"\babout\s+(\d{1,3})\s+years", head, re.I)
    out["age_stated"] = m.group(1) if m else ""

    m = re.search(r"\bon\s+(?:the\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[A-Za-z']+\s+\d{4}(?:\s*\([^)]*\))?)", head)
    out["date_raw"] = m.group(1) if m else ""

    m = re.search(r"\bborn\s+(?:at|in)\s+([A-Z][A-Za-z' -]+?)(?=,|\.|\s+aged|\s+and\b)", head)
    out["born_at"] = m.group(1).strip() if m else ""

    m = re.search(r"\bof\s+([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)", name_raw)
    out["origin_term_in_name"] = m.group(1) if m else ""

    return out


def main():
    pages = load_pages(INPUT_FOLDER)
    if not pages:
        print(f"No .txt files found in {INPUT_FOLDER}")
        return

    full, starts = "", []
    for _, main_text, _, _ in pages:
        starts.append(len(full))
        full += main_text

    # Diagnostics, so a zero result says why.
    n_empty = sum(1 for _, body, _, _ in pages if not body.strip())
    n_word = sum(1 for _, _, _, raw in pages if re.search(r"statement", raw, re.I))
    print(f"{len(pages)} page(s) read; {n_empty} with an empty body after removing MARGINALIA; "
          f"{n_word} mention the word 'statement' anywhere.")

    def page_at(offset: int) -> int:
        return bisect.bisect_right(starts, offset) - 1

    boundaries = sorted({m.start() for m in HEADING.finditer(full)} | {m.start() for m in CUT.finditer(full)})

    rows = []
    for n, m in enumerate(HEADING.finditer(full), start=1):
        nxt = [b for b in boundaries if b > m.start()]
        end = nxt[0] if nxt else len(full)
        body = full[m.start("rest"):end]

        span = body    # text up to the next heading, kept for certificate numbers

        # A statement normally ends with its thumb impression ("L.T.I. of ..."). If one is found,
        # end the statement at that line, so the rest of the page is not counted as part of it.
        thumb = THUMB.search(body)
        if thumb:
            nl = body.find("\n", thumb.end())
            cut = len(body) if nl == -1 else nl
            body = body[:cut]
            end = m.start("rest") + cut

        row = parse_head(body)

        if FATHER_UNKNOWN.search(body[:800]):
            row["father_name_status"] = "unknown"
        elif row["sex_marker"] in ("bin", "bint"):
            row["father_name_status"] = "given"
        else:
            row["father_name_status"] = "absent"

        tail = body.rstrip()[-400:]
        row["authentication_auto"] = "thumb" if thumb else ("signature/seal" if SIGNED.search(tail) else "none found")
        row["certificate_numbers_nearby"] = ";".join(dict.fromkeys(CERT.findall(span)))   # may belong to a covering letter or another person; check
        row["keyword_hints"] = ";".join(k for k, pat in HINTS.items() if re.search(pat, body, re.I))

        p_start, p_end = page_at(m.start()), page_at(max(end - 1, m.start()))
        row.update({
            "statement_id": f"S{n:03d}",
            "start_page": pages[p_start][0],
            "end_page": pages[p_end][0],
            "marginalia_on_start_page": pages[p_start][2],
            "heading_line": tidy(m.group(0))[:200],
            "n_words": len(body.split()),
            # With no thumb impression found, a statement ends at the next heading or "Translation of
            # a letter". If something else follows (a minute, a list), it runs on, so look at these.
            "end_check": ("check end" if len(body.split()) > 800 else "") if thumb
                         else ("check end (no thumb found)" if (len(body.split()) > 800 or p_end - p_start >= 2) else ""),
            "possible_duplicate_of": "",
            "ground_coded": "",
            "counts_as_application": "",
            "notes": "",
            "text": body.strip(),
        })
        rows.append(row)

    # Same name and date as an earlier row: usually a copy of the same statement.
    seen = {}
    for r in rows:
        key = (r["name_raw"].lower(), r["date_raw"])
        if r["date_raw"]:
            if key in seen:
                r["possible_duplicate_of"] = seen[key]
            else:
                seen[key] = r["statement_id"]

    cols = ["statement_id", "start_page", "end_page", "marginalia_on_start_page", "heading_line",
            "name_raw", "sex_auto", "sex_marker", "age_stated", "date_raw", "born_at",
            "origin_term_in_name", "father_name_status", "authentication_auto",
            "certificate_numbers_nearby", "keyword_hints", "n_words", "end_check", "possible_duplicate_of",
            "ground_coded", "counts_as_application", "notes", "text"]
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} statement heading(s) found -> {OUTPUT_CSV}")
    if not rows:
        print("\nNo headings matched. First lines of the first three pages, for checking the layout:")
        for stem, _, _, raw in pages[:3]:
            print(f"--- {stem} ---\n{raw[:300]}\n")
        print("Lines containing 'statement' (first 5):")
        shown = 0
        for stem, _, _, raw in pages:
            for line in raw.splitlines():
                if re.search(r"statement", line, re.I) and shown < 5:
                    print(f"  [{stem}] {line.strip()[:150]}")
                    shown += 1
    for col in ("sex_auto", "father_name_status", "authentication_auto"):
        print(f"  {col}: {dict(Counter(r[col] or '(blank)' for r in rows))}")
    print("These tallies are automatic. Check headings against the pages and code grounds before reporting any count.")


if __name__ == "__main__":
    main()