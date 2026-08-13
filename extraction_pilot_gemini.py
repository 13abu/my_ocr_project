"""
extraction_pilot_gemini.py (v2)

IRP -> thesis pilot: LLM-assisted structured extraction of person/event
tables from transcribed IOR/R/15 folios, using the Gemini API.

Changes from v1 -> v2 -> v3, based on what each run's results showed:
  - v2: Output schema flattened to plain readable strings (arrows /
    semicolons) instead of nested JSON, so the CSV pastes cleanly.
  - v2: Added a mandatory fact-level cross-check step (step 2) --
    comparing folios for conflicting dates/names/amounts. This fixed
    3 of 4 known misses from v1, but v2's runs showed it came at a
    cost: it stopped catching TESTIMONY conflicts (two people's own
    accounts of the same event disagreeing), because step 2 only
    compares discrete facts, not narratives. v2 quietly adopted the
    "official" resolution (e.g. an officer's investigation conclusion)
    as settled fact rather than preserving a denial or a differing
    self-account as a live discrepancy.
  - v3 (this version): Added step 3, TESTIMONY COMPARISON, as a
    separate pass from step 2. Step 2 asks "do two sources state a
    different date/name/amount for the same fact?" Step 3 asks "do
    two people's own descriptions of the same event disagree, even if
    no single named fact technically conflicts?" -- e.g. one person's
    statement says they were released after a routine errand, another
    person's statement of the same incident says they were seized and
    sold. These are compared and reported separately from step 2's
    output, since conflating them is what caused the regression.

Uses the same config.py / API_KEY setup as transcribe_bulk.py.
Run from your main project folder (next to transcriptions/ and config.py):

    python3 extraction_pilot_gemini.py

Output: people.csv, discrepancies.csv, plus raw JSON per case printed
to console so you can eyeball it before trusting either CSV.
"""

import re
import json
import csv
import time
from pathlib import Path

from google import genai
from google.genai import types

from config import API_KEY

TRANSCRIPTIONS_DIR = Path("./transcriptions")
PEOPLE_CSV = Path("./people.csv")
DISCREPANCIES_CSV = Path("./discrepancies.csv")
TESTIMONY_CSV = Path("./testimony_conflicts.csv")
MODELS_TO_TRY = ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
PACE_SECONDS = 5


# ==========================================
# 1. Load transcripts (same MARGINALIA-stripping logic as lexicon_analysis.py)
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


def load_transcripts(directory: Path) -> dict:
    files = {}
    for path in sorted(directory.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        body, _method = extract_body_text(raw)
        if body is not None:
            files[path.stem] = body
    return files


# ==========================================
# 2. Cases -- edit / extend as you add more
# ==========================================

CASES = {
    "Almas": ["0029_7r", "0031_8r", "0035_10r"],
    "Ghazul": ["0080_37r", "0082_38r", "0084_39r", "0086_40r", "0088_41r",
               "0090_42r", "0092_43r", "0094_44r", "0096_45r"],
    "Zahra_Nasir": ["0143_68v", "0151_72v", "0153_73v", "0155_74v",
                    "0157_75v", "0159_76v", "0161_77v", "0163_78v", "0165_79v"],
    "Anbar": ["0245_119v", "0247_120v", "0249_121v"],
}

FEWSHOT_INPUT = """[Maryam bint Mabrook case -- folios 31r-35r]

Statement/correspondence: Reza Lari sold Maryam bint Mabrook, daughter of
Mabrook (a freed slave, deceased) and Fatimah (free woman), for Rs.400 to
Sa'ad bin Ahmad al Sharif of Nejd, on 4 June 1939, in Dubai. Abdul Karim
bin Muhamad brokered the sale. Muhammad bin Habib informed Shaikh Said bin
Maktum (Ruler of Dubai). Reza Lari was arrested, flogged, and confessed.
Maryam was restored to her mother on 9 June 1939. Reza was repatriated to
Iran. Sa'ad bin Ahmad al Sharif was fined Rs.470 and released, as he was a
Saudi subject and the Political Resident decided no further action was
warranted."""

FEWSHOT_ANSWER = """{
  "people": [
    {"person_id": "P201", "canonical_name": "Maryam bint Mabrook", "role": "enslaved person",
     "origin": "", "owner_chain": "Reza Lari (seller) -> Sa'ad bin Ahmad al Sharif (purchaser)",
     "events": "sale (4 June 1939, Dubai, w/ Sa'ad bin Ahmad al Sharif): Rs.400; recovery (9 June 1939, Dubai, w/ mother Fatimah)",
     "outcome": "restored to mother; seller flogged and repatriated to Iran; purchaser fined Rs.470"},
    {"person_id": "P202", "canonical_name": "Reza Lari (Reza Habib Mari)", "role": "seller / Iranian subject",
     "origin": "Iran", "owner_chain": "",
     "events": "sale (4 June 1939, Dubai, w/ Sa'ad bin Ahmad al Sharif): Rs.400",
     "outcome": "flogged; confessed; repatriated to Iran"},
    {"person_id": "P203", "canonical_name": "Sa'ad bin Ahmad al Sharif", "role": "purchaser / trader",
     "origin": "Nejd", "owner_chain": "",
     "events": "purchase (4 June 1939, Dubai, w/ Reza Lari): Rs.400",
     "outcome": "fined Rs.470 and released"}
  ],
  "discrepancies": [],
  "testimony_conflicts": []
}"""

# Second, small worked example, specifically for step 3 -- synthetic
# (not drawn from your real corpus), because the point of a few-shot
# example is to show the SHAPE of a testimony conflict, and using a
# real case you already know the answer to risks the model pattern
# matching to a memorized answer rather than genuinely comparing.
TESTIMONY_FEWSHOT_INPUT = """[Synthetic example folio A]
Statement of Yusuf bin Nasir: My mistress sent me to the market with her nephew, Salim, to buy rice. On our return, my mistress accused me of stealing her ring and had me confined in the fort for two days, after which she sent Salim to bring me back to her house, where I have remained.

[Synthetic example folio B]
Statement of the mistress: Yusuf ran away from my house three months ago. I do not know his whereabouts. If he has been found, I request he be returned to me."""

TESTIMONY_FEWSHOT_ANSWER = """{"testimony_conflicts": [
  {"event": "Yusuf's confinement/disappearance", "people_involved": "Yusuf bin Nasir, the mistress",
   "accounts": "Yusuf bin Nasir (folio A): says he was falsely accused of theft, confined in the fort, then returned to the mistress's house by her own nephew Salim -- i.e. she knew where he was the whole time. | Mistress (folio B): says he ran away three months ago and she does not know his whereabouts.",
   "note": "These are not compatible accounts of the same period -- one describes ongoing custody, the other describes an unexplained disappearance. Neither is confirmed by an officer's investigation in what's given here."}
]}"""

SYSTEM_PROMPT = """You are extracting structured person/event data from transcribed British colonial administrative documents (IOR/R/15 series, Persian Gulf Residency, early-mid 20th century) -- manumission statements, correspondence, and administrative case files concerning enslaved persons in the Gulf.

Do this in TWO SEPARATE STEPS. Do not skip step 2 or fold it into step 1.

STEP 1 -- EXTRACT PEOPLE
For every named PERSON who is a participant in the case (include officials who took a direct action -- granting a certificate, ordering an investigation, receiving a petition -- but not ones merely cc'd), output an object with:
- person_id: placeholder like "PXXX", unique within this case
- canonical_name: name as given in the text
- role: e.g. "enslaved person", "owner", "kidnapper", "Residency Agent", "Ruler", "petitioner", "witness"
- origin: place of origin if stated, else empty string
- owner_chain: if this person was owned/sold, a single string showing owners in order, joined by " -> ". Else empty string.
- events: a single string, each event separated by "; ", in the form "event_type (date, place, w/ other_party): value" -- event_type one of kidnapping, sale, purchase, manumission, petition, recovery, imprisonment, death, dispute, correspondence. Omit any of date/place/other_party/value that isn't stated -- do not guess or estimate a date/value that isn't explicitly written, even from relative context ("three years ago"). If you cannot state a date without inferring it, leave it out of the string entirely rather than writing "approx."
- outcome: one sentence, what ultimately happened to this person in this case

Do NOT invent information not in the text.

STEP 2 -- MANDATORY CROSS-CHECK (separate from step 1; do this even if step 1 already felt complete)
Re-read every folio in this batch a SECOND time, one against another. For every fact that appears in TWO OR MORE folios -- a name, a date, a monetary amount, a role, an event -- explicitly compare the values. List every instance where two folios disagree, even slightly (a date a year apart, a name given two different ways for what seems to be the same person, an amount that differs between folios describing what should be the same transaction). Include cases you already silently picked one value for in step 1 -- step 2 must surface those too, not just the ones you were personally unsure about. If two folios genuinely agree on everything, this list can be empty, but check before concluding that.

For each discrepancy found, output an object with:
- field: what kind of fact conflicts (e.g. "manumission certificate date", "ransom amount", "master's name")
- person: which person this concerns
- sources: which folios/documents contain the conflicting values (use the [filename] markers in the input)
- values_found: the different values, one per source, as a single string like "f.72v: 1940; f.73v: 1939"
- note: one sentence on what this might mean or which is more likely correct, if you have a view

STEP 3 -- MANDATORY TESTIMONY COMPARISON (separate from step 2; do not merge them)
Step 2 looks for conflicting discrete facts (a date, a name, an amount). Step 3 looks for something different: when two or more people give their OWN account, in their own words, of the same event or period, do their accounts of WHAT HAPPENED agree -- not just their facts, but the shape of the story?

This matters most when one person's account was later treated as the "official" or "resolved" version by an officer's investigation -- do NOT let an officer's conclusion make you skip comparing the original accounts against each other. If a mistress says a boy was released after a routine errand and the boy himself describes being seized and sold, that is a testimony conflict even if a later letter says the matter was "resolved" -- report the conflict between the two ACCOUNTS regardless of which one an officer ultimately believed or acted on.

Look specifically for: an accused/implicated person's account of an event, compared against the account of the person it happened to, or against a petitioner's account of the same event. Denials, minimizations, and self-serving omissions all count if they change what the event sounds like, not just if they state a different fact.

For each testimony conflict found, output an object with:
- event: brief description of what event/period is being described differently
- people_involved: names of the people whose accounts differ
- accounts: one string per person's version, separated by " | ", each starting with the person's name
- note: one sentence -- was this resolved by an officer, and if so how, without letting that resolution erase the fact that the accounts themselves disagreed

If, after genuinely comparing, no two accounts in this case disagree on what happened, this list can be empty -- but check for it deliberately, the same way step 2 requires deliberately checking every repeated fact.

OUTPUT FORMAT
Output ONLY a single valid JSON object with exactly three top-level keys: "people" (array), "discrepancies" (array), and "testimony_conflicts" (array). No markdown fences, no commentary before or after.

Here is a worked example for steps 1 and 2:

INPUT:
""" + FEWSHOT_INPUT + """

OUTPUT:
""" + FEWSHOT_ANSWER + """

Here is a separate worked example, specifically for step 3 (testimony comparison -- this example is synthetic, not from your real corpus, to illustrate the SHAPE of what to look for):

INPUT:
""" + TESTIMONY_FEWSHOT_INPUT + """

OUTPUT:
""" + TESTIMONY_FEWSHOT_ANSWER + """

Now do the same for the text you are given below. Remember: steps 2 and 3 are both mandatory and are DIFFERENT checks -- step 2 compares discrete facts across sources, step 3 compares whole accounts/narratives of the same event, even when no single fact technically contradicts. Do both, even for things you already resolved in step 1."""


def extract_case(client: genai.Client, case_text: str) -> dict:
    contents = [f"Now do the same for this text:\n\n{case_text}"]
    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            text = response.text.strip()
            text = re.sub(r"^```(json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                print(f"  WARNING: could not parse JSON response from {model_name}. Raw output:\n{text}\n")
                return {"people": [], "discrepancies": [], "testimony_conflicts": []}
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg or "NOT_FOUND" in err_msg:
                print(f"  {model_name} unavailable, trying next model...")
                last_error = e
                continue
            raise
    raise SystemExit(f"None of the configured models were available. Last error: {last_error}")


def main():
    if not TRANSCRIPTIONS_DIR.exists():
        raise SystemExit(f"Could not find {TRANSCRIPTIONS_DIR}. Run this next to your transcriptions/ folder.")

    client = genai.Client(api_key=API_KEY)
    files = load_transcripts(TRANSCRIPTIONS_DIR)
    print(f"Loaded {len(files)} transcripts from {TRANSCRIPTIONS_DIR}\n")

    all_people = []
    all_discrepancies = []
    all_testimony_conflicts = []

    for case_name, folio_fragments in CASES.items():
        matched_files = []
        for frag in folio_fragments:
            matches = [f for f in files if frag in f]
            if matches:
                matched_files.append(matches[0])
            else:
                print(f"  WARNING: no file matched fragment '{frag}' for case {case_name}")

        case_text = "\n\n---\n\n".join(f"[{fname}]\n{files[fname]}" for fname in matched_files)

        print(f"Extracting case: {case_name} ({len(matched_files)} folios)...")
        result = extract_case(client, case_text)
        people = result.get("people", [])
        discrepancies = result.get("discrepancies", [])
        testimony_conflicts = result.get("testimony_conflicts", [])
        print(f"  -> {len(people)} people, {len(discrepancies)} fact discrepancies, "
              f"{len(testimony_conflicts)} testimony conflicts flagged\n")

        for person in people:
            person["case"] = case_name
            all_people.append(person)
        for disc in discrepancies:
            disc["case"] = case_name
            all_discrepancies.append(disc)
        for tc in testimony_conflicts:
            tc["case"] = case_name
            all_testimony_conflicts.append(tc)

        time.sleep(PACE_SECONDS)

    # Write people.csv -- flat, readable columns
    with open(PEOPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "person_id", "canonical_name", "role", "origin",
                          "owner_chain", "events", "outcome"])
        for row in all_people:
            writer.writerow([
                row.get("case", ""), row.get("person_id", ""), row.get("canonical_name", ""),
                row.get("role", ""), row.get("origin", ""), row.get("owner_chain", ""),
                row.get("events", ""), row.get("outcome", ""),
            ])
    print(f"Wrote {len(all_people)} rows to {PEOPLE_CSV}")

    # Write discrepancies.csv
    with open(DISCREPANCIES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "field", "person", "sources", "values_found", "note"])
        for row in all_discrepancies:
            writer.writerow([
                row.get("case", ""), row.get("field", ""), row.get("person", ""),
                row.get("sources", ""), row.get("values_found", ""), row.get("note", ""),
            ])
    print(f"Wrote {len(all_discrepancies)} rows to {DISCREPANCIES_CSV}")

    # Write testimony_conflicts.csv
    with open(TESTIMONY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "event", "people_involved", "accounts", "note"])
        for row in all_testimony_conflicts:
            writer.writerow([
                row.get("case", ""), row.get("event", ""), row.get("people_involved", ""),
                row.get("accounts", ""), row.get("note", ""),
            ])
    print(f"Wrote {len(all_testimony_conflicts)} rows to {TESTIMONY_CSV}")

    print("\nRaw JSON per case, for eyeballing before you trust any CSV:\n")
    for case_name in CASES:
        case_people = [p for p in all_people if p.get("case") == case_name]
        case_disc = [d for d in all_discrepancies if d.get("case") == case_name]
        case_tc = [t for t in all_testimony_conflicts if t.get("case") == case_name]
        print(f"=== {case_name} ===")
        print(json.dumps({"people": case_people, "discrepancies": case_disc,
                           "testimony_conflicts": case_tc}, indent=2))
        print()


if __name__ == "__main__":
    main()