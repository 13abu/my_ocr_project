# my_ocr_project

Pipeline for the **India Office Records, R/15 series** (Qatar Digital Library) — built to support an MA IRP on the British Gulf Residency archive and how it produced unequal ontological status for different bodies, MA Digital Humanities and Societies, Hamad Bin Khalifa University.

## Repo structure

```
├── config.py                          # loads API key from an external .cfg file
├── transcribe_bulk.py                 # bulk transcription (Gemini API)
├── extraction_pilot_gemini.py         # LLM extraction pilot (people/dates/discrepancies)
├── affective_register_analysis.py     # Instrument 1: affective lexicon + register-shift
├── ner_analysis.py                    # Instrument 2: personhood/provenance grammar (NER)
├── subject_network.py                 # Instrument 3: subject correspondence network
├── officer_network.py                 # Officer-hierarchy network (supporting finding, separate from Instrument 3 proper)
├── requirements.txt                   # third-party dependencies
├── images/                            # source page images, named IOR_R_15_<series><file><image>
├── transcriptions/                    # output .txt files, one per image
├── network/                           # PNG/JSON/GraphML outputs from subject_network.py / officer_network.py
└── spreadsheet.tsv                    # corpus metadata: IOR Ref, Folio, Date, Author/Officer, Recipient, Subject, etc.
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create a config file at `~/qdl_ocr.cfg` (`%USERPROFILE%\qdl_ocr.cfg` on Windows) — **not tracked in this repo**:
```ini
[gemini-credentials]
api-key = YOUR_GEMINI_API_KEY_HERE
```

## Pipeline, in order

1. **`transcribe_bulk.py`** — transcribes page images into `transcriptions/`. Skips already-transcribed files, safe to re-run.
2. **`affective_register_analysis.py`** — Instrument 1. Base 4-category lexicon, intensifier scoring, formula-vs-singularity scoring, and (given `spreadsheet.tsv`) author/subject/recipient-level aggregation and mourning×deference co-occurrence. Run from the project folder, next to `transcriptions/`. Stdlib only — no third-party dependencies for this one.
3. **`ner_analysis.py`** — Instrument 2. Runs spaCy NER as a baseline, then a rule layer on top (bare titles, "of X" origin markers, "bin/bint X" patronymic chains) to test whether the personhood/provenance distinction is recoverable from surface patterns.
4. **`subject_network.py`** / **`officer_network.py`** — Instrument 3, split into two separate questions. `subject_network.py`: which bodies generate genuine replies vs. administrative endpoints, tiered by ontological status. `officer_network.py`: institutional correspondence hierarchy (a separate finding — native Residency Agents outrank every British officer in raw connections). Both need `spreadsheet.tsv`.
5. **`extraction_pilot_gemini.py`** — separate LLM-based pilot testing structured extraction (people, roles, ownership chains) and cross-folio discrepancy detection. Not a source of citable findings on its own — a triage/retrieval aid, validated by hand against known cases.

All four analysis scripts run from the project root — none expect to be moved into their own subfolders without updating their path constants (`TRANSCRIPTIONS_DIR`, `INPUT_TSV`, etc. near the top of each file).

## `spreadsheet.tsv`

Required columns: `IOR Ref`, `Folio`, `Date`, `Author/Officer`, `Recipient`, `Subject`, `Doc Type`, `Language(s)`, `Description`, `QDL Link`. One row per subject — multi-subject folios are split into one row per person, not semicolon-joined, since the network scripts only read the first name before a semicolon.

## Model notes

Transcription uses `gemini-3.5-flash-lite` (500 RPD free-tier quota vs. 20 RPD on earlier Flash models). Benchmarked against a ground-truth transcription of an 1812 IOR manuscript page at ~29% Word Error Rate / ~9% Character Error Rate — usable for triage and bulk first-pass transcription, not a substitute for direct verification against the source image before quoting in written work. Same discipline applies to every script's output here: read against the real document before citing.

## Naming convention

Images follow QDL's own reference format: `IOR_R_15_<series>_<file>_<image number>`, e.g. `IOR_R_15_1_753_0093`. Folio numbers (where confirmed by direct reading) are appended manually at the point of transcription, e.g. `IOR_R_15_1_753_0093_47r`.

## Status

Pilot-stage corpus and analysis for an MA IRP examining how the British Gulf Residency archive (IOR/R/15) produced unequal ontological status for different bodies through administrative, affective, and financial documentary practices. Not intended as a general-purpose OCR or NLP toolkit.
