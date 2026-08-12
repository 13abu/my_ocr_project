# my_ocr_project

Bulk transcription pipeline for the **India Office Records, R/15 series** (Qatar Digital Library) — built to support an MA IRP/thesis on the British Gulf Residency archive, MA Digital Humanities and Societies, Hamad Bin Khalifa University.

## What this does

Downloads and transcribes scanned manuscript/typescript pages from QDL using the Gemini API as a lower-cost, faster-to-set-up alternative to Transkribus for pilot-stage corpus work. Outputs plain-text transcripts, one `.txt` file per source image.

## Repo structure

├── config.py # loads API key from an external .cfg file
├── transcribe_bulk.py # main transcription script
├── images/ # source page images, named IOR_R_15_<series><file><image>
└── transcriptions/ # output .txt files, one per image


## Setup

1. Clone the repo and install dependencies:
```bash
   pip install google-genai pillow
```
2. Create a config file at `~/qdl_ocr.cfg` (or `%USERPROFILE%\qdl_ocr.cfg` on Windows) — **not tracked in this repo**:
```ini
   [gemini-credentials]
   api-key = YOUR_GEMINI_API_KEY_HERE
```
3. Place source images in `images/`.

## Usage

```bash
python3 transcribe_bulk.py
```

The script skips any image that already has a matching `.txt` file in `transcriptions/`, so it can be safely re-run to pick up new images without re-processing existing ones.

## Model notes

Currently using `gemini-3.5-flash-lite` for its higher free-tier daily quota (500 RPD vs. 20 RPD on earlier Flash models). Benchmarked against a ground-truth transcription of an 1812 IOR manuscript page at ~29% Word Error Rate / ~9% Character Error Rate — usable for triage and bulk first-pass transcription, not a substitute for direct verification against the source image before quoting in written work.

## Naming convention

Images follow QDL's own reference format: `IOR_R_15_<series>_<file>_<image number>`, e.g. `IOR_R_15_1_753_0093`. Folio numbers (where confirmed by direct reading) are appended manually at the point of transcription, e.g. `IOR_R_15_1_753_0093_47r`.

## Status

Pilot-stage corpus for an MA IRP examining how the British Gulf Residency archive (IOR/R/15) produced unequal ontological status for different bodies through administrative, affective, and financial documentary practices. Not intended as a general-purpose OCR tool.