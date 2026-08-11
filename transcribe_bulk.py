import os
import time
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image

from config import API_KEY

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = Path("./images")
OUTPUT_FOLDER = Path("./transcriptions")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

PROMPT = """
You are an expert paleographer and archivist. Transcribe this document exactly as it appears.

Rules:
1. Maintain original line breaks, spelling, and punctuation — including apparent typos or errors. Never silently correct, normalize, or "fix" a word, even if the intended word seems obvious.
2. If text is struck through, overtyped, or crossed out, transcribe it inside [struck: ...]. If it is fully illegible, use [struck: illegible] instead of omitting it.
3. If a word or phrase is illegible for any other reason, use [illegible] — never guess or silently skip it.
4. Preserve underlining by wrapping the underlined text in underscores, e.g. _January 16th_.
5. Transcribe marginal annotations separately from the main body text under a heading "MARGINALIA:" — this includes folio/page numbers, stamps, circled figures, and any handwritten notes outside the main text block, with their approximate position (e.g. "top right").
6. If the document contains both typewritten and handwritten portions, note which is which if it's unambiguous — otherwise transcribe without labeling.
7. Do not add commentary, introductory notes, or summary. Return only the labeled transcription.
"""


# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    client = genai.Client(api_key=API_KEY)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in INPUT_FOLDER.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print(f"Found {len(image_files)} image(s) to process.\n")

    for idx, img_path in enumerate(image_files, start=1):
        output_txt_path = OUTPUT_FOLDER / f"{img_path.stem}.txt"

        # Skip already completed files
        if output_txt_path.exists():
            print(f"[{idx}/{len(image_files)}] Skipping {img_path.name} (already transcribed)")
            continue

        print(f"[{idx}/{len(image_files)}] Transcribing {img_path.name}...")

        image = Image.open(img_path)

        # models
        models_to_try = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
        success = False

        for model_name in models_to_try:
            if success:
                break

            delay = 12
            retries = 0

            while retries < 3 and not success:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[image, PROMPT],
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                        )
                    )

                    with open(output_txt_path, "w", encoding="utf-8") as f:
                        f.write(response.text)

                    print(f" -> Saved to {output_txt_path.name} (using {model_name})")
                    success = True
                    # Pause 13 seconds between requests to maintain <5 RPM on free tier
                    time.sleep(13)

                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        print(f" -> Rate limit reached. Waiting {delay}s before retrying...")
                        time.sleep(delay)
                        delay = min(delay * 2, 60)
                        retries += 1
                    elif "404" in err_msg or "NOT_FOUND" in err_msg:
                        # If model identifier is rejected, break loop to try next model in list
                        print(f" -> {model_name} unavailable, trying fallback model...")
                        break
                    else:
                        print(f" -> Error processing {img_path.name}: {e}")
                        break


if __name__ == "__main__":
    main()