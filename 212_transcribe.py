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

# Tried in order. When a model hits its daily cap (or is unavailable) the script
# moves to the next one for the rest of the run instead of retrying it on every image.
# Both Flash Lite models show 15 requests/minute and 500 requests/day on the dashboard.
# Add a Gemma 4 model id here only after testing it on a few pages you have already checked.
MODELS_TO_TRY = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

# Stay under the 15/minute limit with some headroom. This is the minimum gap between
# the starts of two requests; the call's own duration counts towards it.
REQUESTS_PER_MINUTE = 12
MIN_INTERVAL = 60 / REQUESTS_PER_MINUTE

MAX_RETRIES = 3

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
# HELPERS
# ==========================================
def is_daily_cap(err_msg: str) -> bool:
    low = err_msg.lower()
    return "perday" in low or "per day" in low


def is_rate_limit(err_msg: str) -> bool:
    return "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg


def is_not_found(err_msg: str) -> bool:
    return "404" in err_msg or "NOT_FOUND" in err_msg


def is_overloaded(err_msg: str) -> bool:
    low = err_msg.lower()
    return "503" in err_msg or "unavailable" in low or "overloaded" in low


def empty_reason(response) -> str:
    """Why did the model return no text? Returns a short description."""
    parts = []
    try:
        parts.append(f"finish_reason={response.candidates[0].finish_reason}")
    except Exception:
        parts.append("finish_reason=unknown")
    try:
        block = response.prompt_feedback.block_reason
        if block:
            parts.append(f"block_reason={block}")
    except Exception:
        pass
    return ", ".join(parts)


def transcribe(client, model_name, image, state):
    """Try one image on one model.

    Returns (status, text). status is:
      "ok"          text is the transcription
      "blank"       the model returned nothing but stopped normally (usually a blank page);
                    text is a marker line to save in place of a transcription
      "exhausted"   daily cap or repeated rate-limit
      "unavailable" model id rejected
      "error"       skip this image; text is the reason
    """
    delay = 12
    retries = 0
    empties = 0

    while retries < MAX_RETRIES:
        # Keep the request rate under the per-minute limit.
        wait = MIN_INTERVAL - (time.time() - state["last_start"])
        if wait > 0:
            time.sleep(wait)
        state["last_start"] = time.time()

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[image, PROMPT],
                config=types.GenerateContentConfig(temperature=0.1),
            )
            try:
                text = response.text
            except Exception:
                text = None

            if text and text.strip():
                return "ok", text

            # No text came back. Try once more, then decide what it was.
            reason = empty_reason(response)
            empties += 1
            if empties < 2:
                print(f" -> Empty reply ({reason}). Trying once more...")
                continue
            if "STOP" in reason and "block_reason" not in reason:
                return "blank", f"[NO TEXT RETURNED BY MODEL ({reason}). Probably a blank page. Check the image.]"
            return "error", f"empty reply ({reason})"

        except Exception as e:
            err_msg = str(e)
            if is_not_found(err_msg):
                return "unavailable", None
            if is_rate_limit(err_msg):
                if is_daily_cap(err_msg):
                    return "exhausted", None
                print(f" -> Rate limit reached. Waiting {delay}s before retrying...")
            elif is_overloaded(err_msg):
                print(f" -> {model_name} busy. Waiting {delay}s before retrying...")
            else:
                print(f" -> Error: {e}")
                return "error", None
            time.sleep(delay)
            delay = min(delay * 2, 60)
            retries += 1

    return "exhausted", None


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

    state = {"last_start": 0.0}
    model_idx = 0
    saved = 0

    for idx, img_path in enumerate(image_files, start=1):
        output_txt_path = OUTPUT_FOLDER / f"{img_path.stem}.txt"

        # Skip already completed files
        if output_txt_path.exists():
            print(f"[{idx}/{len(image_files)}] Skipping {img_path.name} (already transcribed)")
            continue

        if model_idx >= len(MODELS_TO_TRY):
            break

        print(f"[{idx}/{len(image_files)}] Transcribing {img_path.name}...")
        image = Image.open(img_path)

        while model_idx < len(MODELS_TO_TRY):
            model_name = MODELS_TO_TRY[model_idx]
            status, text = transcribe(client, model_name, image, state)

            if status in ("ok", "blank"):
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                label = "Saved" if status == "ok" else "Saved a blank-page marker"
                print(f" -> {label} to {output_txt_path.name} (using {model_name})")
                saved += 1
                break
            if status in ("exhausted", "unavailable"):
                reason = "daily cap or rate limit reached" if status == "exhausted" else "not available"
                print(f" -> {model_name}: {reason}. Switching model for the rest of this run.")
                model_idx += 1
                continue
            # status == "error": nothing is saved, so the next run tries this image again
            print(f" -> Skipped {img_path.name}: {text}")
            with open(OUTPUT_FOLDER / "_failures.txt", "a", encoding="utf-8") as f:
                f.write(f"{img_path.name}\t{model_name}\t{text}\n")
            break

    if model_idx >= len(MODELS_TO_TRY):
        print("\nAll models are at their limits. Run the script again later; finished files are skipped.")
    print(f"\nDone. {saved} new transcription(s) saved.")


if __name__ == "__main__":
    main()