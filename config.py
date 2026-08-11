import os
import platform
import configparser

HOME_DIR = os.environ["USERPROFILE"] if platform.system() == "Windows" else os.environ["HOME"]
config_file_full_path = os.path.join(HOME_DIR, "qdl_ocr.cfg")

config = configparser.ConfigParser()
config.read(config_file_full_path)

try:
    API_KEY = config["gemini-credentials"]["api-key"]
except KeyError:
    raise Exception(
        f"Could not find [gemini-credentials] api-key in {config_file_full_path}. "
        f"Create the file with:\n\n[gemini-credentials]\napi-key = YOUR_KEY_HERE"
    )