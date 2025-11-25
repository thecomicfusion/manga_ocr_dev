from pathlib import Path

ASSETS_PATH = Path(__file__).parent / "assets_zh"

VERTICAL_TEXT_PROBABILITY = 0.6
FONTS_ROOT = Path(f"{ASSETS_PATH}/ko_font").expanduser()
DATA_SYNTHETIC_ROOT = Path(f"{ASSETS_PATH}/ko_synthetic").expanduser()
BACKGROUND_DIR = Path(f"{ASSETS_PATH}/ko_background").expanduser()
TRAIN_ROOT = Path(f"{ASSETS_PATH}/ko_out").expanduser()