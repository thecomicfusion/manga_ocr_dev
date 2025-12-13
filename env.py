from pathlib import Path

ASSETS_PATH = Path(__file__).parent / "assets_ko"

VERTICAL_TEXT_PROBABILITY = 0.5
FONTS_ROOT = Path(f"{ASSETS_PATH}/ko_font").expanduser()
DATA_SYNTHETIC_ROOT = Path(f"{ASSETS_PATH}/ko_synthetic").expanduser()
BACKGROUND_DIR = Path(f"{ASSETS_PATH}/ko_background").expanduser()
TRAIN_ROOT = Path(f"{ASSETS_PATH}/ko_out").expanduser()

# Force generation of only 3-4 line images (for targeted training)
FORCE_3_4_LINES = False  # Set to False for normal mixed generation
