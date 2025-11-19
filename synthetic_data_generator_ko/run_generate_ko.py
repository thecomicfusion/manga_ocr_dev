import sys
from pathlib import Path

# Add parent directory to Python path so manga_ocr_dev can be imported
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


import traceback
from pathlib import Path

import cv2
import fire
import pandas as pd
from tqdm.contrib.concurrent import thread_map

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator_ko.generator import KoreanSyntheticDataGenerator


generator = KoreanSyntheticDataGenerator()


def _worker(args):
    try:
        i, source, id_, text = args
        filename = f"{id_}.jpg"
        img, text_gt, params = generator.process(text)

        # Skip this sample if rendering failed
        if img is None:
            return None

        cv2.imwrite(str(OUT_DIR / filename), img)

        font_path = Path(params["font_path"]).relative_to(FONTS_ROOT)
        ret = source, id_, text_gt, params["vertical"], str(font_path)
        return ret
    except Exception:
        print(traceback.format_exc())
        return None


def run(package=0, n_random=1000, n_limit=None, max_workers=16):
    """
    Generate a package of synthetic Korean image–text pairs.

    The script expects a CSV file at:
        DATA_SYNTHETIC_ROOT / "lines" / f"{package:04d}.csv"
    with at least the columns: ["source", "id", "line"].

    Additional `n_random` rows with random Korean text are appended.
    """
    package = f"{package:04d}"

    lines_path = DATA_SYNTHETIC_ROOT / "lines" / f"{package}.csv"
    if lines_path.exists():
        lines = pd.read_csv(lines_path)
    else:
        # If you don't have a corpus yet, you can still generate random-only samples.
        lines = pd.DataFrame(columns=["source", "id", "line"])

    random_lines = pd.DataFrame(
        {
            "source": "random_ko",
            "id": [f"random_ko_{package}_{i}" for i in range(n_random)],
            "line": None,
        }
    )
    lines = pd.concat([lines, random_lines], ignore_index=True)
    if n_limit:
        lines = lines.sample(n_limit)

    args = [(i, *values) for i, values in enumerate(lines.values)]

    global OUT_DIR
    OUT_DIR = DATA_SYNTHETIC_ROOT / "img_ko" / package
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = thread_map(_worker, args, max_workers=max_workers, desc=f"Processing Korean package {package}")
    
    # Filter out None values (failed generations)
    data = [d for d in data if d is not None]
    
    if not data:
        print("Warning: No data was successfully generated!")
        return
    
    data = pd.DataFrame(data, columns=["source", "id", "text", "vertical", "font_path"])
    meta_path = DATA_SYNTHETIC_ROOT / "meta_ko" / f"{package}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(meta_path, index=False)


if __name__ == "__main__":
    fire.Fire(run)


