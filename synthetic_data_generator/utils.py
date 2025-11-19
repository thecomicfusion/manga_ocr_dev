import pandas as pd
import unicodedata
from PIL import Image

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT


def get_background_df(background_dir):
    background_df = []
    for path in background_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        
        # Try to parse coordinates from filename (format: something_ymin_ymax_xmin_xmax.ext)
        try:
            ymin, ymax, xmin, xmax = [int(v) for v in path.stem.split("_")[-4:]]
            h = ymax - ymin
            w = xmax - xmin
        except (ValueError, IndexError):
            # If filename doesn't have coordinates, read actual image dimensions
            try:
                with Image.open(path) as img:
                    w, h = img.size
            except Exception:
                # Skip files that can't be read
                continue
        
        ratio = w / h if h > 0 else 1.0

        background_df.append(
            {
                "path": str(path),
                "h": h,
                "w": w,
                "ratio": ratio,
            }
        )
    background_df = pd.DataFrame(background_df)
    return background_df



def get_font_meta():
    df = pd.read_csv(ASSETS_PATH / "fonts.csv")
    df.font_path = df.font_path.apply(lambda x: str(FONTS_ROOT / x))
    font_map = {row.font_path: set(row.supported_chars) for row in df.dropna().itertuples()}
    return df, font_map
