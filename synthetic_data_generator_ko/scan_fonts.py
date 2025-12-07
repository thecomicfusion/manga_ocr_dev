from tqdm.contrib.concurrent import process_map
from PIL import ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import pandas as pd
import numpy as np
import PIL
import sys
from pathlib import Path

# Add parent directory to Python path so manga_ocr_dev can be imported
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT  # noqa: E402

print(ASSETS_PATH)
vocab = pd.read_csv(ASSETS_PATH / "ko_vocab.csv").char.values


def has_glyph(font, glyph):
    # Only check single characters; skip multi-character strings
    if len(str(glyph)) != 1:
        return False
    for table in font["cmap"].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False


def process(font_path):
    """
    Get supported characters list for a given font.
    Font metadata is not always reliable, so try to render each character and see if anything shows up.
    Still not perfect, because sometimes unsupported characters show up as rectangles.
    """

    try:
        font_path = str(font_path)
        ttfont = TTFont(font_path)
        pil_font = ImageFont.truetype(font_path, 24)

        supported_chars = []

        for char in vocab:
            if not has_glyph(ttfont, char):
                continue

            image = PIL.Image.new("L", (40, 40), 255)
            draw = ImageDraw.Draw(image)
            draw.text((10, 0), char, 0, font=pil_font)
            if (np.array(image) != 255).sum() == 0:
                continue

            supported_chars.append(char)

        supported_chars = "".join(supported_chars)
    except Exception as e:
        print(f"Error while processing {font_path}: {e}")
        supported_chars = ""

    return supported_chars


def main():
    path_in = FONTS_ROOT
    out_path = ASSETS_PATH / "fonts.csv"

    suffixes = {".TTF", ".otf", ".ttc", ".ttf"}
    font_paths = [path for path in path_in.glob(
        "**/*") if path.suffix in suffixes]

    if not font_paths:
        print(f"No font files found in {path_in}")
        return

    data = process_map(process, font_paths, max_workers=4)

    font_paths = [str(path.relative_to(FONTS_ROOT)) for path in font_paths]
    data = pd.DataFrame({"font_path": font_paths, "supported_chars": data})
    # Ensure supported_chars is string type
    data["supported_chars"] = data["supported_chars"].astype(str)
    data["num_chars"] = data.supported_chars.str.len()
    data["label"] = "regular"
    data.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
