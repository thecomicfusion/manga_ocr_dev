# scripts/calculate_font_padding.py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from pathlib import Path


def calculate_font_padding(font_path, test_char="あ", font_size=48):
    """
    Calculate the inherent padding ratio for a font.
    Returns a value between 0 (no padding) and 1 (lots of padding).
    """
    try:
        font = ImageFont.truetype(font_path, font_size)

        # Create image and draw test character
        img = Image.new('L', (font_size * 2, font_size * 3), color=255)
        draw = ImageDraw.Draw(img)
        draw.text((font_size // 2, font_size), test_char, font=font, fill=0)

        # Find actual character bounds
        arr = np.array(img)
        rows = np.where(arr.min(axis=1) < 255)[0]

        if len(rows) == 0:
            return 0.0  # No character rendered

        actual_height = rows[-1] - rows[0]
        total_height = font_size

        # Padding ratio: how much of the font size is padding
        padding_ratio = 1.0 - (actual_height / total_height)
        return max(0.0, min(1.0, padding_ratio))

    except Exception as e:
        print(f"Error processing {font_path}: {e}")
        return 0.0  # Default to no padding if error


def main():
    from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT

    # Read existing fonts.csv
    fonts_df = pd.read_csv("assets_zh/fonts.csv")

    # Calculate padding for each font
    padding_values = []
    for _, row in fonts_df.iterrows():
        font_path = str("assets_zh/ko_font/" + row.font_path)
        padding = calculate_font_padding(font_path)
        padding_values.append(padding)
        print(f"{row.font_path}: padding_ratio={padding:.3f}")

    # Add padding column to dataframe
    fonts_df['padding_ratio'] = padding_values

    # IMPORTANT: Explicitly specify column order when saving
    fonts_df.to_csv("assets_zh/fonts.csv",
                    columns=['font_path', 'supported_chars',
                             'num_chars', 'label', 'padding_ratio'],
                    index=False)
    print(f"\nSaved updated fonts.csv with padding_ratio column")


if __name__ == "__main__":
    main()
