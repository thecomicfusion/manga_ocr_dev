import csv
from pathlib import Path

META_DIR = Path("assets/ko_synthetic/meta_ko")
ASSET_DIR = Path("assets/ko_synthetic/img_ko")
TARGET_FONT = "Hakgyoansim_SiganpyoR.ttf"
IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp"]


def delete_images_for_id(folder_id: str, sample_id: str) -> None:
    """
    Delete images whose filename starts with the given sample_id (before extension).
    """
    deleted_any = False
    for ext in IMG_EXTS:
        img_path = ASSET_DIR / folder_id / f"{sample_id}{ext}"
        if img_path.exists():
            print(f"Deleting: {img_path}")
            img_path.unlink()
            deleted_any = True
    if not deleted_any:
        print(f"No image found for id={sample_id}")


def main():
    for csv_path in META_DIR.glob("*.csv"):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("font_path") == TARGET_FONT:
                    sample_id = row.get("id")
                    if not sample_id:
                        continue
                    delete_images_for_id(
                        csv_path.name.split(".")[0], sample_id)


if __name__ == "__main__":
    main()
