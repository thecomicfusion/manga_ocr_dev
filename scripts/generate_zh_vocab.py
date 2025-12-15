import pandas as pd
import unicodedata
from pathlib import Path


def is_valid_char(char):
    """Check if a character is valid and printable (not a control/box character)."""
    if not char or len(char) != 1:
        return False
    # Skip whitespace
    if char.isspace():
        return False
    # Get unicode category
    category = unicodedata.category(char)
    # Skip control characters (Cc), format characters (Cf),
    # surrogate (Cs), private use (Co), unassigned (Cn)
    if category.startswith(('C', 'Z')):
        return False
    # Skip characters that are not printable
    if not char.isprintable():
        return False
    return True


ASSETS_PATH = Path(__file__).parent.parent / "assets_zh"

# Load existing vocab
vocab_path = ASSETS_PATH / "ko_vocab.csv"
if vocab_path.exists():
    vocab_df = pd.read_csv(vocab_path)
    # Filter out invalid/box characters from existing vocab
    vocab_df = vocab_df[vocab_df["char"].astype(str).apply(is_valid_char)]
    vocab_chars = set(vocab_df["char"].astype(str))
    print(f"Loaded {len(vocab_chars)} valid characters from ko_vocab.csv")
else:
    vocab_df = pd.DataFrame({"char": []})
    vocab_chars = set()

# Extract unique characters from comic_sentences.txt
sentences_path = ASSETS_PATH / "comic_sentences.txt"
new_chars_from_sentences = set()

if sentences_path.exists():
    print(f"Reading {sentences_path}...")
    with open(sentences_path, "r", encoding="utf-8") as f:
        for line in f:
            for char in line.strip():
                if char and char not in vocab_chars:
                    new_chars_from_sentences.add(char)
    print(
        f"Found {len(new_chars_from_sentences)} new unique characters from comic_sentences.txt")
else:
    print(f"Warning: {sentences_path} not found!")

# Basic CJK Unified Ideographs: U+4E00 to U+9FFF (common Chinese characters)
cjk_chars = [chr(cp) for cp in range(0x4E00, 0xA000)]

# Characters not yet in your CSV
missing_cjk = [ch for ch in cjk_chars if ch not in vocab_chars]
print(f"Missing basic CJK chars: {len(missing_cjk)}")

# Extra punctuation and special characters
extra_chars = [
    "!", "\\", "#", "$", "%", "&", "'", "(", ")", "*", "+", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ":", ";", "<", "=", ">", "?", "@",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "[", "]", "^", "_", "`",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "{", "|", "}", "~",
    "²", "´", "·", "×", "é",
    "а", "е", "и", "н", "о", "р", "с", "т",
    "—", "―", "'", "'", """, """, "•", "․", "‧",
    "★", "☆", "☞", "♀", "♡", "♥", "✔", "✨", "❤",
    "「", "」", "『", "』", "【", "】", "〔", "〕",
    "の", "・", "＂", "％", "＆", "＋", "，", "－", "：", "；",
]

# Combine all new characters
all_new_chars = set(missing_cjk) | set(extra_chars) | new_chars_from_sentences
# Remove any that are already in vocab and filter out invalid/box characters
all_new_chars = [
    ch for ch in all_new_chars if ch not in vocab_chars and is_valid_char(ch)]

print(f"Total new characters to add: {len(all_new_chars)}")

# Build extended vocab
extra_df = pd.DataFrame({"char": list(all_new_chars)})
full_df = pd.concat([vocab_df, extra_df],
                    ignore_index=True).drop_duplicates("char")

print(f"Final vocab size: {len(full_df)}")
full_df.to_csv(vocab_path, index=False)
print(f"Saved to {vocab_path}")
