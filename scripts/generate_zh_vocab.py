import os
import pandas as pd

vocab_df = pd.DataFrame({"char": []})

vocab_chars = set(vocab_df["char"].astype(str))

# Basic CJK Unified Ideographs: U+4E00 to U+9FFF (common Chinese characters)
cjk_chars = [chr(cp) for cp in range(0x4E00, 0xA000)]

# Characters not yet in your CSV
missing_cjk = [ch for ch in cjk_chars if ch not in vocab_chars]

print("Missing basic CJK chars:", len(missing_cjk))

# Build extended vocab
extra_chars = [
    "!", "\\", "#", "$", "%", "&", "'", "(", ")", "*", "+", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ":", ";", "<", "=", ">", "?", "@",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "[", "]", "^", "_", "`",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "{", "|", "}", "~",
    "²", "´", "·", "×", "é",
    "а", "е", "и", "н", "о", "р", "с", "т",
    "—", "―", "‘", "’", "“", "”", "•", "․", "..", "...", "‧",
    "★", "☆", "☞", "♀", "♡", "♥", "✔", "✨", "❤",
    "「", "」", "『", "』", "【", "】", "〔", "〕",
    "の", "・", "＂", "％", "＆", "＋", "，", "－", "：", "；",
]

extra_df = pd.DataFrame({"char": missing_cjk + extra_chars})
full_df = pd.concat([vocab_df, extra_df],
                    ignore_index=True).drop_duplicates("char")

full_df.to_csv("assets_zh/ko_vocab.csv", index=False)
