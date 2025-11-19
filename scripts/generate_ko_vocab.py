import pandas as pd

# Load your existing vocab
vocab_df = pd.DataFrame({"char": []})
vocab_chars = set(vocab_df["char"].astype(str))

# All modern Hangul syllables: U+AC00 to U+D7A3
hangul_syllables = [chr(cp) for cp in range(0xAC00, 0xD7A4)]

# Syllables not yet in your CSV
missing_hangul = [ch for ch in hangul_syllables if ch not in vocab_chars]

print("Missing modern Hangul syllables:", len(missing_hangul))

# If you want to create an extended CSV:
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

extra_df = pd.DataFrame({"char": missing_hangul + extra_chars})
full_df = pd.concat([vocab_df, extra_df],
                    ignore_index=True).drop_duplicates("char")

full_df.to_csv("assets_ko/ko_vocab.csv", index=False)
