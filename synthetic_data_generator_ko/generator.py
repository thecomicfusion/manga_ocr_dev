import re
from pathlib import Path

import numpy as np
import pandas as pd

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.synthetic_data_generator.utils import get_font_meta


def _default_korean_vocab():
    """
    Build a simple default Korean character set.

    This covers the Hangul syllables block plus common ASCII punctuation and digits.
    It is used only if an explicit vocab file is not found.
    """
    hangul_start = 0xAC00
    hangul_end = 0xD7A3
    hangul_chars = [chr(cp) for cp in range(hangul_start, hangul_end + 1)]

    ascii_chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    punctuation = list(" .,!?-…“”‘’\"'()[]:;")
    return np.array(hangul_chars + ascii_chars + punctuation)


class KoreanSyntheticDataGenerator:
    """
    Synthetic data generator for Korean text.

    Differences from the original Japanese generator:
    - No Japanese tokenization (no `budou` / tinysegmenter).
    - No furigana or ruby markup.
    - No kanji/hiragana/katakana checks.
    - Simpler whitespace/punctuation-based word splitting.
    """

    def __init__(self, vocab_path: Path | None = None):
        # Length distribution (generic, language-agnostic)
        self.len_to_p = pd.read_csv(ASSETS_PATH / "len_to_p.csv")

        # Load vocab: prefer explicit Korean vocab file if provided or present.
        if vocab_path is None:
            ko_vocab_candidate = ASSETS_PATH / "ko_vocab.csv"
            if ko_vocab_candidate.exists():
                vocab_path = ko_vocab_candidate

        if vocab_path is not None and Path(vocab_path).exists():
            vocab = pd.read_csv(vocab_path).char.values
        else:
            vocab = _default_korean_vocab()

        self.vocab = np.array(vocab)

        # Font metadata is language-agnostic; you should generate fonts.csv with Korean fonts.
        self.fonts_df, self.font_map = get_font_meta()
        self.font_labels, self.font_p = self._get_font_labels_prob()
        self.renderer = Renderer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, text: str | None = None, override_css_params: dict | None = None):
        """
        Generate (image, normalized_text, css_params) triple.

        If `text` is None, random Korean text is generated using only characters
        supported by the randomly selected font.
        """
        if override_css_params is None:
            override_css_params = {}

        if text is None:
            # If using random text, choose font first and then generate text using
            # only characters supported by that font.
            if "font_path" not in override_css_params:
                font_path = self._get_random_font()
                vocab = self.font_map[font_path]
                override_css_params["font_path"] = font_path
            else:
                font_path = override_css_params["font_path"]
                vocab = self.font_map.get(font_path, set(self.vocab))

            words = self._get_random_words(vocab)
        else:
            # Basic normalization (keep it language-agnostic).
            text = text.replace("\u3000", " ")  # full-width space -> normal space
            text = text.replace("…", "...")  # ellipsis variant
            words = self._split_into_words(text)

        lines = self._words_to_lines(words)
        text_gt = "\n".join(lines)

        # Ensure font is picked even when user provides text.
        if "font_path" not in override_css_params:
            override_css_params["font_path"] = self._get_random_font(text_gt)

        font_path = override_css_params.get("font_path")
        if font_path:
            vocab = self.font_map.get(font_path, set(self.vocab))

            # Remove characters unsupported by the chosen font.
            lines = ["".join([c for c in line if c in vocab or c.isspace()]) for line in lines]
            text_gt = "\n".join(lines)

        img, params = self.renderer.render(lines, override_css_params)
        return img, text_gt, params

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def _get_random_words(self, vocab: set[str] | np.ndarray):
        vocab = np.array(list(vocab))
        max_text_len = int(
            np.random.choice(self.len_to_p.len, p=self.len_to_p.p)
        )

        words: list[str] = []
        text_len = 0
        while True:
            word_len = int(np.random.randint(1, 5))
            word = "".join(np.random.choice(vocab, word_len))
            words.append(word)
            text_len += len(word)
            if text_len + word_len >= max_text_len:
                break

        return words

    def _split_into_words(self, text: str):
        """
        Simple tokenizer for Korean text.

        Strategy:
        - Split on whitespace.
        - Further separate trailing punctuation (e.g. "단어입니다." -> "단어입니다", ".").
        This is intentionally simple but works well enough for synthetic data.
        """
        max_text_len = int(
            np.random.choice(self.len_to_p.len, p=self.len_to_p.p)
        )

        rough_tokens = text.strip().split()
        words: list[str] = []
        text_len = 0

        for tok in rough_tokens:
            # Separate trailing punctuation like .,!? etc.
            m = re.match(r"(.+?)([.!?]+)$", tok)
            if m:
                core, punct = m.groups()
                candidates = [core, punct]
            else:
                candidates = [tok]

            for w in candidates:
                if not w:
                    continue
                words.append(w)
                text_len += len(w)
                if text_len >= max_text_len:
                    return words

        return words

    def _words_to_lines(self, words: list[str]):
        """
        Group words into multiple lines of reasonable length.
        """
        text = "".join(words)
        if not text:
            return [""]

        max_num_lines = 10
        min_line_len = max(1, len(text) // max_num_lines)
        max_line_len_cap = 25
        max_line_len = int(
            np.clip(np.random.poisson(8), min_line_len, max_line_len_cap)
        )

        lines: list[str] = []
        line = ""
        for word in words:
            if len(line) + len(word) > max_line_len and line:
                lines.append(line)
                line = ""
            line += word
        if line:
            lines.append(line)

        return lines

    # ------------------------------------------------------------------
    # Font helpers
    # ------------------------------------------------------------------
    def _is_font_supporting_text(self, font_path: str, text: str) -> bool:
        chars = self.font_map[font_path]
        for c in text:
            if c.isspace():
                continue
            if c not in chars:
                return False
        return True

    def _get_font_labels_prob(self):
        labels = {
            "common": 0.2,
            "regular": 0.75,
            "special": 0.05,
        }
        labels = {k: labels[k] for k in self.fonts_df.label.unique()}
        p = np.array(list(labels.values()))
        p = p / p.sum()
        labels = list(labels.keys())
        return labels, p

    def _get_random_font(self, text: str | None = None) -> str:
        label = np.random.choice(self.font_labels, p=self.font_p)
        df = self.fonts_df[self.fonts_df.label == label]

        if text is None:
            return df.sample(1).iloc[0].font_path

        valid_mask = df.font_path.apply(lambda x: self._is_font_supporting_text(x, text))
        if not valid_mask.any():
            # If text contains characters not supported by any font, pick fonts with many supported chars.
            valid_mask = df.num_chars >= df.num_chars.quantile(0.75)

        return str(FONTS_ROOT / df[valid_mask].sample(1).iloc[0].font_path)


