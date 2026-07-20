"""Character-level tokenizer matching the released model's UnicodeProcessor
(supertonic/py/helper.py): raw text is normalized, wrapped in a <lang>...</lang>
tag, and each character is mapped to its Unicode code point. Training only needs
a bijection from characters to small integer ids, so we build the code-point
indexer lazily from the training corpus instead of shipping the 65536-entry table.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

AVAILABLE_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr",
    "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk",
    "sl", "sv", "tr", "uk", "vi", "na", "ur",
]

_PUNCT_END = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")


def normalize_text(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not _PUNCT_END.search(text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


@dataclass
class CharTokenizer:
    """Maps characters (by Unicode code point) to contiguous ids. `PAD` is always id 0."""

    char2id: dict = field(default_factory=dict)

    PAD_ID: int = 0

    @classmethod
    def build_from_texts(cls, texts: list[str], langs: list[str]) -> "CharTokenizer":
        chars = set()
        for text, lang in zip(texts, langs):
            chars.update(normalize_text(text, lang))
        char2id = {c: i + 1 for i, c in enumerate(sorted(chars))}
        return cls(char2id=char2id)

    def extend_with_texts(self, texts: list[str], langs: list[str]) -> "CharTokenizer":
        """Returns a new tokenizer that keeps every existing char->id mapping unchanged
        (so a pretrained embedding table stays row-aligned) and appends ids for any
        character in `texts` that isn't already covered -- e.g. a new language's script
        missing from the ported model's unicode_indexer.json."""
        chars = set()
        for text, lang in zip(texts, langs):
            chars.update(normalize_text(text, lang))
        new_chars = sorted(chars - self.char2id.keys())
        next_id = max(self.char2id.values(), default=0) + 1
        char2id = dict(self.char2id)
        for c in new_chars:
            char2id[c] = next_id
            next_id += 1
        return CharTokenizer(char2id=char2id)

    def encode(self, text: str, lang: str) -> list[int]:
        text = normalize_text(text, lang)
        return [self.char2id.get(c, self.PAD_ID) for c in text]

    @property
    def vocab_size(self) -> int:
        return len(self.char2id) + 1

    def to_dict(self) -> dict:
        return self.char2id

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(char2id=d)
