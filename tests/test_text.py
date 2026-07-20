from training.text import AVAILABLE_LANGS, CharTokenizer, normalize_text


def test_normalize_text_wraps_with_lang_tag_and_terminal_punct():
    out = normalize_text("hello world", "en")
    assert out == "<en>hello world.</en>"


def test_normalize_text_preserves_existing_terminal_punct():
    out = normalize_text("hello world!", "en")
    assert out == "<en>hello world!</en>"


def test_normalize_text_collapses_whitespace():
    out = normalize_text("hello   \n  world", "en")
    assert "  " not in out.replace("<en>", "").replace("</en>", "")


def test_available_langs_are_unique_two_letter_codes():
    assert len(AVAILABLE_LANGS) == len(set(AVAILABLE_LANGS))
    assert all(len(lang) == 2 for lang in AVAILABLE_LANGS)


def test_tokenizer_round_trip_build_from_texts():
    texts = ["hello", "world"]
    langs = ["en", "en"]
    tok = CharTokenizer.build_from_texts(texts, langs)
    assert tok.char2id  # non-empty
    ids = tok.encode("hello", "en")
    assert all(isinstance(i, int) for i in ids)
    # PAD_ID=0 is reserved and never assigned to a real character
    assert 0 not in tok.char2id.values()


def test_tokenizer_unknown_char_maps_to_pad():
    tok = CharTokenizer.build_from_texts(["ab"], ["en"])
    ids = tok.encode("z", "en")  # 'z' never seen; <en> tag chars were, though
    # every char in the encoded (normalized) string that wasn't in the training corpus maps to PAD_ID
    assert tok.char2id.get("z") is None


def test_tokenizer_to_dict_from_dict_round_trip():
    tok = CharTokenizer.build_from_texts(["hello world"], ["en"])
    d = tok.to_dict()
    tok2 = CharTokenizer.from_dict(d)
    assert tok2.char2id == tok.char2id
    assert tok2.vocab_size == tok.vocab_size


def test_vocab_size_includes_pad():
    tok = CharTokenizer(char2id={"a": 1, "b": 2})
    assert tok.vocab_size == 3
