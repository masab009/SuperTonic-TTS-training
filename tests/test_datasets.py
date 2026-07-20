import numpy as np
import soundfile as sf
import torch

from training.datasets import (
    AutoencoderDataset,
    TextAudioDataset,
    autoencoder_collate,
    load_audio,
    load_filelist,
    text_audio_collate,
)
from training.text import CharTokenizer


def _write_wav(path, seconds, sample_rate, freq=220.0):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    wav = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), wav, sample_rate)


def _make_filelist(tmp_path, sample_rate):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav", 0.5, sample_rate)
    _write_wav(wav_dir / "b.wav", 1.5, sample_rate)
    filelist = tmp_path / "train.txt"
    filelist.write_text("a.wav|hello there|en\nb.wav|bonjour le monde|fr\n", encoding="utf-8")
    return filelist, wav_dir


def test_load_filelist_parses_pipe_delimited_entries(tmp_path):
    filelist, _ = _make_filelist(tmp_path, 8000)
    entries = load_filelist(filelist)
    assert entries == [("a.wav", "hello there", "en"), ("b.wav", "bonjour le monde", "fr")]


def test_load_filelist_defaults_lang_to_en(tmp_path):
    filelist = tmp_path / "f.txt"
    filelist.write_text("x.wav|just text\n", encoding="utf-8")
    entries = load_filelist(filelist)
    assert entries == [("x.wav", "just text", "en")]


def test_load_audio_returns_mono_and_sr(tmp_path):
    sr = 8000
    _write_wav(tmp_path / "m.wav", 0.2, sr)
    wav, out_sr = load_audio(str(tmp_path / "m.wav"))
    assert out_sr == sr
    assert wav.dim() == 1


def test_autoencoder_dataset_fixed_length_crop(tmp_path):
    sr = 8000
    filelist, wav_dir = _make_filelist(tmp_path, sr)
    seg_samples = 4000
    ds = AutoencoderDataset(str(filelist), str(wav_dir), sample_rate=sr, segment_samples=seg_samples)
    assert len(ds) == 2
    for i in range(len(ds)):
        wav = ds[i]
        assert wav.shape == (seg_samples,)
    batch = autoencoder_collate([ds[0], ds[1]])
    assert batch.shape == (2, seg_samples)


def test_autoencoder_dataset_pads_short_audio(tmp_path):
    sr = 8000
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "short.wav", 0.05, sr)  # 400 samples
    filelist = tmp_path / "f.txt"
    filelist.write_text("short.wav|hi|en\n", encoding="utf-8")
    ds = AutoencoderDataset(str(filelist), str(wav_dir), sample_rate=sr, segment_samples=4000)
    wav = ds[0]
    assert wav.shape == (4000,)
    assert torch.all(wav[400:] == 0)


def test_text_audio_dataset_and_collate(tmp_path):
    sr = 8000
    filelist, wav_dir = _make_filelist(tmp_path, sr)
    tok = CharTokenizer.build_from_texts(["hello there", "bonjour le monde"], ["en", "fr"])
    ds = TextAudioDataset(str(filelist), str(wav_dir), tok, sample_rate=sr, max_audio_seconds=10.0)
    wav0, ids0 = ds[0]
    wav1, ids1 = ds[1]
    assert wav0.shape[0] == int(0.5 * sr)
    assert wav1.shape[0] == int(1.5 * sr)

    wav_padded, wav_lengths, text_padded, text_lengths = text_audio_collate([ds[0], ds[1]])
    assert wav_padded.shape[0] == 2
    assert wav_padded.shape[1] == wav_lengths.max()
    assert torch.all(wav_padded[0, wav_lengths[0] :] == 0)
    assert text_padded.shape[0] == 2
    assert text_padded.shape[1] == text_lengths.max()
    assert torch.all(text_padded[0, text_lengths[0] :] == 0)


def test_text_audio_dataset_truncates_long_audio(tmp_path):
    sr = 8000
    filelist, wav_dir = _make_filelist(tmp_path, sr)
    tok = CharTokenizer.build_from_texts(["hello there", "bonjour le monde"], ["en", "fr"])
    ds = TextAudioDataset(str(filelist), str(wav_dir), tok, sample_rate=sr, max_audio_seconds=1.0)
    wav1, _ = ds[1]  # b.wav is 1.5s, should be truncated to 1.0s
    assert wav1.shape[0] == sr
