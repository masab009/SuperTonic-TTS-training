"""Filelist-based datasets. Expected filelist format, one utterance per line:

    relative/path/to/audio.wav|transcript text|lang_code

`lang_code` is optional and defaults to "en" (see training.text.AVAILABLE_LANGS).
Paths are resolved relative to `root_dir`. Any torchaudio-readable format works.
"""
from __future__ import annotations

import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from training.text import CharTokenizer


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Returns mono waveform (T,) and sample rate. Uses soundfile directly to
    avoid torchaudio's torchcodec/ffmpeg I/O backend dependency."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    wav = torch.from_numpy(wav).mean(dim=1)
    return wav, sr


def load_filelist(path: str) -> list[tuple[str, str, str]]:
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        wav_path, text = parts[0], parts[1]
        lang = parts[2] if len(parts) > 2 else "en"
        entries.append((wav_path, text, lang))
    return entries


class AutoencoderDataset(Dataset):
    """Stage 1: random fixed-length audio crops for the GAN-trained speech autoencoder."""

    def __init__(self, filelist: str, root_dir: str, sample_rate: int, segment_samples: int):
        self.entries = load_filelist(filelist)
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples

    def __len__(self):
        return len(self.entries)

    def _load(self, wav_path: str) -> torch.Tensor:
        wav, sr = load_audio(str(self.root_dir / wav_path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav_path, _, _ = self.entries[idx]
        wav = self._load(wav_path)
        n = self.segment_samples
        if wav.shape[0] < n:
            wav = torch.nn.functional.pad(wav, (0, n - wav.shape[0]))
        else:
            start = random.randint(0, wav.shape[0] - n)
            wav = wav[start : start + n]
        return wav


def autoencoder_collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


class TextAudioDataset(Dataset):
    """Stage 2/3: (audio, text) pairs for the text-to-latent module and duration
    predictor. Target latents are computed on the fly in the training loop by a
    frozen pretrained speech autoencoder, so this dataset just returns raw audio.
    """

    def __init__(self, filelist: str, root_dir: str, tokenizer: CharTokenizer, sample_rate: int, max_audio_seconds: float = 10.0):
        self.entries = load_filelist(filelist)
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_seconds * sample_rate)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        wav_path, text, lang = self.entries[idx]
        wav, sr = load_audio(str(self.root_dir / wav_path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > self.max_samples:
            wav = wav[: self.max_samples]
        text_ids = torch.tensor(self.tokenizer.encode(text, lang), dtype=torch.long)
        return wav, text_ids


def text_audio_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    wavs, text_ids_list = zip(*batch)
    wav_lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    text_lengths = torch.tensor([t.shape[0] for t in text_ids_list], dtype=torch.long)

    wav_padded = torch.zeros(len(wavs), int(wav_lengths.max()))
    for i, w in enumerate(wavs):
        wav_padded[i, : w.shape[0]] = w

    text_padded = torch.zeros(len(text_ids_list), int(text_lengths.max()), dtype=torch.long)
    for i, t in enumerate(text_ids_list):
        text_padded[i, : t.shape[0]] = t

    return wav_padded, wav_lengths, text_padded, text_lengths
