"""오디오 변환과 파일 입출력.

녹음은 장치가 지원하는 샘플레이트·채널로 받고, 전사 직전에 whisper.cpp 입력 규격
(16 kHz, mono, 16-bit PCM WAV)으로 바꾼다.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

WHISPER_SR = 16000


class AudioFileError(Exception):
    pass


def to_mono(audio: np.ndarray) -> np.ndarray:
    """(n,) 또는 (n, ch) → (n,) float32. 여러 채널은 평균으로 합친다."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1) if a.shape[1] > 1 else a[:, 0]
    return np.ascontiguousarray(a, dtype=np.float32)


def to_whisper_pcm(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """장치 샘플레이트의 float 녹음을 16 kHz mono int16 으로 바꾼다."""
    mono = to_mono(audio)
    if mono.size and sample_rate != WHISPER_SR:
        mono = soxr.resample(mono, sample_rate, WHISPER_SR, quality="HQ")
    return np.round(np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_wav(path: Path, pcm16: np.ndarray, sample_rate: int = WHISPER_SR) -> None:
    """16-bit PCM mono WAV 저장."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.ascontiguousarray(pcm16, dtype="<i2").tobytes())


def load_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """WAV/FLAC/AIFF/OGG/MP3 → ((n, ch) float32, 샘플레이트)."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile 은 형식별로 다른 예외를 던진다
        raise AudioFileError(f"오디오 파일을 읽지 못했습니다: {exc}") from exc
    if data.size == 0:
        raise AudioFileError("오디오 파일에 소리 데이터가 없습니다.")
    return data, int(sr)


def pcm_slice(pcm16: np.ndarray, start_s: float, end_s: float, sample_rate: int = WHISPER_SR) -> np.ndarray:
    a = max(0, int(start_s * sample_rate))
    b = min(len(pcm16), int(end_s * sample_rate))
    return pcm16[a:b]


def duration_s(pcm16: np.ndarray, sample_rate: int = WHISPER_SR) -> float:
    return len(pcm16) / float(sample_rate)


def rms_dbfs(block: np.ndarray) -> float:
    """float 블록의 RMS 레벨(dBFS). 디지털 무음은 -120."""
    if block.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 1e-6 else -120.0
