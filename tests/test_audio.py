import wave

import numpy as np
import pytest
import soundfile as sf

from clinic_scribe.audio import AudioFileError, load_audio_file, rms_dbfs, to_whisper_pcm, write_wav


def tone(rate: int, seconds: float, channels: int = 1, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    x = (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.stack([x] * channels, axis=1)


def test_48k_stereo_becomes_16k_mono_int16():
    pcm = to_whisper_pcm(tone(48000, 1.0, channels=2), 48000)
    assert pcm.dtype == np.int16 and pcm.ndim == 1
    assert len(pcm) == 16000
    assert 0.45 < np.max(np.abs(pcm)) / 32767 < 0.55  # 레벨이 유지된다


def test_44k_duration_is_preserved():
    pcm = to_whisper_pcm(tone(44100, 2.5), 44100)
    assert abs(len(pcm) - 40000) <= 2


def test_out_of_range_samples_are_clipped():
    pcm = to_whisper_pcm(np.full((1600, 1), 1.7, dtype=np.float32), 16000)
    assert pcm.max() == 32767


def test_written_wav_is_16k_mono_pcm16(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, to_whisper_pcm(tone(48000, 0.5), 48000))
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 8000)


def test_load_float_wav(tmp_path):
    path = tmp_path / "float.wav"
    sf.write(path, tone(44100, 1.0, channels=2), 44100, subtype="FLOAT")
    data, rate = load_audio_file(path)
    assert rate == 44100 and data.shape == (44100, 2)


def test_unreadable_file_raises(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_text("not audio")
    with pytest.raises(AudioFileError):
        load_audio_file(path)


def test_rms_dbfs():
    assert rms_dbfs(np.zeros(100, np.float32)) == -120.0
    assert abs(rms_dbfs(tone(16000, 1.0, amp=1.0)[:, 0]) + 3.01) < 0.05
