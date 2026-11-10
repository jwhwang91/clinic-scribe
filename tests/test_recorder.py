import time

import numpy as np
import soundfile as sf

from clinic_scribe.recorder import FileInputStream, Recorder, demo_device


def test_snapshot_returns_requested_range():
    rec = Recorder()
    rec.samplerate, rec.channels = 1000, 1
    for i in range(10):  # 100 프레임짜리 블록 10개, 값은 블록 번호
        rec._queue.put(np.full((100, 1), i, np.float32))
    rec.drain()
    snap = rec.snapshot(0.25, 0.55)  # 250 ~ 550 프레임
    assert snap.shape == (300, 1)
    assert snap[0, 0] == 2 and snap[-1, 0] == 5
    assert rec.snapshot(2.0, 3.0).shape[0] == 0  # 아직 녹음되지 않은 구간
    assert rec.snapshot(-1.0, 0.05).shape == (50, 1)


def test_file_input_stream_paces_audio_like_a_microphone(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.full((48000, 2), 0.1, np.float32), 48000)
    blocks = []
    stream = FileInputStream(str(path), lambda block, frames, t, s: blocks.append(block.copy()), speed=4.0)
    stream.start()
    assert stream.active
    time.sleep(0.15)
    stream.stop()
    assert not stream.active
    total = sum(len(b) for b in blocks)
    assert 4 * 1024 <= total <= 0.3 * 4 * 48000  # 0.15초 × 4배속 ≈ 28800 프레임 (넉넉하게)
    assert blocks[0].shape == (1024, 2) and abs(blocks[0][0, 0] - 0.1) < 1e-3  # WAV 16-bit 양자화


def test_recorder_with_demo_device(tmp_path):
    path = tmp_path / "consult.wav"
    sf.write(path, np.full((44100, 1), 0.2, np.float32), 44100)
    rec = Recorder()
    rec.start(demo_device(path, speed=20.0))
    assert rec.active and rec.samplerate == 44100
    deadline = time.monotonic() + 3
    while rec.elapsed < 1.0 and time.monotonic() < deadline:
        rec.drain()
        time.sleep(0.01)
    audio, rate = rec.stop()
    assert not rec.active and rate == 44100
    assert audio.shape[0] >= 44100 and abs(audio[0, 0] - 0.2) < 1e-3  # WAV 16-bit 양자화
