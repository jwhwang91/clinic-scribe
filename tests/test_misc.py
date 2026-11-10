import json

import numpy as np

from clinic_scribe.config import Settings, VadParams
from clinic_scribe.export import export_session
from clinic_scribe.live_monitor import LiveActivityMonitor
from clinic_scribe.session import Session, TranscriptRun
from clinic_scribe.whisper import SpeechRegion, TranscriptionResult, TranscriptSegment


def test_settings_roundtrip_ignores_unknown_keys(tmp_path):
    path = tmp_path / "settings.json"
    s = Settings(claude_model="claude-sonnet-x", vad=VadParams(min_silence_ms=900))
    s.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["obsolete_option"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = Settings.load(path)
    assert loaded.claude_model == "claude-sonnet-x" and loaded.vad.min_silence_ms == 900


def test_settings_load_broken_file_gives_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{oops", encoding="utf-8")
    assert Settings.load(path) == Settings()


def test_live_monitor_marks_speech_start_and_silence_end():
    rate = 16000
    rng = np.random.default_rng(0)
    noise = lambda s: (rng.standard_normal(int(rate * s)) * 0.001).astype(np.float32)  # noqa: E731
    t = np.arange(int(rate * 1.0)) / rate
    speech = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = np.concatenate([noise(2.0), speech, noise(2.0), speech, noise(0.3)])
    monitor = LiveActivityMonitor(min_speech_ms=150, min_silence_ms=600)
    events = []
    for i in range(0, len(audio), 512):
        events += monitor.feed(audio[i : i + 512], rate)
    assert events == ["start", "end", "start"]  # 마지막 발화 뒤 0.3초 쉼으로는 클립을 닫지 않는다
    first = monitor.clips[0]
    assert abs(first.start - 2.0) < 0.1 and abs(first.end - 3.0) < 0.1
    assert monitor.clips[1].end is None and monitor.in_speech


def test_export_writes_markdown_and_optional_wav(tmp_path):
    session = Session(session_id=1, source="파일 · test.wav", pcm16=np.zeros(16000, np.int16))
    result = TranscriptionResult(
        [TranscriptSegment(0.5, 1.0, "네.", region=0)], [SpeechRegion(0.4, 1.1)], True, VadParams(), 1.0, 0.2
    )
    run = TranscriptRun(1, result)
    session.runs.append(run)
    written = export_session(tmp_path / "out", session, run, None, include_audio=False)
    assert [p.name for p in written] == ["out.md"]
    text = written[0].read_text(encoding="utf-8")
    assert "검토용 초안" in text and "[00:00.5 → 00:01.0] 네." in text and "00:00.4 – 00:01.1" in text
    written = export_session(tmp_path / "out2.md", session, run, None, include_audio=True)
    assert [p.name for p in written] == ["out2.md", "out2.wav"]
