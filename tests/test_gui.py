"""오프스크린 Qt 화면 테스트. 마이크는 가짜 입력 스트림으로 대신한다(권한 대화상자 없음)."""

import threading
import time

import numpy as np
import pytest
import soxr

from clinic_scribe import mic_permission, recorder
from clinic_scribe.audio import load_audio_file, to_mono
from clinic_scribe.config import Settings
from tests.helpers import make_fake_cli

pytestmark = [pytest.mark.gui, pytest.mark.models]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def wait_until(qapp, condition, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


FAKE_DEVICE = recorder.InputDevice(index=7, name="가짜 내장 마이크", channels=1, default_samplerate=48000, is_default=True)


class FakeInputStream:
    """sounddevice.InputStream 대역: 지정한 소리를 4배속으로 콜백에 흘려 준다."""

    source = np.zeros(48000, np.float32)

    def __init__(self, device, samplerate, channels, dtype, callback):
        self.callback, self.samplerate, self.channels = callback, samplerate, channels
        self.active = False
        self._thread = None

    def start(self):
        self.active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        block = 1024
        pos = 0
        while self.active and pos < len(self.source):
            chunk = self.source[pos : pos + block]
            self.callback(np.repeat(chunk[:, None], self.channels, axis=1), len(chunk), None, None)
            pos += block
            time.sleep(block / self.samplerate / 4)
        while self.active:  # 소스가 끝나면 조용한 입력을 계속 보낸다
            self.callback(np.full((block, self.channels), 1e-4, np.float32), block, None, None)
            time.sleep(block / self.samplerate / 4)

    def stop(self):
        self.active = False
        if self._thread:
            self._thread.join()

    abort = stop

    def close(self):
        pass


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    from clinic_scribe.ui import main_window

    monkeypatch.setattr(main_window, "list_input_devices", lambda refresh=False: [FAKE_DEVICE])
    monkeypatch.setattr(mic_permission, "authorization_status", lambda: "authorized")
    monkeypatch.setattr(recorder.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(recorder.sd, "check_input_settings", lambda **kw: None)
    s = Settings()
    s.auto_summarize = False
    s.claude_cli = make_fake_cli(tmp_path, "success")
    w = main_window.MainWindow(s)
    w.interactive = False
    w.show()
    yield w
    w.shutdown()
    w.deleteLater()


def source_audio(path, rate=48000):
    data, sr = load_audio_file(path)
    return soxr.resample(to_mono(data), sr, rate).astype(np.float32)


def test_record_button_flow_with_fake_microphone(window, qapp, samples):
    FakeInputStream.source = source_audio(samples / "mock_consult_short.wav")
    window.settings.auto_summarize = True
    assert window.record_btn.text().endswith("녹음 시작")
    window.record_btn.click()
    assert window.recorder.active and window.record_btn.text().endswith("종료하고 요약")
    assert "녹음 중" in window.status_badge.text()
    assert wait_until(qapp, lambda: window.recorder.elapsed > 10.5, 10)  # 4배속이라 약 2.6초
    assert window.elapsed_label.text() == "00:10"
    window.record_btn.click()  # 종료하고 요약
    assert not window.recorder.active  # 마이크 해제
    assert wait_until(qapp, lambda: window.session.draft is not None, 120)
    assert window.session.original_rate == 48000
    transcript = window.transcript_view.toPlainText()
    assert "사흘 전부터" in transcript and "[00:00." in transcript
    assert window.regions_table.rowCount() == 2
    assert "검토용 초안" in window.draft_view.toPlainText()
    window.copy_btn.click()
    assert qapp.clipboard().text().startswith("[검토용 초안]")
    assert [window.pipeline.states[k] for k in ("record", "transcribe", "summary", "review")] == [
        "done", "done", "done", "active",
    ]


def test_live_preview_shows_text_while_still_recording(window, qapp, samples):
    """대시보드: 녹음 버튼을 누른 상태에서 말이 끊기면 그 부분이 실시간 전사 탭에 바로 나온다."""
    FakeInputStream.source = source_audio(samples / "mock_consult_short.wav")
    window.live_check.setChecked(True)
    window.record_btn.click()
    assert window.tabs.currentWidget() is window.live_view
    assert window.pipeline.states["record"] == "active"
    assert wait_until(qapp, lambda: any(e.status == "done" and "사흘" in e.text for e in window._live_entries), 30)
    assert window.recorder.active  # 아직 녹음 중인데 전사가 먼저 보인다
    assert "사흘" in window.live_view.toPlainText()
    assert window.timeline.live and window.timeline.regions
    window.record_btn.click()
    assert wait_until(qapp, lambda: window.session.runs, 60)
    assert window.pipeline.states["record"] == "done" and window.pipeline.states["transcribe"] == "done"
    assert window.tabs.currentWidget() is window.transcript_view
    assert not window.timeline.live and len(window.timeline.regions) == 2
    assert "사흘" in window.live_view.toPlainText()  # 실시간 미리보기는 비교용으로 남는다


def test_demo_input_device_runs_without_microphone(qapp, monkeypatch, samples):
    from clinic_scribe.ui import main_window

    monkeypatch.setattr(main_window, "list_input_devices", lambda refresh=False: [])
    monkeypatch.setattr(mic_permission, "authorization_status", lambda: "denied")  # 마이크가 막혀 있어도 데모는 된다
    s = Settings()
    s.auto_summarize = False
    w = main_window.MainWindow(s, demo_file=samples / "mock_consult_short.wav", demo_speed=8.0)
    w.interactive = False
    try:
        assert w.device_combo.currentText().startswith("데모 입력")
        assert w.record_btn.isEnabled()
        w.record_btn.click()
        assert w.recorder.active and "데모 입력" in w.status_badge.text()
        assert wait_until(qapp, lambda: w.recorder.elapsed > 10.5, 10)
        w.record_btn.click()
        assert wait_until(qapp, lambda: w.session.runs, 60)
        assert "사흘" in w.transcript_view.toPlainText()
        assert w.session.source.startswith("데모 입력")
    finally:
        w.shutdown()


def test_cancel_recording_releases_microphone(window, qapp):
    FakeInputStream.source = np.zeros(48000 * 5, np.float32) + 1e-3
    window.record_btn.click()
    assert window.recorder.active
    window.cancel_rec_btn.click()
    assert not window.recorder.active and window.session is None
    assert window.record_btn.text().endswith("녹음 시작")


def test_digital_silence_warns_about_permission(window, qapp, monkeypatch):
    # 아주 작은 잡음이라도 들어오면 정상 입력으로 본다.
    FakeInputStream.source = np.full(48000 * 3, 1e-3, np.float32)
    window.record_btn.click()
    assert not wait_until(qapp, lambda: window.error_label.isVisible(), 1.5)
    window._on_cancel_recording()

    class ZeroStream(FakeInputStream):  # 마이크 권한이 거부되면 macOS 는 스트림을 열어 주되 0 만 보낸다
        def _run(self):
            while self.active:
                self.callback(np.zeros((1024, self.channels), np.float32), 1024, None, None)
                time.sleep(1024 / self.samplerate / 4)

    monkeypatch.setattr(recorder.sd, "InputStream", ZeroStream)
    window.record_btn.click()
    assert wait_until(qapp, lambda: window.error_label.isVisible(), 5)
    assert "디지털 무음" in window.error_label.text() and "개인정보 보호 및 보안" in window.error_label.text()
    window._on_cancel_recording()
    assert not window.recorder.active


def test_denied_permission_is_shown(window, qapp, monkeypatch):
    monkeypatch.setattr(mic_permission, "authorization_status", lambda: "denied")
    window.record_btn.click()
    assert not window.recorder.active
    assert "마이크 권한이 거부" in window.error_label.text()
    assert "시스템 설정 → 개인정보 보호 및 보안 → 마이크" in window.error_label.text()


def test_stream_open_failure_is_shown(window, qapp, monkeypatch):
    def broken(**kwargs):
        raise recorder.sd.PortAudioError("Error opening InputStream: Internal PortAudio error [PaErrorCode -9986]")

    monkeypatch.setattr(recorder.sd, "InputStream", broken)
    window.record_btn.click()
    assert not window.recorder.active
    assert "마이크 입력 스트림을 열지 못했습니다" in window.error_label.text()


def test_no_input_device(qapp, monkeypatch):
    from clinic_scribe.ui import main_window

    monkeypatch.setattr(main_window, "list_input_devices", lambda refresh=False: [])
    w = main_window.MainWindow(Settings())
    w.interactive = False
    assert not w.record_btn.isEnabled() and w.open_btn.isEnabled()
    assert "입력 장치가 없습니다" in w.error_label.text()
    w.shutdown()


def test_vad_comparison_on_same_recording(window, qapp, samples):
    window.vad_check.setChecked(True)
    window.open_audio_file(samples / "long_pause.wav", confirm=False)
    assert wait_until(qapp, lambda: len(window.session.runs) == 1)
    assert window.regions_table.rowCount() == 2
    assert "[00:13." in window.transcript_view.toPlainText()  # 원본 녹음 기준 시각
    window.vad_check.setChecked(False)
    window.retranscribe_btn.click()
    assert wait_until(qapp, lambda: len(window.session.runs) == 2)
    assert "VAD 켬" in window.compare_a.currentText() and "VAD 끔" in window.compare_b.currentText()
    assert "사흘" in window.compare_view_a.toPlainText() and "사흘" in window.compare_view_b.toPlainText()


def test_summary_failure_keeps_transcript_and_retry_works(window, qapp, samples, tmp_path):
    window.settings.auto_summarize = True
    window.settings.claude_cli = make_fake_cli(tmp_path, "usage_limit")
    window.open_audio_file(samples / "mock_consult_short.wav", confirm=False)
    assert wait_until(qapp, lambda: window.session.summary_error is not None)
    assert window.summary_error_label.isVisible() and "사용량 제한" in window.summary_error_label.text()
    assert "사흘 전부터" in window.transcript_view.toPlainText()  # 전사문은 남아 있음
    assert window.summary_btn.isEnabled() and window.summary_btn.text() == "요약 다시 시도"
    window.settings.claude_cli = make_fake_cli(tmp_path, "success")
    window.summary_btn.click()  # 다시 녹음하지 않고 같은 전사문으로 요약만 재시도
    assert wait_until(qapp, lambda: window.session.draft is not None)
    assert not window.summary_error_label.isVisible()
    assert len(window.session.runs) == 1
    assert "3일 전부터 식후 속쓰림" in window.draft_view.toPlainText()


def test_silence_file_is_not_summarized(window, qapp, samples, tmp_path):
    record = tmp_path / "called.json"
    window.settings.auto_summarize = True
    window.settings.claude_cli = make_fake_cli(tmp_path, "success", record)
    window.vad_check.setChecked(True)
    window.open_audio_file(samples / "silence.wav", confirm=False)
    assert wait_until(qapp, lambda: window.session.runs)
    qapp.processEvents()
    assert not record.exists()
    assert "음성이 감지되지 않아" in window.draft_view.toPlainText()


def test_new_session_ignores_previous_results(window, qapp, samples):
    window.open_audio_file(samples / "injection.wav", confirm=False)
    window.open_audio_file(samples / "mock_consult_short.wav", confirm=False)  # 앞 작업이 끝나기 전에 새 세션
    assert wait_until(qapp, lambda: window.session.runs)
    time.sleep(1.5)  # 취소된 앞 작업이 늦게 끝나도
    qapp.processEvents()
    assert window.session.source.endswith("mock_consult_short.wav")
    assert len(window.session.runs) == 1
    assert "기침" not in window.transcript_view.toPlainText()
