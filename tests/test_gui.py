"""오프스크린 Qt 화면 테스트. 마이크는 가짜 입력 스트림으로 대신한다(권한 대화상자 없음)."""

import json
import os
import sys
import threading
import time

import numpy as np
import pytest
import soxr

from clinic_scribe import mic_permission, recorder
from clinic_scribe.audio import load_audio_file, to_mono, to_whisper_pcm
from clinic_scribe.config import Settings
from tests.helpers import make_fake_cli

pytestmark = [pytest.mark.gui, pytest.mark.models]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# 느린 컴퓨터(GPU 없는 CI 등)에서는 가짜 마이크를 실제 속도로 흘리고 기다리는 시간을 늘린다:
#   CLINIC_SCRIBE_TEST_SPEED   가짜 마이크 배속 (기본 4)
#   CLINIC_SCRIBE_TEST_PATIENCE 기다리는 시간 배수 (기본 1)
SPEED = float(os.environ.get("CLINIC_SCRIBE_TEST_SPEED", "4"))
PATIENCE = float(os.environ.get("CLINIC_SCRIBE_TEST_PATIENCE", "1"))


def wait_until(qapp, condition, timeout=120.0):
    deadline = time.monotonic() + timeout * PATIENCE
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


FAKE_DEVICE = recorder.InputDevice(index=7, name="가짜 내장 마이크", channels=1, default_samplerate=48000, is_default=True)


class FakeInputStream:
    """sounddevice.InputStream 대역: 지정한 소리를 SPEED 배속(기본 4)으로 콜백에 흘려 준다."""

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
            time.sleep(block / self.samplerate / SPEED)
        while self.active:  # 소스가 끝나면 조용한 입력을 계속 보낸다
            self.callback(np.full((block, self.channels), 1e-4, np.float32), block, None, None)
            time.sleep(block / self.samplerate / SPEED)

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
    monkeypatch.setattr(mic_permission, "lid_closed", lambda: False)  # 이 Mac 의 실제 덮개 상태와 무관하게
    monkeypatch.setattr(recorder.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(recorder.sd, "check_input_settings", lambda **kw: None)
    s = Settings()
    s.auto_summarize = False
    s.view_mode = "developer"  # 아래 테스트 대부분은 개발자 대시보드의 위젯을 본다
    s.summary_mode = "chart"
    s.claude_cli = make_fake_cli(tmp_path, "success")
    w = main_window.MainWindow(s)
    w.interactive = False
    w.show()
    yield w
    w.shutdown()
    w.deleteLater()


def press_space(window, autorepeat=False):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication

    QApplication.sendEvent(
        window, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", autorepeat)
    )


def transcript_text(session) -> str:
    return " ".join(seg.text for seg in session.runs[-1].result.segments) if session.runs else ""


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
    assert "사흘전부터" in transcript.replace(" ", "") and "[00:00." in transcript  # 띄어쓰기는 Whisper 마다 다름
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
        assert not w.demo_warning.isHidden()  # 마이크가 녹음되지 않는다는 경고
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


def test_demo_warning_follows_selected_device(qapp, monkeypatch, samples):
    from clinic_scribe.ui import main_window

    monkeypatch.setattr(main_window, "list_input_devices", lambda refresh=False: [FAKE_DEVICE])
    w = main_window.MainWindow(Settings(), demo_file=samples / "mock_consult_short.wav")
    w.interactive = False
    try:
        assert w.device_combo.currentIndex() == 0 and not w.demo_warning.isHidden()
        w.device_combo.setCurrentIndex(1)  # 마이크로 바꾸면 경고가 사라진다
        assert w._selected_device() == FAKE_DEVICE and w.demo_warning.isHidden()
    finally:
        w.shutdown()


def test_cancel_recording_releases_microphone(window, qapp):
    FakeInputStream.source = np.zeros(48000 * 5, np.float32) + 1e-3
    window.record_btn.click()
    assert window.recorder.active
    window.cancel_rec_btn.click()
    assert not window.recorder.active and window.session is None
    assert window.record_btn.text().endswith("녹음 시작")


def test_closing_while_recording_asks_first(window, qapp, monkeypatch):
    FakeInputStream.source = np.zeros(48000 * 5, np.float32) + 1e-3
    window.record_btn.click()
    assert window.recorder.active
    questions = []
    monkeypatch.setattr(window, "_confirm", lambda title, text: questions.append(text) or False)
    window.close()  # 아니오: 창과 녹음을 그대로 둔다
    assert window.isVisible() and window.recorder.active
    assert "녹음 중입니다" in questions[-1]
    monkeypatch.setattr(window, "_confirm", lambda title, text: True)
    window.close()
    assert not window.isVisible() and not window.recorder.active


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
                time.sleep(1024 / self.samplerate / SPEED)

    monkeypatch.setattr(recorder.sd, "InputStream", ZeroStream)
    window.record_btn.click()
    assert wait_until(qapp, lambda: window.error_label.isVisible(), 5)
    settings_place = {"darwin": "개인정보 보호 및 보안", "win32": "개인 정보 및 보안"}.get(sys.platform, "소리 설정")
    assert "디지털 무음" in window.error_label.text() and settings_place in window.error_label.text()
    window._on_cancel_recording()
    assert not window.recorder.active


def test_denied_permission_is_shown(window, qapp, monkeypatch):
    monkeypatch.setattr(mic_permission, "authorization_status", lambda: "denied")
    window.record_btn.click()
    assert not window.recorder.active
    assert window.error_label.text() == mic_permission.denied_message("denied")  # OS 에 맞는 안내
    if sys.platform == "darwin":
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
    from clinic_scribe.records import MAX_PAUSE_S
    from clinic_scribe.whisper import format_time

    window.vad_check.setChecked(True)
    window.open_audio_file(samples / "long_pause.wav", confirm=False)
    assert wait_until(qapp, lambda: len(window.session.runs) == 1)
    assert window.regions_table.rowCount() == 2
    # 저장하는 녹음은 긴 무음을 줄인 것이고(원본 19.3초), 전사 시각도 그 녹음 기준이다
    # (9초 무음 뒤 원본 13.47초에 시작한 "오늘은 경과를…" 이 앞으로 온다)
    r = window.session.runs[-1].result
    assert window.session.audio_trimmed and r.audio_duration < 19.3 - 5
    assert len(window.session.pcm16) / 16000 == pytest.approx(r.audio_duration, abs=0.01)
    assert r.regions[1].start - r.regions[0].end == pytest.approx(MAX_PAUSE_S, abs=0.02)
    after_pause = next(seg for seg in r.segments if "경과" in seg.text)
    assert after_pause.start < 8 and f"[{format_time(after_pause.start)}" in window.transcript_view.toPlainText()
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
    assert "사흘전부터" in window.transcript_view.toPlainText().replace(" ", "")  # 전사문은 남아 있음
    assert window.summary_btn.isEnabled() and window.summary_btn.text() == "요약 다시 시도"
    window.settings.claude_cli = make_fake_cli(tmp_path, "success")
    window.summary_btn.click()  # 다시 녹음하지 않고 같은 전사문으로 요약만 재시도
    assert wait_until(qapp, lambda: window.session.draft is not None)
    assert not window.summary_error_label.isVisible()
    assert len(window.session.runs) == 1
    assert "3일 전부터 식후 속쓰림" in window.draft_view.toPlainText()


def test_general_conversation_summary_and_switching_kind(window, qapp, samples, tmp_path):
    """진료와 관계없는 대화도 '일반 대화 요약'으로 요약되고, 종류를 바꿔 같은 전사문으로 다시 요약할 수 있다."""
    record = tmp_path / "general.json"
    window.settings.auto_summarize = True
    window.settings.claude_cli = make_fake_cli(tmp_path, "general", record)
    window.summary_mode_combo.setCurrentIndex(window.summary_mode_combo.findData("general"))
    assert window.draft_title.text() == "주요 내용 요약" and window.settings.summary_mode == "general"
    window.open_audio_file(samples / "daily_chat.wav", confirm=False)
    assert wait_until(qapp, lambda: window.session.draft is not None)
    assert "한강에서 자전거" in window.draft_view.toPlainText() and "주요 내용" in window.draft_view.toPlainText()
    assert "주요 내용 요약" in window.draft_basis_label.text()
    sent = json.loads(record.read_text(encoding="utf-8"))
    assert "one_line" in sent["argv"][sent["argv"].index("--json-schema") + 1]  # 주요 내용 요약 스키마로 요청
    assert "치킨" in sent["stdin"]  # 일상 대화 전사문이 그대로 전달됨
    # 진료 차트로 바꾸면 안내가 뜨고, 다시 녹음하지 않고 같은 전사문으로 다시 요약한다.
    window.settings.claude_cli = make_fake_cli(tmp_path, "success")
    window.summary_mode_combo.setCurrentIndex(window.summary_mode_combo.findData("chart"))
    assert "요약 종류를 바꿨습니다" in window.draft_basis_label.text()
    window.summary_btn.click()
    assert wait_until(qapp, lambda: window.session.draft.kind_label == "차트 초안")
    assert len(window.session.runs) == 1 and "주호소·현병력" in window.draft_view.toPlainText()


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


def test_each_session_keeps_its_own_results(window, qapp, samples):
    window.open_audio_file(samples / "injection.wav", confirm=False)
    window.open_audio_file(samples / "mock_consult_short.wav", confirm=False)  # 앞 진료 처리 중에 새 진료
    assert wait_until(qapp, lambda: len(window.sessions) == 2 and all(s.runs for s in window.sessions), 60)
    first, second = window.sessions
    assert window.session is second and second.source.endswith("mock_consult_short.wav")
    assert len(first.runs) == 1 and len(second.runs) == 1
    assert "기침" in transcript_text(first) and "기침" not in transcript_text(second)  # 서로 섞이지 않는다
    assert "기침" not in window.transcript_view.toPlainText()  # 화면에는 보고 있는 진료만


def test_simple_view_space_bar_and_hidden_results(qapp, monkeypatch, samples, tmp_path):
    """기본 화면: 스페이스바로 시작·종료, 대화 내용·요약은 눌러야 보인다."""
    from clinic_scribe.ui import main_window

    monkeypatch.setattr(main_window, "list_input_devices", lambda refresh=False: [FAKE_DEVICE])
    monkeypatch.setattr(mic_permission, "authorization_status", lambda: "authorized")
    monkeypatch.setattr(recorder.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(recorder.sd, "check_input_settings", lambda **kw: None)
    s = Settings()
    s.claude_cli = make_fake_cli(tmp_path, "success")
    w = main_window.MainWindow(s)
    w.interactive = False
    w.show()
    try:
        assert w.stack.currentWidget() is w.simple
        assert w.simple.transcript_view.isHidden() and w.simple.summary_view.isHidden()
        FakeInputStream.source = source_audio(samples / "mock_consult_short.wav")
        press_space(w)
        assert w.recorder.active and "녹음 종료" in w.simple.record_btn.text()
        press_space(w, autorepeat=True)  # 길게 누른 반복 입력은 무시
        assert w.recorder.active
        assert wait_until(qapp, lambda: w.recorder.elapsed > 10.5, 10)
        assert w.simple.elapsed.text() == "00:10"
        press_space(w)
        assert not w.recorder.active
        assert wait_until(qapp, lambda: w.session.draft is not None)
        assert w.simple.transcript_view.isHidden() and w.simple.summary_view.isHidden()  # 끝나도 숨긴 채
        assert "준비됨" in w.simple.status.text()
        rows = w.simple.history_items()
        assert len(rows) == 1 and "요약 준비됨" in rows[0].text()
        w.simple.show_summary_btn.click()
        assert not w.simple.summary_view.isHidden() and "주요 내용" in w.simple.summary_view.toPlainText()
        w.simple.show_transcript_btn.click()
        assert "사흘" in w.simple.transcript_view.toPlainText()
        w.simple.copy_btn.click()
        assert qapp.clipboard().text().startswith("[자동 요약]")
        w.simple.developer_btn.click()
        assert w.stack.currentWidget() is w.dev_page and w.settings.view_mode == "developer"
    finally:
        w.shutdown()


def test_simple_view_rereads_devices_when_list_opens(window, qapp, monkeypatch):
    from clinic_scribe.ui import main_window

    usb = recorder.InputDevice(index=9, name="USB 마이크", channels=1, default_samplerate=44100, is_default=False)
    calls = []

    def fake_list(refresh=False):
        calls.append(refresh)
        return [FAKE_DEVICE, usb]

    monkeypatch.setattr(main_window, "list_input_devices", fake_list)
    combo = window.simple.device_combo
    combo.showPopup()
    combo.hidePopup()
    assert calls == [True]  # 목록을 펼칠 때 앱을 켠 뒤 꽂은 장치까지 다시 읽는다
    assert any("USB 마이크" in combo.itemText(i) for i in range(combo.count()))
    assert combo.currentText().startswith(FAKE_DEVICE.name)  # 고른 장치는 그대로
    FakeInputStream.source = np.zeros(48000 * 3, np.float32) + 1e-3
    window.record_btn.click()
    combo.showPopup()
    combo.hidePopup()
    assert calls == [True]  # 녹음 중에는 장치를 다시 읽지 않는다
    window._on_cancel_recording()


def test_simple_view_draws_every_state(qapp, capfd):
    from clinic_scribe.ui.simple_view import HistoryRow, SimpleView

    view = SimpleView()
    view.resize(1100, 900)
    view.show()
    view.set_history(
        [
            HistoryRow(3, "10:12 ~", "녹음 중", "recording", day="오늘 · 11월 12일 (목)"),
            HistoryRow(2, "10:02–10:11 · 09:05", "요약 준비됨", "done", "다음 환자 신호 '들어오세요'", "오늘 · 11월 12일 (목)"),
            HistoryRow(1234, "17:40–17:52 · 12:01", "전사 실패", "error", day="어제 · 11월 11일 (수)"),
        ],
        2,
    )
    assert view.history_items()[1].text() == "진료 #2 · 10:02–10:11 · 09:05 · 요약 준비됨 · 다음 환자 신호 '들어오세요'"
    assert view.history.count() == 5  # 날짜 제목 줄 2개 + 기록 3줄
    for recording in (False, True):
        view.set_recording(recording, True)
        view.level.set_level(-20.0)
        view.set_status("● 녹음 중", "recording")
        view.set_error("요약 실패: 사용량 제한", warn=True)
        view.show_summary_btn.setEnabled(True)
        view.show_summary_btn.setChecked(True)
        assert not view.grab().isNull()
    view.set_recording(False, False)  # 권한 요청 중 등 누를 수 없는 상태
    view.level.set_level(-120.0)
    assert not view.grab().isNull()
    view.close()
    assert "Traceback" not in capfd.readouterr().err  # 직접 그리는 버튼·막대·기록 줄에서 예외가 없었다


def test_closed_lid_switches_away_from_builtin_mic(window, qapp, monkeypatch):
    """맥북 덮개를 닫으면(클램쉘) 내장 마이크는 0 만 들어온다: 다른 입력이 있으면 그것을 고르고, 없으면 녹음 전에 알린다."""
    from clinic_scribe.ui import main_window

    builtin = recorder.InputDevice(index=3, name="MacBook Pro 마이크", channels=1, default_samplerate=48000, is_default=True)
    iphone = recorder.InputDevice(index=0, name="‘iPhone (2)’ 마이크", channels=1, default_samplerate=48000, is_default=False)
    monkeypatch.setattr(mic_permission, "lid_closed", lambda: True)
    monkeypatch.setattr(main_window, "list_input_devices", lambda refresh=False: [iphone, builtin])
    window.settings.input_device = ""
    window._load_devices(refresh=False)
    assert window._selected_device() == iphone
    assert "덮개" in window.error_label.text() and "iPhone" in window.error_label.text()
    assert window.settings.input_device == ""  # 자동으로 고른 장치는 저장하지 않는다 (덮개를 열면 다시 내장 마이크)

    monkeypatch.setattr(main_window, "list_input_devices", lambda refresh=False: [builtin])
    window._load_devices(refresh=False)
    assert window._selected_device() == builtin
    window.record_btn.click()
    assert not window.recorder.active  # 무음을 녹음하지 않는다
    assert "덮개" in window.error_label.text() and "외장 마이크" in window.error_label.text()


def test_space_bar_through_window_toggles_once(window, qapp):
    """실제 키 입력처럼 창(QWindow)을 거쳐 위젯으로 전달돼도, 한 번 누르면 한 번만 바뀐다."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    FakeInputStream.source = np.zeros(48000 * 3, np.float32) + 1e-3
    handle = window.windowHandle()
    assert handle is not None
    space = Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "
    qapp.sendEvent(handle, QKeyEvent(QEvent.Type.KeyPress, *space))
    assert window.recorder.active
    qapp.sendEvent(handle, QKeyEvent(QEvent.Type.KeyRelease, *space))
    assert window.recorder.active  # 손을 떼는 것으로는 바뀌지 않는다
    window._on_cancel_recording()


def test_space_bar_leaves_text_fields_and_dialogs_alone(window, qapp, monkeypatch):
    from PySide6.QtWidgets import QLineEdit, QWidget

    from clinic_scribe.ui import main_window

    FakeInputStream.source = np.zeros(48000 * 3, np.float32) + 1e-3
    editor = QLineEdit()
    monkeypatch.setattr(main_window.QApplication, "focusWidget", staticmethod(lambda: editor))
    press_space(window)
    assert not window.recorder.active  # 입력칸에 글을 쓰는 중에는 녹음하지 않는다
    monkeypatch.setattr(main_window.QApplication, "focusWidget", staticmethod(lambda: None))
    dialog = QWidget()
    monkeypatch.setattr(main_window.QApplication, "activeModalWidget", staticmethod(lambda: dialog))
    press_space(window)
    assert not window.recorder.active  # 확인 창이 떠 있으면 그 창이 키를 쓴다
    monkeypatch.setattr(main_window.QApplication, "activeModalWidget", staticmethod(lambda: None))
    press_space(window)
    assert window.recorder.active
    window._on_cancel_recording()


def heard_live(window, monkeypatch) -> list[str]:
    """실시간 전사가 받아 적은 말 (가상 대화). 신호 단어를 못 찾았을 때 무엇을 들었는지 실패 메시지에 보인다."""
    heard: list[str] = []
    original = window._check_cues
    monkeypatch.setattr(window, "_check_cues", lambda segments: heard.extend(s.text for s in segments) or original(segments))
    return heard


def test_auto_split_on_next_patient_cue(window, qapp, samples, monkeypatch):
    """스페이스바를 누르지 않아도 '들어오세요'에서 진료가 나뉘고 녹음은 이어진다."""
    window.settings.min_session_s = 5
    window.settings.split_silence_s = 3600  # 느린 컴퓨터에서 실시간 전사가 늦게 끝나도 무음으로 먼저 나누지 않게 (아래 테스트 참고)
    heard = heard_live(window, monkeypatch)
    FakeInputStream.source = source_audio(samples / "two_patients.wav")
    window.record_btn.click()
    assert wait_until(qapp, lambda: len(window.sessions) == 2, 40), heard
    assert window.recorder.active  # 녹음은 끊기지 않는다
    first, second = window.sessions
    assert "들어오세요" in first.end_reason and second.auto_continued, (first.end_reason, heard)
    assert wait_until(qapp, lambda: window.recorder.elapsed > 19.5, 20)
    window.record_btn.click()
    assert wait_until(qapp, lambda: first.runs and second.runs, 60)
    t1, t2 = transcript_text(first), transcript_text(second)
    assert "사흘" in t1 and "목이" not in t1
    assert "목이" in t2 and "사흘" not in t2
    assert len(window.simple.history_items()) == 2


def enroll_voice(window, path, name, encoder):
    """가상 음성 파일로 의사 목소리를 등록한다 (등록 창과 같은 계산)."""
    from clinic_scribe.speakers import SAMPLE_RATE, enrollment_embeddings, speech_only

    data, rate = load_audio_file(path)
    speech = speech_only(to_whisper_pcm(data, rate))
    window.voices.enroll(name, enrollment_embeddings(encoder, speech), len(speech) / SAMPLE_RATE)
    window._speaker_encoder_cache = encoder
    window._refresh_voices_button()


def test_voice_dialog_enrolls_from_file_and_removes(window, qapp, samples, speaker_encoder):
    from clinic_scribe.ui.voice_dialog import VoiceDialog

    dialog = VoiceDialog(window.voices, lambda: speaker_encoder, None, window)
    dialog.interactive = False
    assert not dialog.train_btn.isEnabled() and not dialog.enroll_btn.isEnabled()  # 입력 장치·전사·소리가 아직 없다
    dialog.load_file(samples / "mock_consult_short.wav")
    assert "초뿐" in dialog.message.text()  # 말소리가 10초보다 짧다
    dialog.load_file(samples / "voice_enroll_doctor.wav")
    dialog.name_edit.setText("김원장")
    dialog.enroll_btn.click()
    assert window.voices.names() == [] and "동의" in dialog.message.text()  # 동의 확인 전에는 등록하지 않는다
    dialog.consent.setChecked(True)
    dialog.enroll_btn.click()
    assert window.voices.names() == ["김원장"] and window.voices.path.is_file() and dialog.changed
    assert "김원장" in dialog.list.item(0).text()
    dialog.list.setCurrentRow(0)
    dialog.remove_btn.click()
    assert window.voices.names() == []
    dialog.reject()


def test_guided_reading_registers_voice_and_learns_diction(window, qapp, samples, speaker_encoder):
    """따라 읽기 학습: 문장을 읽고 쉬면 다음 문장으로 넘어가고, 끝나면 분석해 목소리와 그 의사의 딕션 사전을 저장한다."""
    from clinic_scribe.training import DEFAULT_PROMPTS
    from clinic_scribe.ui.voice_dialog import VoiceDialog

    runner, params = window._whisper_runner(), window._vad_params()

    def transcribe(pcm, prompt=None):
        result = runner.transcribe(pcm, use_vad=True, params=params, prompt=prompt)
        if prompt is None:  # 학습 전: Whisper 빌드마다 틀리는 말이 달라, '당화혈색소'를 잘못 들은 것으로 바꿔 학습 과정을 확인한다
            for seg in result.segments:
                seg.text = seg.text.replace("당화혈색소", "당하 혈색소")
        return result

    dialog = VoiceDialog(window.voices, lambda: speaker_encoder, FAKE_DEVICE, window, diction=window.diction,
                         transcribe=transcribe, base_prompt=window.settings.whisper_prompt)
    dialog.interactive = False
    panel = dialog.training
    dialog.name_edit.setText("김원장")
    dialog.train_btn.click()
    assert dialog.voice_stack.currentWidget() is not panel and "동의" in dialog.message.text()  # 동의가 먼저
    dialog.consent.setChecked(True)
    FakeInputStream.source = source_audio(samples / "training_session.wav")  # 가상 의사가 문장 12개를 차례로 읽는다
    dialog.train_btn.click()
    assert dialog.voice_stack.currentWidget() is panel and panel.prompt_label.text() == DEFAULT_PROMPTS[0]
    assert wait_until(qapp, lambda: panel._index >= 1, 20)  # 다 읽고 쉬면 저절로 다음 문장
    assert panel.prompt_label.text() == DEFAULT_PROMPTS[1]
    assert wait_until(qapp, lambda: panel.result is not None or panel.error is not None, 120)
    result = panel.result
    assert result is not None and result.correct_after > result.correct_before
    assert panel.findings.count() == len(result.findings) > 0
    panel.save_btn.click()
    assert dialog.voice_stack.currentIndex() == 0 and "저장했습니다" in dialog.message.text()
    assert window.voices.names() == ["김원장"]
    learned = window.diction.entry("김원장")
    assert any((c.wrong, c.right) == ("당하 혈색소", "당화혈색소") for c in learned.corrections)
    assert "당화혈색소" in learned.vocabulary  # 다음부터 힌트로도 쓴다
    dialog.reject()


def test_transcript_marks_registered_doctor(window, qapp, samples, speaker_encoder):
    from clinic_scribe.speakers import PATIENT

    enroll_voice(window, samples / "voice_enroll_doctor.wav", "김원장", speaker_encoder)
    assert window.simple.voices_btn.text() == "의사 목소리 1"
    window.open_audio_file(samples / "mock_consult.wav", confirm=False)
    assert wait_until(qapp, lambda: window.session is not None and window.session.runs and not window._tasks)
    # 질문과 대답이 따로 줄이 되고, 줄마다 말한 사람이 실제 역할과 맞는다 ("네." 처럼 짧은 말은 판단하지 않음)
    utterances = json.loads((samples / "manifest.json").read_text(encoding="utf-8"))["mock_consult.wav"]["utterances"]
    lines = window.session.runs[-1].result.segments
    assert len(lines) == len(utterances)
    for line, utt in zip(lines, utterances):
        expected = None if utt["end"] - utt["start"] < 1.0 else ("김원장" if utt["role"] == "doctor" else PATIENT)
        assert line.speaker == expected, utt["text"]
    assert "김원장" in window.transcript_view.toPlainText() and "환자" in window.transcript_view.toPlainText()
    assert "의사 김원장" in window.simple.history_items()[0].text()


def test_cue_splits_only_when_registered_doctor_says_it(window, qapp, samples, speaker_encoder, monkeypatch):
    """의사를 등록했으면 '들어오세요'를 말한 목소리가 등록한 의사일 때만 진료를 나눈다."""
    window.settings.min_session_s = 5
    # 느린 컴퓨터(Linux CI)에서 실시간 전사가 90초 넘게 늦으면 무음으로 먼저 나뉘어, 늦게 온 '들어오세요'는
    # 이미 나뉜 앞 진료의 말로 건너뛴다. 이 테스트는 신호 단어만 보므로 무음으로는 나누지 않게 한다
    window.settings.split_silence_s = 3600
    checked = []
    original = window._cue_from_doctor
    monkeypatch.setattr(window, "_cue_from_doctor", lambda seg: checked.append(original(seg)) or checked[-1])
    heard = heard_live(window, monkeypatch)

    # 다른 사람만 등록: 대화 속 의사(등록 안 됨)의 '들어오세요'로는 나누지 않는다
    enroll_voice(window, samples / "voice_enroll_other.wav", "박원장", speaker_encoder)
    FakeInputStream.source = source_audio(samples / "two_patients.wav")
    window.record_btn.click()
    assert wait_until(qapp, lambda: window.recorder.elapsed > 19.5, 40)
    assert wait_until(qapp, lambda: checked, 60), heard  # GPU 없는 컴퓨터는 실시간 전사가 녹음보다 늦게 신호 단어에 닿는다
    assert not any(checked) and len(window.sessions) == 1
    window._on_cancel_recording()

    # 대화 속 의사 목소리를 등록: 나눈다
    checked.clear()
    enroll_voice(window, samples / "voice_enroll_doctor.wav", "김원장", speaker_encoder)
    window.record_btn.click()
    assert wait_until(qapp, lambda: len(window.sessions) == 2, 40), heard
    assert checked[-1] is True and "들어오세요" in window.sessions[0].end_reason
    window._on_cancel_recording()


def test_correcting_transcript_learns_for_next_time(window, qapp, samples):
    """전사문에서 틀린 말을 고치면 딕션 사전에 쌓이고, 다음 진료부터 자동으로 고쳐진다."""
    from clinic_scribe.diction import COMMON, DictionBook

    window.open_audio_file(samples / "doctor_drugs.wav", confirm=False)
    assert wait_until(qapp, lambda: window.session is not None and window.session.runs and not window._tasks)
    first = window.session
    assert "트림에 부티는" in transcript_text(first)
    window._on_summary_clicked()
    assert wait_until(qapp, lambda: first.draft is not None and not window._tasks)
    assert window.simple.retry_btn.isHidden()

    assert window._add_correction("트림에 부티는", "트리메부틴은", COMMON)
    assert "트리메부틴은" in transcript_text(first) and "트림에 부티는" not in transcript_text(first)
    assert "고침: 트림에 부티는 → 트리메부틴은" in window.transcript_view.toPlainText()  # 원래 들린 말을 함께 보여 준다
    assert first.summary_stale and not window.simple.retry_btn.isHidden()
    assert window.simple.retry_btn.text() == "고친 전사문으로 다시 요약"
    saved = DictionBook.load(window.diction.path)
    assert saved.entry(COMMON).corrections[0].right == "트리메부틴은"
    assert "트리메부틴은" in saved.entry(COMMON).vocabulary  # 다음부터 힌트로도 쓴다
    assert not window._add_correction("가", "나", COMMON)  # 한 글자짜리는 저장하지 않는다

    window.open_audio_file(samples / "doctor_drugs.wav", confirm=False)  # 같은 말이 나오는 다음 진료
    assert wait_until(qapp, lambda: window.session is not first and window.session.runs and not window._tasks)
    assert "트리메부틴" in transcript_text(window.session)


def test_correction_dialog_checks_input(qapp):
    from PySide6.QtWidgets import QDialog

    from clinic_scribe.ui.diction_dialog import CorrectionDialog

    dialog = CorrectionDialog("트림에 부티는", ["김원장", "이원장"], "김원장")
    assert dialog.scope_combo.itemText(0) == "공통 (모든 발화)" and dialog.scope_combo.currentData() == "김원장"
    dialog.accept()  # 바른 말이 비어 있다
    assert dialog.result() != QDialog.DialogCode.Accepted.value and "바른 말" in dialog.message.text()
    dialog.right_edit.setText("트리메부틴은")
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted.value
    assert dialog.values() == ("트림에 부티는", "트리메부틴은", "김원장", True)


def test_voice_dialog_diction_tab_edits_dictionary(window, qapp):
    from clinic_scribe.diction import COMMON, Correction, DictionBook
    from clinic_scribe.ui.voice_dialog import VoiceDialog

    window.voices.enroll("김원장", [np.eye(8, dtype=np.float32)[1]], 20)
    dialog = VoiceDialog(window.voices, lambda: None, None, window, diction=window.diction)
    dialog.interactive = False
    editor = dialog.diction_editor
    dialog.tabs.setCurrentWidget(editor)
    assert [editor.scope_combo.itemData(i) for i in range(editor.scope_combo.count())] == [COMMON, "김원장"]
    editor.vocab_edit.setPlainText("트리메부틴, 세티리진\n트리메부틴")
    editor._add_row("현타객", "현탁액")
    editor._add_row("가", "나")  # 두 글자 미만이라 빠진다
    editor.save()
    assert window.diction.entry(COMMON).vocabulary == ["트리메부틴", "세티리진"]
    assert window.diction.entry(COMMON).corrections == [Correction("현타객", "현탁액")]
    assert "1개는 뺐습니다" in editor.message.text()
    editor.select_scope("김원장")
    editor.vocab_edit.setPlainText("레바미피드")
    dialog.accept()  # 닫을 때도 고친 사전을 저장한다
    loaded = DictionBook.load(window.diction.path)
    assert loaded.entry("김원장").vocabulary == ["레바미피드"]
    assert loaded.entry(COMMON).vocabulary == ["트리메부틴", "세티리진"]


def test_auto_split_on_long_silence(window, qapp, samples):
    window.settings.split_silence_s = 4
    FakeInputStream.source = source_audio(samples / "long_pause.wav")
    window.record_btn.click()
    assert wait_until(qapp, lambda: len(window.sessions) == 2, 40)
    first, second = window.sessions
    assert first.end_reason == "무음 4초" and window.recorder.active
    assert wait_until(qapp, lambda: window.recorder.elapsed > 19.5, 20)
    window.record_btn.click()
    assert wait_until(qapp, lambda: first.runs and second.runs, 60)
    assert "사흘" in transcript_text(first) and "경과" not in transcript_text(first)
    assert "경과" in transcript_text(second)


def test_recording_stops_itself_after_long_idle(window, qapp, samples):
    window.settings.auto_split = False
    window.settings.auto_stop_idle_s = 3
    FakeInputStream.source = source_audio(samples / "mock_consult_short.wav")
    window.record_btn.click()
    assert wait_until(qapp, lambda: not window.recorder.active, 30)  # 말이 끝나고 3초 뒤 스스로 멈춘다
    assert "녹음 자동 멈춤" in window.session.end_reason
    assert wait_until(qapp, lambda: window.session.runs, 60)
    assert "사흘" in transcript_text(window.session)


def history_days(view) -> list[str]:
    from clinic_scribe.ui.simple_view import _ROW_ROLE

    items = (view.history.item(i) for i in range(view.history.count()))
    return [item.data(_ROW_ROLE) for item in items if isinstance(item.data(_ROW_ROLE), str)]


def test_records_survive_restart_and_can_be_deleted(window, qapp, samples):
    """끝난 진료는 저장되어 앱을 다시 열어도 보이고(무음을 줄인 녹음 · 전사문 · 요약), 지우면 녹음까지 사라진다."""
    from clinic_scribe.ui import main_window

    window.settings.auto_summarize = True
    window.open_audio_file(samples / "long_pause.wav", confirm=False)
    assert wait_until(qapp, lambda: window.session.draft is not None and not window._tasks)
    first = window.session
    store = window.store
    assert window.saver.flush()
    saved = store.read_audio(first.session_id)
    assert np.array_equal(saved, first.pcm16) and len(saved) / 16000 < 19.3 - 5  # 긴 무음을 줄인 녹음
    assert sys.platform == "win32" or store.audio_path(first.session_id).stat().st_mode & 0o777 == 0o600
    questions = []
    window._confirm = lambda title, text: questions.append(text) or True
    window.close()
    assert not questions  # 다 저장되어 있으니 종료할 때 묻지 않는다

    reopened = main_window.MainWindow(window.settings)  # 앱을 다시 연다
    reopened.interactive = False
    try:
        rows = reopened.simple.history_items()
        assert len(rows) == 1 and f"진료 #{first.session_id}" in rows[0].text() and "요약 준비됨" in rows[0].text()
        assert history_days(reopened.simple)[0].startswith("오늘 · ")
        assert not reopened.sessions[0].loaded  # 목록만 먼저 읽는다
        reopened._on_history_selected(first.session_id)
        assert "사흘" in reopened.simple.transcript_view.toPlainText()
        assert "주호소·현병력" in reopened.simple.summary_view.toPlainText()
        assert "무음을 줄여 저장한 녹음 기준" in reopened.transcript_view.toPlainText()
        assert reopened.play_btn.isEnabled() and reopened.session.pcm16 is None  # 녹음은 재생할 때 읽는다
        reopened.open_audio_file(samples / "mock_consult_short.wav", confirm=False)
        assert reopened.session.session_id == first.session_id + 1  # 번호가 이어진다
        assert wait_until(qapp, lambda: reopened.session.draft is not None and not reopened._tasks)

        reopened._on_history_selected(first.session_id)
        assert reopened.simple.delete_btn.isEnabled()
        reopened.simple.delete_btn.click()  # 고른 기록 삭제 (확인 창은 자동 검증에서 건너뜀)
        assert [s.session_id for s in reopened.sessions] == [first.session_id + 1] and reopened.session is None
        assert not reopened.simple.delete_btn.isEnabled()
        assert "삭제했습니다" in reopened.simple.status.text()
        assert reopened.saver.flush()
        assert [row.id for row in store.load_index()] == [first.session_id + 1]
        assert not store.audio_path(first.session_id).exists()
    finally:
        reopened.shutdown()


def test_closing_during_transcription_keeps_audio_for_retry(window, qapp, samples):
    """전사 도중 앱을 닫아도 녹음은 남고, 다음에 열면 '전사 다시 시도'로 이어서 전사한다."""
    from clinic_scribe.ui import main_window

    window.open_audio_file(samples / "mock_consult_short.wav", confirm=False)
    session = window.session
    assert wait_until(qapp, lambda: session.stored_audio, 30)  # 받은 녹음을 전사보다 먼저 저장한다
    questions = []
    window._confirm = lambda title, text: questions.append(text) or True
    window.close()
    assert "전사·요약 중인 진료가 1건" in questions[-1]

    reopened = main_window.MainWindow(window.settings)
    reopened.interactive = False
    try:
        assert "전사 실패" in reopened.simple.history_items()[0].text()
        reopened._on_history_selected(session.session_id)
        assert "앱이 종료되었습니다" in reopened.simple.error.text()
        assert reopened.simple.retry_btn.text() == "전사 다시 시도" and not reopened.simple.retry_btn.isHidden()
        reopened.simple.retry_btn.click()
        assert wait_until(qapp, lambda: reopened.session.runs and not reopened._tasks)
        assert "사흘전부터" in transcript_text(reopened.session).replace(" ", "")
        assert reopened.session.audio_trimmed and reopened.simple.retry_btn.isHidden()
        assert reopened.saver.flush()
        assert reopened.store.load_index()[0].status == "전사 완료"
    finally:
        reopened.shutdown()


def test_recording_session_cannot_be_deleted(window, qapp):
    FakeInputStream.source = np.zeros(48000 * 5, np.float32) + 1e-3
    window.record_btn.click()
    assert window.recorder.active and not window.simple.delete_btn.isEnabled()
    window._on_delete_requested(window.session.session_id)
    assert window.recorder.active and len(window.sessions) == 1 and "지울 수 없습니다" in window.simple.error.text()


def test_day_labels():
    from datetime import date

    from clinic_scribe.ui.simple_view import day_label

    today = date(2026, 11, 12)
    assert day_label(today, today) == "오늘 · 11월 12일 (목)"
    assert day_label(date(2026, 11, 11), today) == "어제 · 11월 11일 (수)"
    assert day_label(date(2026, 11, 3), today) == "11월 3일 (화)"
    assert day_label(date(2025, 12, 30), today) == "2025년 12월 30일 (화)"


def test_last_korean_syllable_is_committed_before_a_click(window, qapp, monkeypatch):
    """한글 마지막 글자가 조합 중일 때 포커스를 받지 않는 단추·체크박스를 누르면 먼저 확정한다 ('황재웅' 이 '황재' 로 저장되던 문제)."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QInputMethodEvent
    from PySide6.QtTest import QTest

    from clinic_scribe.ui import widgets
    from clinic_scribe.ui.voice_dialog import VoiceDialog

    dialog = VoiceDialog(window.voices, window._speaker_encoder, None, window)
    dialog.interactive = False
    dialog.show()
    edit = dialog.name_edit
    edit.setText("황재")
    qapp.sendEvent(edit, QInputMethodEvent("웅", []))  # 마지막 글자 '웅' 이 아직 조합 중
    assert edit.text() == "황재"

    class MacLikeInputMethod:
        """macOS 입력기처럼 commit 하면 조합 중인 글자를 입력란에 넣는다 (조합 중인 것이 없으면 아무 일도 없음)."""

        composing, calls = "웅", []

        def commit(self):
            self.calls.append("commit")
            if self.composing:
                done = QInputMethodEvent("", [])
                done.setCommitString(self.composing)
                qapp.sendEvent(edit, done)
                self.composing = ""

        def reset(self):
            self.calls.append("reset")

    ime = MacLikeInputMethod()
    monkeypatch.setattr(widgets.QApplication, "focusWidget", staticmethod(lambda: edit))
    monkeypatch.setattr(widgets.QGuiApplication, "inputMethod", staticmethod(lambda: ime))
    QTest.mouseClick(edit, Qt.MouseButton.LeftButton)  # 입력란 안을 누르면(커서 옮기기) 그대로 둔다
    assert ime.calls == [] and edit.text() == "황재"
    QTest.mouseClick(dialog.consent, Qt.MouseButton.LeftButton)  # 입력란 밖을 누르면 누르기 직전에 확정
    assert ime.calls[0] == "commit" and edit.text() == "황재웅" and dialog.consent.isChecked()
    assert dialog._checked_name() == "황재웅"
    dialog.reject()


def test_renaming_doctor_updates_voice_diction_and_records(window, qapp):
    from clinic_scribe.config import VadParams
    from clinic_scribe.diction import DictionBook
    from clinic_scribe.session import TranscriptRun
    from clinic_scribe.speakers import VoiceBook
    from clinic_scribe.ui.voice_dialog import VoiceDialog
    from clinic_scribe.whisper import TranscriptionResult, TranscriptSegment

    window.voices.enroll("황재", [np.ones(192, np.float32)], 30.0)
    window.voices.save()
    window.diction.add_correction("황재", "트림에 부티는", "트리메부틴은")
    window.diction.save()
    session = window._create_session("파일 · 진료.wav")
    lines = [TranscriptSegment(0.0, 1.5, "어디가 불편하세요?", speaker="황재"), TranscriptSegment(2.0, 3.0, "배가요.", speaker="환자")]
    session.runs.append(TranscriptRun(session.new_run_id(), TranscriptionResult(lines, [], True, VadParams(), 3.0, 0.5, doctor="황재")))
    window._persist(session)
    assert window.saver.flush()

    dialog = VoiceDialog(window.voices, window._speaker_encoder, None, window, diction=window.diction)
    dialog.interactive = False
    dialog.list.setCurrentRow(0)
    assert dialog.rename_btn.isEnabled() and dialog.rename("황재", "황재웅")
    assert dialog.renamed == [("황재", "황재웅")] and "황재웅" in dialog.list.item(0).text()
    assert not dialog.rename("황재웅", "환자")  # '환자' 는 쓸 수 없는 이름
    dialog.reject()
    window._rename_doctor("황재", "황재웅")  # 창을 닫은 뒤 메인 창이 하는 일

    assert VoiceBook.load(window.voices.path).names() == ["황재웅"]  # 목소리 특징값은 그대로, 이름만
    saved = DictionBook.load(window.diction.path)
    assert saved.doctors() == ["황재웅"] and saved.entry("황재웅").corrections[0].right == "트리메부틴은"
    assert [seg.speaker for seg in session.runs[0].result.segments] == ["황재웅", "환자"]
    assert window.saver.flush()
    row = window.store.load_index()[-1]
    assert row.doctor == "황재웅" and '"speaker": "황재웅"' in json.dumps(window.store.load_data(row.id), ensure_ascii=False)


def test_local_model_summary_never_leaves_this_computer(window, qapp, samples, tmp_path, monkeypatch):
    """요약 방식을 로컬 모델로 고르면 전사문을 Claude 로 보내지 않고 이 컴퓨터의 모델(가짜 llama-server)이 요약한다."""
    from clinic_scribe import installer
    from tests.helpers import make_fake_llama_server

    claude_called = tmp_path / "claude-called.json"
    window.settings.claude_cli = make_fake_cli(tmp_path, "success", claude_called)
    window.settings.summary_engine, window.settings.local_model = "local", "qwen3-8b"
    server = make_fake_llama_server(tmp_path)
    model_file = tmp_path / "Qwen3-8B-Q4_K_M.gguf"
    model_file.write_bytes(b"GGUF")
    monkeypatch.setattr(installer, "llama_server_path", lambda: server)
    monkeypatch.setattr(installer, "local_model_path", lambda model: model_file)
    monkeypatch.setattr(installer, "is_valid", lambda item, path, check_hash=False: True)
    window._refresh_engine_texts()
    assert "이 컴퓨터에서만" in window.simple.note.text() and "밖으로 보내지 않습니다" in window.privacy_label.text()
    window.settings.auto_summarize = True
    window.open_audio_file(samples / "mock_consult_short.wav", confirm=False)
    assert wait_until(qapp, lambda: window.session.draft is not None and not window._tasks)
    assert window.session.draft.model == "Qwen3 8B (이 컴퓨터)"
    assert "3일 전부터 식후 속쓰림" in window.draft_view.toPlainText()
    assert not claude_called.exists()  # Claude 로는 아무것도 보내지 않았다
    llm = window._local_llm
    assert llm.running
    window.shutdown()
    assert not llm.running  # 앱을 닫으면 로컬 모델 서버도 끈다


def test_missing_local_model_explains_how_to_get_it(window, qapp, samples, monkeypatch, tmp_path):
    from clinic_scribe import installer

    monkeypatch.setattr(installer, "llama_server_path", lambda: tmp_path / "없는-llama-server")
    monkeypatch.setattr(installer, "local_model_path", lambda model: tmp_path / model.file.name)
    window.settings.summary_engine = "local"
    window.settings.auto_summarize = True
    window.open_audio_file(samples / "mock_consult_short.wav", confirm=False)
    assert wait_until(qapp, lambda: window.session.runs and not window._tasks)
    assert window.session.summary_short == "모델 없음" and "설치 · 요약 방식" in window.session.summary_note
    assert window.session.draft is None and window._local_llm is None  # 서버를 띄우지 않았다


def test_specialty_picks_reading_sentences_and_is_remembered(window, qapp, monkeypatch):
    from clinic_scribe.training import SPECIALTY_PROMPTS, build_prompts
    from clinic_scribe.ui.voice_dialog import VoiceDialog

    window.voices.enroll("황재웅", [np.ones(192, np.float32)], 30.0, "정형외과")
    dialog = VoiceDialog(window.voices, window._speaker_encoder, FAKE_DEVICE, window, diction=window.diction,
                         transcribe=lambda pcm16, prompt=None: None)
    dialog.interactive = False
    assert "정형외과" in dialog.list.item(0).text()
    dialog.name_edit.setText("황재웅")  # 이미 등록한 의사: 그 의사의 진료과를 고른다
    assert dialog.specialty_combo.currentText() == "정형외과"
    dialog.name_edit.setText("김원장")
    dialog.specialty_combo.setCurrentText("피부과")
    dialog.consent.setChecked(True)
    started = []
    monkeypatch.setattr(dialog.training, "start", lambda *args: started.append(args))
    dialog.train_btn.click()
    name, _device, prompts, specialty = started[0]
    assert (name, specialty) == ("김원장", "피부과")
    assert prompts == build_prompts([], specialty="피부과") == list(SPECIALTY_PROMPTS["피부과"])
    dialog.reject()


def test_history_patient_names_search_and_grouping(window, qapp):
    """진료 기록: 환자 이름을 적으면 줄 첫머리에 보이고, 이름·날짜·내용으로 찾고, 환자별로 묶는다 (가상 진료)."""
    from datetime import datetime, timedelta

    from clinic_scribe.config import VadParams
    from clinic_scribe.session import Session, TranscriptRun
    from clinic_scribe.ui.simple_view import _ROW_ROLE
    from clinic_scribe.whisper import TranscriptionResult, TranscriptSegment

    def add(session_id, patient, text, days_ago):
        sess = Session(session_id=session_id, source="마이크", duration_s=300,
                       created_at=datetime.now() - timedelta(days=days_ago, minutes=10 - session_id % 10))
        sess.runs = [TranscriptRun(1, TranscriptionResult([TranscriptSegment(0, 2, text)], None, True, VadParams(), 5.0, 0.1))]
        window.sessions.append(sess)
        window.set_patient(sess, patient)  # 저장도 한다

    add(101, "김민준", "명치가 아프고 검은 변을 봤어요", 1)
    add(102, "이서연", "기침이 나고 열은 없어요", 0)
    add(103, "", "목이 따끔거려요", 0)
    add(104, "김 민준", "속이 쓰려요", 0)
    assert window.saver.flush()

    def shown():
        rows = [window.simple.history.item(i).data(_ROW_ROLE) for i in range(window.simple.history.count())]
        return [row if isinstance(row, str) else row.session_id for row in rows]

    window._refresh_history()
    assert [r for r in shown() if isinstance(r, int)] == [104, 103, 102, 101]  # 최신순, 날짜별
    first = window.simple.history_items()[0].data(_ROW_ROLE)
    assert first.patient == "김 민준" and "김 민준" in first.text

    def search(text, expected):
        window.simple.search.setText(text)
        assert wait_until(qapp, lambda: [r for r in shown() if isinstance(r, int)] == expected, 5), (text, shown())

    search("김민준", [104, 101])  # 띄어쓰기 무시
    search("명치", [101])  # 대화 내용
    search("어제", [101])
    search("기침 이서연", [102])  # 낱말이 모두 맞아야
    search("#103", [103])
    search("없는말", [])
    assert "찾는 진료 기록이 없습니다" in window.simple.empty.text() and window.simple.history_tab.text() == "진료 기록  0/4"
    search("", [104, 103, 102, 101])

    window.simple.group_patient.click()
    assert window.settings.history_group == "patient"
    assert shown() == ["김 민준", 104, 101, "이서연", 102, "이름 없는 진료", 103]  # 가나다순, 같은 환자는 최신순
    assert window.simple.history_items()[0].data(_ROW_ROLE).patient == ""  # 묶음 제목이 이름이니 줄에는 날짜·시각
    window.simple.group_day.click()
    assert window.settings.history_group == "day"


def test_record_button_starts_stops_and_copies_the_summary(window, qapp, samples, monkeypatch):
    """녹음 버튼(전역 단축키): 씨차트를 보며 눌러 녹음을 시작·종료하고, 요약이 끝나면 클립보드에 복사한다 (튀어 두 번 눌린 것은 한 번)."""
    from clinic_scribe.ui import main_window

    cues = []
    monkeypatch.setattr(main_window, "play_cue", cues.append)
    FakeInputStream.source = source_audio(samples / "mock_consult_short.wav")
    window.settings.auto_summarize = True
    qapp.clipboard().setText("앞서 복사한 것")
    window._on_hotkey()
    assert window.recorder.active and window._recording_session.via_hotkey and cues == ["start"]
    assert window.overlay.kind == "recording" and window.overlay.text.startswith("녹음 중")
    window._on_hotkey()  # 0.8초 안에 또 눌림(버튼이 튐): 한 번으로 본다
    assert window.recorder.active
    assert wait_until(qapp, lambda: window.recorder.elapsed > 10.5, 10)
    window._last_hotkey = 0.0
    window._on_hotkey()
    assert not window.recorder.active and cues == ["start", "stop"]
    assert window.overlay.kind == "busy"  # 전사·요약 중
    assert qapp.clipboard().text() == "앞서 복사한 것"
    assert wait_until(qapp, lambda: window.session.draft is not None, 120)
    assert qapp.clipboard().text().startswith("[검토용 초안]")  # 요약 복사 단추와 같은 글
    assert window.overlay.kind == "done" and "Ctrl+V" in window.overlay.detail and cues[-1] == "done"
