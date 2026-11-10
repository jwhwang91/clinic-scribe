"""메인 창: 녹음 → whisper.cpp 전사(로컬) → Claude Code CLI 차트 초안."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QColor, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from clinic_scribe import mic_permission, procutil
from clinic_scribe.audio import WHISPER_SR, AudioFileError, load_audio_file, pcm_slice, rms_dbfs, to_mono, to_whisper_pcm
from clinic_scribe.chart import ChartDraft, verify_draft
from clinic_scribe.config import DEFAULT_CLAUDE_MODEL, Settings, VadParams
from clinic_scribe.export import export_session
from clinic_scribe.live_monitor import LiveActivityMonitor
from clinic_scribe.recorder import InputDevice, Recorder, RecorderError, list_input_devices
from clinic_scribe.session import Session, TranscriptRun
from clinic_scribe.summarizer import ClaudeConfig, SummaryError, summarize
from clinic_scribe.ui.settings_dialog import SettingsDialog
from clinic_scribe.ui.tasks import Job, Task, TaskRunner
from clinic_scribe.ui.widgets import LevelMeter
from clinic_scribe.whisper import TranscriptionResult, WhisperConfig, WhisperError, WhisperRunner, format_time

STATUS_COLORS = {
    "idle": ("#e5e7eb", "#374151"),
    "recording": ("#fee2e2", "#b91c1c"),
    "busy": ("#dbeafe", "#1d4ed8"),
    "summary": ("#ede9fe", "#6d28d9"),
    "done": ("#dcfce7", "#166534"),
    "warn": ("#ffedd5", "#9a3412"),
    "error": ("#fee2e2", "#b91c1c"),
}
PRIVACY_NOTE = (
    "녹음과 전사(whisper.cpp)는 이 Mac 에서만 처리됩니다. "
    "요약을 요청하면 전사문이 Claude Code CLI 를 통해 Claude 서비스(Anthropic)로 전송됩니다. 가상 진료 데이터로만 테스트하세요."
)
DRAFT_PLACEHOLDER = "녹음을 마치거나 오디오 파일을 불러오면 이곳에 검토용 차트 초안이 표시됩니다."


def _mmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class _PermissionBridge(QObject):
    result = Signal(bool)


class MainWindow(QMainWindow):
    # 전사/요약 처리가 끝날 때마다 "transcribe" / "summary" 를 보낸다 (자동 검증용).
    activity_finished = Signal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.interactive = True  # False 면 확인 대화상자를 건너뛴다 (자동 검증용)
        self.recorder = Recorder()
        self.live = LiveActivityMonitor()
        self.tasks = TaskRunner(self)
        self.tasks.finished.connect(self._on_task_finished)
        self._permission = _PermissionBridge(self)
        self._permission.result.connect(self._on_permission_result)
        self.session: Session | None = None
        self._session_counter = 0
        self._transcribe_task: Task | None = None
        self._summary_task: Task | None = None
        self._waiting_permission = False
        self._silent_warned = False
        self._devices: list[InputDevice] = []
        self._status: tuple[str, str] = ("대기 중", "idle")

        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(50)
        self._rec_timer.timeout.connect(self._on_rec_tick)
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(500)
        self._busy_timer.timeout.connect(self._refresh_status_badge)

        self.setWindowTitle("clinic-scribe — 모의 진료 녹음 · 전사 · 검토용 차트 초안")
        self._build_ui()
        self._load_devices(refresh=False)
        self._refresh_mic_status()
        self._refresh_tool_status()
        self._show_session()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # 입력 장치 줄
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("입력 장치"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(320)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_row.addWidget(self.device_combo)
        self.refresh_devices_btn = QPushButton("장치 새로고침")
        self.refresh_devices_btn.clicked.connect(lambda: self._load_devices(refresh=True))
        device_row.addWidget(self.refresh_devices_btn)
        self.default_device_label = QLabel()
        self.default_device_label.setStyleSheet("color: gray;")
        device_row.addWidget(self.default_device_label)
        device_row.addStretch(1)
        self.mic_status_label = QLabel()
        device_row.addWidget(self.mic_status_label)
        settings_btn = QPushButton("설정…")
        settings_btn.clicked.connect(self._open_settings)
        device_row.addWidget(settings_btn)
        root.addLayout(device_row)

        # 녹음 줄
        record_row = QHBoxLayout()
        self.record_btn = QPushButton()
        self.record_btn.setMinimumSize(270, 64)
        font = self.record_btn.font()
        font.setPointSize(18)
        font.setBold(True)
        self.record_btn.setFont(font)
        self.record_btn.clicked.connect(self._on_record_clicked)
        record_row.addWidget(self.record_btn)
        self.cancel_rec_btn = QPushButton("녹음 취소")
        self.cancel_rec_btn.setToolTip("녹음을 버리고 마이크를 놓습니다.")
        self.cancel_rec_btn.clicked.connect(self._on_cancel_recording)
        record_row.addWidget(self.cancel_rec_btn)

        meter_col = QVBoxLayout()
        self.elapsed_label = QLabel("00:00")
        mono = QFont("Menlo")
        mono.setPointSize(22)
        self.elapsed_label.setFont(mono)
        meter_col.addWidget(self.elapsed_label)
        self.level_meter = LevelMeter()
        meter_col.addWidget(self.level_meter)
        record_row.addLayout(meter_col, 1)

        state_col = QVBoxLayout()
        self.status_badge = QLabel()
        self.status_badge.setMinimumWidth(260)
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_col.addWidget(self.status_badge)
        self.live_indicator = QLabel()
        self.live_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_col.addWidget(self.live_indicator)
        record_row.addLayout(state_col)

        file_col = QVBoxLayout()
        self.open_btn = QPushButton("오디오 파일 불러오기…")
        self.open_btn.setToolTip("WAV 등 녹음 파일로 같은 전사·요약 과정을 실행합니다 (마이크 없이 테스트).")
        self.open_btn.clicked.connect(self._on_open_file)
        file_col.addWidget(self.open_btn)
        self.cancel_task_btn = QPushButton("처리 취소")
        self.cancel_task_btn.setToolTip("진행 중인 전사·요약을 멈춥니다. 녹음과 전사문은 남습니다.")
        self.cancel_task_btn.clicked.connect(self._on_cancel_tasks)
        file_col.addWidget(self.cancel_task_btn)
        record_row.addLayout(file_col)
        root.addLayout(record_row)

        # VAD 옵션 줄
        vad_row = QHBoxLayout()
        v = self.settings.vad
        self.vad_check = QCheckBox("음성 구간 자동 분리 (Silero VAD)")
        self.vad_check.setChecked(self.settings.vad_enabled)
        self.vad_check.toggled.connect(self._on_vad_options_changed)
        vad_row.addWidget(self.vad_check)
        self.silence_spin = self._spin(v.min_silence_ms, 100, 5000, 50, "이보다 긴 무음에서 구간을 나눕니다. 짧은 쉼은 합쳐집니다.")
        self.pad_spin = self._spin(v.speech_pad_ms, 0, 1000, 10, "구간 앞뒤에 더하는 여유. 문장 첫소리·끝소리 잘림을 막습니다.")
        self.min_speech_spin = self._spin(v.min_speech_ms, 30, 2000, 10, "이보다 짧은 소리는 발화로 보지 않습니다. 짧은 대답이 빠지면 줄이세요.")
        for label, spin in (("무음 분리 기준", self.silence_spin), ("구간 앞뒤 여유", self.pad_spin), ("최소 발화", self.min_speech_spin)):
            vad_row.addWidget(QLabel(label))
            vad_row.addWidget(spin)
        self.retranscribe_btn = QPushButton("같은 녹음 다시 전사")
        self.retranscribe_btn.setToolTip("현재 VAD 설정으로 같은 녹음을 다시 전사합니다. 이전 결과는 '비교' 탭에서 볼 수 있습니다.")
        self.retranscribe_btn.clicked.connect(self._on_retranscribe)
        vad_row.addWidget(self.retranscribe_btn)
        vad_row.addStretch(1)
        self.live_check = QCheckBox("실시간 발화 표시 (실험)")
        self.live_check.setToolTip("녹음 중 발화 시작·무음 종료를 표시합니다(에너지 기반). 최종 구간은 녹음 후 Silero VAD 로 계산합니다.")
        self.live_check.setChecked(self.settings.live_monitor)
        self.live_check.toggled.connect(self._on_live_toggled)
        vad_row.addWidget(self.live_check)
        root.addLayout(vad_row)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error_label.hide()
        root.addWidget(self.error_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_transcript_panel())
        splitter.addWidget(self._build_draft_panel())
        splitter.setSizes([640, 640])
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.tool_status_label = QLabel()
        self.tool_status_label.setStyleSheet("color: gray;")
        self.statusBar().addWidget(self.tool_status_label, 1)
        self._on_live_toggled(self.live_check.isChecked())
        self._on_vad_options_changed()

    def _spin(self, value: int, lo: int, hi: int, step: int, tip: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setSuffix(" ms")
        spin.setValue(value)
        spin.setToolTip(tip)
        spin.valueChanged.connect(self._on_vad_options_changed)
        return spin

    def _build_transcript_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        title = QLabel("전사문")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        header.addWidget(title)
        self.source_label = QLabel()
        self.source_label.setStyleSheet("color: gray;")
        header.addWidget(self.source_label)
        header.addStretch(1)
        header.addWidget(QLabel("결과"))
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(300)
        self.run_combo.currentIndexChanged.connect(lambda _i: self._show_run(self._current_run()))
        header.addWidget(self.run_combo)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.transcript_view = QTextBrowser()
        self.tabs.addTab(self.transcript_view, "전사문")

        regions = QWidget()
        rl = QVBoxLayout(regions)
        self.regions_info = QLabel()
        self.regions_info.setWordWrap(True)
        self.regions_info.setStyleSheet("color: gray;")
        rl.addWidget(self.regions_info)
        self.regions_table = QTableWidget(0, 5)
        self.regions_table.setHorizontalHeaderLabels(["#", "시작", "종료", "길이", "전사"])
        self.regions_table.verticalHeader().setVisible(False)
        self.regions_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.regions_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.regions_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.regions_table.setWordWrap(True)
        hh = self.regions_table.horizontalHeader()
        for col in range(4):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.regions_table.doubleClicked.connect(lambda _i: self._play_selected_region())
        rl.addWidget(self.regions_table, 1)
        play_row = QHBoxLayout()
        self.play_btn = QPushButton("▶ 선택 구간 재생")
        self.play_btn.setToolTip("구간 앞뒤가 잘리지 않았는지 들어 봅니다. (행을 두 번 눌러도 재생)")
        self.play_btn.clicked.connect(self._play_selected_region)
        self.stop_play_btn = QPushButton("■ 정지")
        self.stop_play_btn.clicked.connect(self._stop_playback)
        play_row.addWidget(self.play_btn)
        play_row.addWidget(self.stop_play_btn)
        play_row.addStretch(1)
        rl.addLayout(play_row)
        self.tabs.addTab(regions, "음성 구간")

        compare = QWidget()
        cl = QHBoxLayout(compare)
        self.compare_a = QComboBox()
        self.compare_b = QComboBox()
        self.compare_view_a = QTextBrowser()
        self.compare_view_b = QTextBrowser()
        for combo, view in ((self.compare_a, self.compare_view_a), (self.compare_b, self.compare_view_b)):
            col = QVBoxLayout()
            col.addWidget(combo)
            col.addWidget(view, 1)
            combo.currentIndexChanged.connect(self._refresh_compare)
            cl.addLayout(col, 1)
        self.tabs.addTab(compare, "비교")

        live = QWidget()
        ll = QVBoxLayout(live)
        live_info = QLabel(
            "실험 모드: 녹음 중 소리 크기로 발화 시작·무음 종료를 표시합니다. 조용한 구간에서 클립이 닫혀도 "
            "녹음(진료 세션)은 '종료하고 요약'을 누를 때까지 계속됩니다. 최종 음성 구간은 녹음 후 Silero VAD 로 다시 계산합니다."
        )
        live_info.setWordWrap(True)
        live_info.setStyleSheet("color: gray;")
        ll.addWidget(live_info)
        self.live_list = QListWidget()
        ll.addWidget(self.live_list, 1)
        self.tabs.addTab(live, "실시간(실험)")
        layout.addWidget(self.tabs, 1)
        return panel

    def _build_draft_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("차트 초안 (검토용 초안)")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(title)
        privacy = QLabel(PRIVACY_NOTE)
        privacy.setWordWrap(True)
        privacy.setStyleSheet("color: #92400e; background: #fffbeb; border: 1px solid #fcd34d; padding: 4px;")
        layout.addWidget(privacy)
        self.draft_basis_label = QLabel()
        self.draft_basis_label.setStyleSheet("color: gray;")
        layout.addWidget(self.draft_basis_label)
        self.summary_error_label = QLabel()
        self.summary_error_label.setWordWrap(True)
        self.summary_error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_error_label.setStyleSheet(
            "color: #991b1b; background: #fef2f2; border: 1px solid #fca5a5; padding: 6px;"
        )
        self.summary_error_label.hide()
        layout.addWidget(self.summary_error_label)
        self.draft_view = QTextBrowser()
        layout.addWidget(self.draft_view, 1)
        buttons = QHBoxLayout()
        self.summary_btn = QPushButton("요약 요청")
        self.summary_btn.setToolTip("선택한 전사 결과로 차트 초안을 (다시) 만듭니다. 다시 녹음할 필요가 없습니다.")
        self.summary_btn.clicked.connect(self._on_summary_clicked)
        buttons.addWidget(self.summary_btn)
        self.evidence_check = QCheckBox("근거 문장 표시")
        self.evidence_check.setChecked(self.settings.show_evidence)
        self.evidence_check.toggled.connect(lambda _c: self._show_draft())
        buttons.addWidget(self.evidence_check)
        buttons.addStretch(1)
        self.copy_btn = QPushButton("복사")
        self.copy_btn.setToolTip("'검토용 초안' 표시와 함께 클립보드에 복사합니다.")
        self.copy_btn.clicked.connect(self._on_copy)
        buttons.addWidget(self.copy_btn)
        self.export_btn = QPushButton("내보내기…")
        self.export_btn.setToolTip("고른 위치에 전사문·차트 초안(.md)과 선택 시 녹음(.wav)을 저장합니다.")
        self.export_btn.clicked.connect(self._on_export)
        buttons.addWidget(self.export_btn)
        layout.addLayout(buttons)
        return panel

    # ------------------------------------------------------------ 상태 표시
    def _set_status(self, text: str, kind: str = "idle") -> None:
        self._status = (text, kind)
        self._refresh_status_badge()

    def _refresh_status_badge(self) -> None:
        busy = self._transcribe_task is not None or self._summary_task is not None
        if self.recorder.active:
            text, kind = "● 녹음 중", "recording"
        elif self._waiting_permission:
            text, kind = "마이크 권한 요청 중…", "busy"
        elif busy:
            parts = []
            if self._transcribe_task is not None:
                parts.append(f"전사 중… {self._transcribe_task.elapsed:.0f}초")
            if self._summary_task is not None:
                parts.append(f"요약 중… {self._summary_task.elapsed:.0f}초")
            text, kind = " · ".join(parts), ("busy" if self._transcribe_task is not None else "summary")
        else:
            text, kind = self._status
        bg, fg = STATUS_COLORS.get(kind, STATUS_COLORS["idle"])
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 6px; padding: 6px 10px; font-weight: bold;"
        )
        if busy and not self._busy_timer.isActive():
            self._busy_timer.start()
        elif not busy and self._busy_timer.isActive():
            self._busy_timer.stop()

    def _show_error(self, message: str, warn: bool = False) -> None:
        style = (
            "color: #9a3412; background: #fff7ed; border: 1px solid #fdba74; padding: 6px;"
            if warn
            else "color: #991b1b; background: #fef2f2; border: 1px solid #fca5a5; padding: 6px;"
        )
        self.error_label.setStyleSheet(style)
        self.error_label.setText(message)
        self.error_label.show()

    def _clear_error(self) -> None:
        self.error_label.clear()
        self.error_label.hide()

    def _refresh_controls(self) -> None:
        recording = self.recorder.active
        s = self.session
        has_audio = s is not None and s.pcm16 is not None
        run = self._current_run()
        summarizing = self._summary_task is not None
        transcribing = self._transcribe_task is not None

        self.record_btn.setText("■  종료하고 요약" if recording else "●  녹음 시작")
        self.record_btn.setStyleSheet(
            "background: #dc2626; color: white; border-radius: 8px;" if recording
            else "background: #2563eb; color: white; border-radius: 8px;"
        )
        self.record_btn.setEnabled(not self._waiting_permission and (recording or bool(self._devices)))
        self.cancel_rec_btn.setEnabled(recording)
        self.device_combo.setEnabled(not recording)
        self.refresh_devices_btn.setEnabled(not recording)
        self.open_btn.setEnabled(not recording and not self._waiting_permission)
        self.cancel_task_btn.setEnabled(transcribing or summarizing)
        self.retranscribe_btn.setEnabled(has_audio and not recording and not transcribing)
        can_summarize = run is not None and not run.result.is_empty and not summarizing and not recording
        self.summary_btn.setEnabled(can_summarize)
        self.summary_btn.setText("요약 다시 시도" if s is not None and (s.draft or s.summary_error) else "요약 요청")
        self.copy_btn.setEnabled(s is not None and s.draft is not None)
        self.export_btn.setEnabled(s is not None and s.has_results)
        self.play_btn.setEnabled(has_audio and not recording)
        self._refresh_status_badge()

    def _refresh_mic_status(self) -> None:
        status = mic_permission.authorization_status()
        label = mic_permission.STATUS_LABEL.get(status, status)
        color = {"authorized": "#166534", "denied": "#b91c1c", "restricted": "#b91c1c"}.get(status, "#6b7280")
        self.mic_status_label.setText(f"마이크 권한: {label}")
        self.mic_status_label.setStyleSheet(f"color: {color};")
        self.mic_status_label.setToolTip(
            f"권한을 받는 앱: {mic_permission.host_app_name()}\n확인 위치: {mic_permission.SETTINGS_PATH}"
        )

    def _refresh_tool_status(self) -> None:
        s = self.settings
        cli = s.whisper_cli_path()
        model = s.whisper_model_path()
        vad = s.vad_model_path()
        claude = s.claude_cli_path()

        def mark(ok: bool) -> str:
            return "✓" if ok else "✗ 없음"

        self.tool_status_label.setText(
            f"whisper-cli {mark(bool(cli))} · Whisper 모델 {model.name} {mark(model.is_file())} · "
            f"VAD {vad.name} {mark(vad.is_file())} · Claude CLI {mark(bool(claude))} ({s.claude_model})"
        )
        self.tool_status_label.setToolTip(
            f"whisper-cli: {cli}\nWhisper 모델: {model}\nVAD 모델: {vad}\nclaude: {claude}"
        )

    # ------------------------------------------------------------ 입력 장치
    def _load_devices(self, refresh: bool) -> None:
        try:
            self._devices = list_input_devices(refresh=refresh)
        except RecorderError as exc:
            self._devices = []
            self._show_error(str(exc))
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for device in self._devices:
            self.device_combo.addItem(device.label(), device.index)
        if not self._devices:
            self.device_combo.addItem("입력 장치 없음 — 오디오 파일 불러오기로 테스트하세요", -1)
        preferred = next((i for i, d in enumerate(self._devices) if d.name == self.settings.input_device), None)
        if preferred is None:
            preferred = next((i for i, d in enumerate(self._devices) if d.is_default), 0)
        self.device_combo.setCurrentIndex(preferred)
        self.device_combo.blockSignals(False)
        default = next((d for d in self._devices if d.is_default), None)
        self.default_device_label.setText(f"시스템 기본 입력: {default.name if default else '없음'}")
        if not self._devices:
            self._show_error("사용할 수 있는 입력 장치가 없습니다. 마이크를 연결한 뒤 '장치 새로고침'을 누르거나, 오디오 파일을 불러와 테스트하세요.", warn=True)
        self._refresh_controls()

    def _selected_device(self) -> InputDevice | None:
        index = self.device_combo.currentData()
        return next((d for d in self._devices if d.index == index), None)

    def _on_device_changed(self, _i: int) -> None:
        device = self._selected_device()
        if device is not None:
            self.settings.input_device = device.name

    # ---------------------------------------------------------------- 세션
    def _confirm(self, title: str, text: str) -> bool:
        if not self.interactive:
            return True
        answer = QMessageBox.question(
            self, title, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_new_session(self) -> bool:
        if self._transcribe_task is not None or self._summary_task is not None:
            if not self._confirm("새로 시작", "진행 중인 전사·요약을 취소하고 새로 시작할까요?"):
                return False
        elif self.session is not None and self.session.has_results and not self.session.exported:
            if not self._confirm("새로 시작", "현재 전사·요약 결과는 내보내지 않았습니다. 지우고 새로 시작할까요?"):
                return False
        return True

    def _new_session(self, source: str) -> Session:
        self._cancel_tasks()
        self._stop_playback()
        self._session_counter += 1
        self.session = Session(session_id=self._session_counter, source=source)
        self.live.reset()
        self.live_list.clear()
        self._clear_error()
        self._set_status("대기 중", "idle")
        self._show_session()
        return self.session

    def _cancel_tasks(self) -> None:
        for task in (self._transcribe_task, self._summary_task):
            if task is not None:
                task.cancel.set()
        self._transcribe_task = None
        self._summary_task = None

    def _current_run(self) -> TranscriptRun | None:
        if self.session is None:
            return None
        return self.session.run(self.run_combo.currentData())

    def _show_session(self) -> None:
        s = self.session
        self.source_label.setText(f"· {s.source}" if s else "")
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run in s.runs if s else []:
            self.run_combo.addItem(run.label, run.run_id)
        if s and s.runs:
            self.run_combo.setCurrentIndex(len(s.runs) - 1)
        self.run_combo.blockSignals(False)
        for combo in (self.compare_a, self.compare_b):
            combo.blockSignals(True)
            combo.clear()
            for run in s.runs if s else []:
                combo.addItem(run.label, run.run_id)
            combo.blockSignals(False)
        if s and s.runs:
            self.compare_a.setCurrentIndex(max(0, len(s.runs) - 2))
            self.compare_b.setCurrentIndex(len(s.runs) - 1)
        self._refresh_compare()
        self._show_run(self._current_run())
        self._show_draft()

    # ---------------------------------------------------------------- 녹음
    def _on_record_clicked(self) -> None:
        if self.recorder.active:
            self._stop_recording_and_process()
            return
        if not self._confirm_new_session():
            return
        device = self._selected_device()
        if device is None:
            self._show_error("입력 장치가 없습니다. 마이크를 연결하거나 오디오 파일을 불러오세요.")
            return
        status = mic_permission.authorization_status()
        self._refresh_mic_status()
        if status in ("denied", "restricted"):
            self._show_error(mic_permission.denied_message(status))
            return
        if status == "not_determined" and mic_permission.request_access(self._permission.result.emit):
            self._waiting_permission = True
            self._show_error("macOS 마이크 권한 대화상자에서 '허용'을 눌러 주세요.", warn=True)
            self._refresh_controls()
            return
        self._start_recording(device)

    @Slot(bool)
    def _on_permission_result(self, granted: bool) -> None:
        self._waiting_permission = False
        self._refresh_mic_status()
        if not granted:
            self._show_error(mic_permission.denied_message("denied"))
            self._refresh_controls()
            return
        self._clear_error()
        device = self._selected_device()
        if device is not None:
            self._start_recording(device)
        else:
            self._refresh_controls()

    def _start_recording(self, device: InputDevice) -> None:
        self._new_session(source=f"마이크 · {device.name}")
        try:
            self.recorder.start(device)
        except RecorderError as exc:
            self._show_error(f"{exc}\n장치를 다시 고르거나 '장치 새로고침' 후 다시 시도하세요.")
            self._set_status("녹음 시작 실패", "error")
            self._refresh_controls()
            return
        self._silent_warned = False
        self.live = LiveActivityMonitor(
            min_speech_ms=self.min_speech_spin.value(), min_silence_ms=self.silence_spin.value()
        )
        self.elapsed_label.setText("00:00")
        self._rec_timer.start()
        self._refresh_controls()

    def _on_rec_tick(self) -> None:
        blocks = self.recorder.drain()
        if blocks:
            mono = to_mono(np.concatenate(blocks))
            self.level_meter.set_level(rms_dbfs(mono))
            if self.live_check.isChecked():
                changed = False
                for block in blocks:
                    if self.live.feed(to_mono(block), self.recorder.samplerate):
                        changed = True
                if changed:
                    self._refresh_live_list()
        self.elapsed_label.setText(_mmss(self.recorder.elapsed))
        if self.live_check.isChecked():
            speaking = self.live.in_speech
            self.live_indicator.setText("실시간: ● 발화 중" if speaking else "실시간: ○ 조용함")
            self.live_indicator.setStyleSheet("color: #15803d; font-weight: bold;" if speaking else "color: gray;")
        if not self._silent_warned and self.recorder.looks_digitally_silent():
            self._silent_warned = True
            self._show_error(mic_permission.silent_input_message(), warn=True)
        if self.recorder.active and not self.recorder.stream_alive():
            self._show_error("입력 장치에서 소리가 더 들어오지 않습니다(연결이 끊겼을 수 있음). 지금까지 녹음한 내용으로 전사합니다.", warn=True)
            self._stop_recording_and_process()

    def _refresh_live_list(self) -> None:
        self.live_list.clear()
        for i, clip in enumerate(self.live.clips, 1):
            end = format_time(clip.end) if clip.end is not None else "진행 중…"
            self.live_list.addItem(f"클립 {i}: {format_time(clip.start)} – {end}")
        self.live_list.scrollToBottom()

    def _stop_recording_and_process(self) -> None:
        self._rec_timer.stop()
        session = self.session
        try:
            audio, samplerate = self.recorder.stop()
        except Exception as exc:
            self._show_error(f"녹음을 정상적으로 끝내지 못했습니다: {exc}")
            self._refresh_controls()
            return
        self.level_meter.set_level(-120.0)
        if self.live.clips and self.live.clips[-1].end is None:
            self.live.clips[-1].end = self.live.time
        if session is not None:
            session.live_clips = list(self.live.clips)
        self._refresh_live_list()
        if session is None or len(audio) < 0.3 * max(1, samplerate):
            self._set_status("녹음이 너무 짧아 전사하지 않았습니다.", "warn")
            self._refresh_controls()
            return
        session.original_rate, session.original_channels = samplerate, audio.shape[1]
        self._start_transcription(source_audio=(audio, samplerate), auto_summary=self.settings.auto_summarize)

    def _on_cancel_recording(self) -> None:
        if not self.recorder.active:
            return
        self._rec_timer.stop()
        self.recorder.abort()
        self.level_meter.set_level(-120.0)
        self.elapsed_label.setText("00:00")
        self.session = None
        self._show_session()
        self._set_status("녹음을 취소했습니다 (마이크 해제)", "warn")
        self._refresh_controls()

    # ------------------------------------------------------------ 파일 불러오기
    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "오디오 파일 불러오기", str(Path.cwd()),
            "오디오 파일 (*.wav *.flac *.aiff *.aif *.ogg *.mp3);;모든 파일 (*)",
        )
        if path:
            self.open_audio_file(Path(path))

    def open_audio_file(self, path: Path, confirm: bool = True) -> None:
        if self.recorder.active:
            return
        if confirm and not self._confirm_new_session():
            return
        self._new_session(source=f"파일 · {path.name}")
        self._start_transcription(source_path=path, auto_summary=self.settings.auto_summarize)

    # ---------------------------------------------------------------- 전사
    def _vad_params(self) -> VadParams:
        v = self.settings.vad
        return VadParams(
            threshold=v.threshold,
            min_speech_ms=self.min_speech_spin.value(),
            min_silence_ms=self.silence_spin.value(),
            speech_pad_ms=self.pad_spin.value(),
            max_speech_s=v.max_speech_s,
            samples_overlap_s=v.samples_overlap_s,
        )

    def _on_vad_options_changed(self, *_args) -> None:
        self.settings.vad_enabled = self.vad_check.isChecked()
        params = self._vad_params()
        self.settings.vad = params
        for spin in (self.silence_spin, self.pad_spin, self.min_speech_spin):
            spin.setEnabled(self.vad_check.isChecked())

    def _whisper_runner(self) -> WhisperRunner:
        s = self.settings
        return WhisperRunner(
            WhisperConfig(
                cli=s.whisper_cli_path(),
                model=s.whisper_model_path(),
                vad_model=s.vad_model_path(),
                language=s.language,
                prompt=s.whisper_prompt,
                threads=s.whisper_threads,
            )
        )

    def _start_transcription(self, *, source_audio=None, source_path: Path | None = None, auto_summary: bool) -> None:
        session = self.session
        if session is None:
            return
        use_vad = self.vad_check.isChecked()
        params = self._vad_params()
        runner = self._whisper_runner()
        job = Job("transcribe", session.session_id, session.new_run_id(), auto_summary)
        pending = {"audio": source_audio}

        def work(cancel) -> TranscriptionResult:
            if session.pcm16 is None:
                if source_path is not None:
                    data, rate = load_audio_file(source_path)
                    session.original_rate, session.original_channels = rate, data.shape[1]
                else:
                    data, rate = pending.pop("audio")
                session.pcm16 = to_whisper_pcm(data, rate)  # 원본 버퍼는 여기서 놓는다
                del data
            return runner.transcribe(session.pcm16, use_vad=use_vad, params=params, cancel=cancel)

        self._transcribe_task = self.tasks.start(job, work)
        self._clear_error()
        self._refresh_controls()

    def _on_retranscribe(self) -> None:
        if self.session is None or self.session.pcm16 is None:
            return
        self._start_transcription(auto_summary=False)

    def _on_cancel_tasks(self) -> None:
        self._cancel_tasks()
        self._set_status("처리를 취소했습니다. 전사문·녹음은 남아 있습니다.", "warn")
        self._show_draft()
        self._refresh_controls()

    @Slot(object, object, object)
    def _on_task_finished(self, task: Task, result, error) -> None:
        if task is self._transcribe_task:
            self._transcribe_task = None
        if task is self._summary_task:
            self._summary_task = None
        job = task.job
        stale = self.session is None or job.session_id != self.session.session_id or task.cancel.is_set()
        if not stale:
            if job.kind == "transcribe":
                self._on_transcribed(job, result, error)
            else:
                self._on_summarized(job, result, error)
        self._refresh_controls()
        if not stale:
            self.activity_finished.emit(job.kind)

    def _on_transcribed(self, job: Job, result: TranscriptionResult | None, error) -> None:
        session = self.session
        if error is not None:
            if isinstance(error, procutil.Cancelled):
                self._set_status("전사를 취소했습니다.", "warn")
                return
            if isinstance(error, (WhisperError, AudioFileError)):
                message = str(error)
            else:
                message = f"전사 중 예기치 못한 오류: {type(error).__name__}: {error}"
            self._show_error(message + "\n녹음은 남아 있으니 문제를 고친 뒤 '같은 녹음 다시 전사'를 누르세요."
                             if session.pcm16 is not None else message)
            self._set_status("전사 실패", "error")
            self._refresh_controls()
            return
        run = TranscriptRun(job.run_id, result)
        session.runs.append(run)
        self._show_session()
        if not job.auto_summary:
            self._set_status(f"전사 완료 · {result.elapsed:.1f}초 · {len(result.segments)}개 문장", "done")
            return
        reason = self._summary_block_reason(run)
        if reason:
            session.summary_note = reason
            self._show_draft()
            self._set_status(f"전사 완료 · {reason}", "warn")
            return
        self._start_summary(run)

    def _summary_block_reason(self, run: TranscriptRun) -> str | None:
        r = run.result
        if r.no_speech_detected:
            return "음성이 감지되지 않아 요약을 요청하지 않았습니다."
        if r.is_empty:
            return "전사된 발화가 없어 요약을 요청하지 않았습니다."
        if not self.settings.claude_cli_path():
            return "Claude Code CLI(claude)를 찾지 못해 요약하지 않았습니다. 설정에서 경로를 지정하세요."
        return None

    # ---------------------------------------------------------------- 요약
    def _on_summary_clicked(self) -> None:
        run = self._current_run()
        if run is None or run.result.is_empty:
            return
        if run.result.no_speech_detected and not self._confirm(
            "요약 요청", "VAD 가 음성을 찾지 못한 녹음입니다. 전사문이 무음에서 생긴 환각일 수 있습니다. 그래도 요약을 요청할까요?"
        ):
            return
        self._start_summary(run)

    def _start_summary(self, run: TranscriptRun) -> None:
        session = self.session
        s = self.settings
        cfg = ClaudeConfig(
            cli=s.claude_cli_path(),
            model=s.claude_model.strip() or DEFAULT_CLAUDE_MODEL,
            timeout_s=s.claude_timeout_s,
            effort=s.claude_effort,
        )
        segments = run.result.segments
        based_on = run.label

        def work(cancel) -> ChartDraft:
            structured, _meta = summarize(segments, cfg, cancel)
            try:
                draft = ChartDraft.from_structured(structured, model=cfg.model, based_on=based_on)
            except ValueError as exc:
                raise SummaryError("parse", f"Claude 응답이 차트 초안 형식과 다릅니다: {exc} 다시 시도하세요.") from exc
            verify_draft(draft, segments)
            return draft

        session.summary_error = None
        session.summary_note = None
        self._summary_task = self.tasks.start(Job("summary", session.session_id, run.run_id), work)
        self._show_draft()
        self._refresh_controls()

    def _on_summarized(self, job: Job, draft: ChartDraft | None, error) -> None:
        session = self.session
        if error is not None:
            if isinstance(error, SummaryError) and error.kind == "cancelled":
                session.summary_note = "요약을 취소했습니다."
                self._set_status("요약 취소 · 전사문은 그대로 있습니다", "warn")
            else:
                session.summary_error = (
                    str(error) if isinstance(error, SummaryError) else f"요약 중 예기치 못한 오류: {type(error).__name__}"
                )
                self._set_status("요약 실패 · 전사문은 그대로 있습니다", "warn")
        else:
            session.draft = draft
            session.draft_run_id = job.run_id
            self._set_status("완료 · 검토용 차트 초안을 확인하세요", "done")
        self._show_draft()

    def _show_draft(self) -> None:
        s = self.session
        if s is not None and s.summary_error:
            self.summary_error_label.setText(f"요약 실패: {s.summary_error}\n전사문은 그대로 있습니다. '요약 다시 시도'로 같은 전사문을 다시 요약할 수 있습니다.")
            self.summary_error_label.show()
        else:
            self.summary_error_label.hide()
        if s is not None and s.draft is not None:
            run = s.run(s.draft_run_id)
            self.draft_basis_label.setText(f"요약 기준: {run.label if run else s.draft.based_on}")
            self.draft_view.setHtml(s.draft.to_html(include_evidence=self.evidence_check.isChecked()))
        else:
            self.draft_basis_label.setText("")
            if self._summary_task is not None:
                message = "요약 중… Claude 응답을 기다리고 있습니다. 전사문은 왼쪽에서 바로 확인할 수 있습니다."
            elif s is not None and s.summary_note:
                message = s.summary_note
            else:
                message = DRAFT_PLACEHOLDER
            self.draft_view.setHtml(f'<p style="color:#6b7280;">{html.escape(message)}</p>')
        self._refresh_controls()

    def _on_copy(self) -> None:
        if self.session is None or self.session.draft is None:
            return
        QGuiApplication.clipboard().setText(self.session.draft.to_text(include_evidence=self.evidence_check.isChecked()))
        self._set_status("검토용 초안을 클립보드에 복사했습니다", "done")

    def _on_export(self) -> None:
        s = self.session
        if s is None or not s.has_results:
            return
        default = Path.home() / "Documents" / f"clinic-scribe-{s.created_at:%Y%m%d-%H%M}.md"
        path, _ = QFileDialog.getSaveFileName(self, "결과 내보내기", str(default), "Markdown (*.md)")
        if not path:
            return
        include_audio = s.pcm16 is not None and self._confirm("녹음 저장", "녹음(16 kHz WAV)도 같은 이름으로 저장할까요?")
        run = s.run(s.draft_run_id) or self._current_run()
        try:
            written = export_session(Path(path), s, run, s.draft, include_audio, self.evidence_check.isChecked())
        except OSError as exc:
            self._show_error(f"내보내기 실패: {exc}")
            return
        s.exported = True
        self._set_status("내보냄: " + ", ".join(p.name for p in written), "done")

    # ------------------------------------------------------------ 결과 표시
    def _show_run(self, run: TranscriptRun | None) -> None:
        self.transcript_view.setHtml(self._transcript_html(run))
        self._fill_regions(run)

    def _transcript_html(self, run: TranscriptRun | None) -> str:
        if run is None:
            if self._transcribe_task is not None:
                return '<p style="color:#6b7280;">전사 중…</p>'
            return '<p style="color:#6b7280;">녹음을 마치거나 오디오 파일을 불러오면 타임스탬프가 있는 전사문이 표시됩니다.</p>'
        r = run.result
        head = (f'<p style="color:#6b7280;">{html.escape(run.label)} · 녹음 {format_time(r.audio_duration)} · '
                f"처리 {r.elapsed:.1f}초 · 시각은 원본 녹음 기준</p>")
        if not r.segments:
            note = "음성이 감지되지 않았습니다." if r.no_speech_detected else "전사된 발화가 없습니다."
            return head + f'<p style="color:#6b7280;">{note}</p>'
        rows = [head]
        for seg in r.segments:
            stamp = f"[{format_time(seg.start)} → {format_time(seg.end)}]"
            text = html.escape(seg.text)
            if not seg.speech_detected:
                rows.append(
                    f'<p style="background:#ffedd5;color:#9a3412;">{stamp} ⚠ 음성이 감지되지 않은 곳의 전사(환각 의심): {text}</p>'
                )
            else:
                rows.append(f'<p><span style="color:#6b7280;font-family:Menlo;">{stamp}</span>&nbsp; {text}</p>')
        return "".join(rows)

    def _fill_regions(self, run: TranscriptRun | None) -> None:
        table = self.regions_table
        table.setRowCount(0)
        if run is None:
            self.regions_info.setText("녹음을 전사하면 음성이 있는 구간과 시작·종료 시각이 표시됩니다.")
            return
        r = run.result
        rows: list[tuple[str, float, float, str, bool]] = []
        if r.regions is None:
            self.regions_info.setText("VAD 모델 또는 whisper-vad-speech-segments 를 찾지 못해 음성 구간 대신 전사 문장을 표시합니다.")
            rows = [(str(i + 1), seg.start, seg.end, seg.text, not seg.speech_detected) for i, seg in enumerate(r.segments)]
        else:
            p = r.vad_params
            if not r.regions:
                self.regions_info.setText(f"음성이 감지되지 않았습니다 (Silero VAD · {p.describe()}).")
            elif r.vad_enabled:
                self.regions_info.setText(
                    f"Silero VAD 음성 구간 {len(r.regions)}개 · {p.describe()} · 시각은 원본 녹음 기준. "
                    f"{p.min_silence_ms}ms 보다 짧은 쉼은 한 구간으로 합쳐집니다. 구간을 재생해 첫소리·끝소리가 잘렸는지 확인하세요."
                )
            else:
                self.regions_info.setText(
                    f"이 전사는 VAD 없이 녹음 전체를 처리했습니다. 아래 구간은 같은 설정의 Silero VAD 로 찾은 음성 구간(비교용)이며, "
                    f"⚠ 줄은 음성이 없는 곳에서 나온 전사입니다."
                )
            rows = [(str(i + 1), reg.start, reg.end, r.region_text(i), False) for i, reg in enumerate(r.regions)]
            rows += [("⚠", seg.start, seg.end, seg.text, True) for seg in r.segments if seg.region is None]
            rows.sort(key=lambda row: row[1])
        warn_bg = QColor("#ffedd5")
        for label, start, end, text, suspicious in rows:
            row = table.rowCount()
            table.insertRow(row)
            values = (label, format_time(start), format_time(end), f"{end - start:.1f}초", text or "(전사 없음)")
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, (start, end))
                if suspicious:
                    item.setBackground(warn_bg)
                    item.setForeground(QColor("#9a3412"))
                table.setItem(row, col, item)
        table.resizeRowsToContents()

    def _refresh_compare(self, *_args) -> None:
        s = self.session
        for combo, view in ((self.compare_a, self.compare_view_a), (self.compare_b, self.compare_view_b)):
            run = s.run(combo.currentData()) if s else None
            view.setHtml(self._transcript_html(run) if run else '<p style="color:#6b7280;">같은 녹음을 VAD 설정을 바꿔 다시 전사하면 여기서 나란히 비교할 수 있습니다.</p>')

    def _play_selected_region(self) -> None:
        s = self.session
        items = self.regions_table.selectedItems()
        if s is None or s.pcm16 is None or not items or self.recorder.active:
            return
        start, end = items[0].data(Qt.ItemDataRole.UserRole)
        clip = pcm_slice(s.pcm16, start, end).astype(np.float32) / 32768.0
        if clip.size == 0:
            return
        try:
            sd.play(clip, WHISPER_SR)
        except Exception as exc:
            self._show_error(f"재생하지 못했습니다: {exc}")

    def _stop_playback(self) -> None:
        try:
            sd.stop()
        except Exception:
            pass

    def _on_live_toggled(self, checked: bool) -> None:
        self.settings.live_monitor = checked
        self.live_indicator.setVisible(checked)
        if checked:
            self.live_indicator.setText("실시간: ○ 대기")
            self.live_indicator.setStyleSheet("color: gray;")

    # -------------------------------------------------------------- 설정
    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            dialog.apply()
            self.settings.save()
            self._refresh_tool_status()

    # -------------------------------------------------------------- 종료
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        s = self.session
        if s is not None and s.has_results and not s.exported and not self._confirm(
            "종료", "내보내지 않은 전사·요약 결과가 있습니다. 종료하면 사라집니다. 종료할까요?"
        ):
            event.ignore()
            return
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        """마이크를 놓고, 진행 중인 외부 프로세스를 끝내고, 설정을 저장한다."""
        self._rec_timer.stop()
        if self.recorder.active:
            self.recorder.abort()
        self._cancel_tasks()
        procutil.terminate_all()
        self._stop_playback()
        self.settings.show_evidence = self.evidence_check.isChecked()
        try:
            self.settings.save()
        except OSError:
            pass
        self.session = None
