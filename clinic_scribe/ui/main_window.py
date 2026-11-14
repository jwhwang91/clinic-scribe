"""메인 창.

화면 두 개를 함께 쓴다.
- 간단 화면(기본): 큰 녹음 버튼·경과 시간·상태와 진료 기록 목록만 보인다. 대화 내용·요약은 눌러야 펼쳐진다.
- 개발자 대시보드: 입력 파형, 진행 단계, 음성 구간 타임라인, 실시간/최종 전사, 음성 구간, VAD 비교, 요약.
스페이스바로 녹음을 시작·종료한다. 녹음 버튼(전역 단축키, 기본 F13)은 씨차트 등 다른 프로그램이 앞에 있어도 녹음을 시작·종료하고,
화면 구석의 작은 표시창과 소리로 알리며, 그렇게 녹음한 진료는 요약이 끝나면 클립보드에 복사한다.
씨차트 연결을 정했으면(emr_link) 녹음할 때 씨차트에 열린 환자 이름을 진료 기록에 붙이고, 버튼으로 녹음한 진료의 요약은
같은 환자일 때만 씨차트 진료 기록 칸 끝에 넣는다 (저장은 의사가 씨차트에서).
녹음 중 긴 침묵이나 다음 환자 신호("들어오세요" 등)가 나오면
그 진료를 끝내고 녹음은 이어서 다음 진료로 넘어간다(자동 진료 나누기). 진료마다 따로 전사·요약한다.
진료 기록(무음을 줄인 녹음 · 전사문 · 요약)은 저장소(records.py)에 저장되어 앱을 다시 열어도 남고, 지울 수 있다.
"""

from __future__ import annotations

import html
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QBoxLayout,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from clinic_scribe import emr_link, installer, mic_permission, procutil, updater
from clinic_scribe.hotkey import GlobalHotkey, HotkeyError, describe as describe_key
from clinic_scribe.audio import WHISPER_SR, AudioFileError, load_audio_file, pcm_slice, rms_dbfs, to_mono, to_whisper_pcm
from clinic_scribe.auto_split import find_cue, parse_cues
from clinic_scribe.chart import ChartDraft, verify_draft
from clinic_scribe.config import DEFAULT_CLAUDE_MODEL, SUMMARY_MODES, Settings, VadParams, diction_path, records_dir, voices_path
from clinic_scribe.conversation_summary import ConversationSummary
from clinic_scribe.diction import COMMON, DictionBook, DictionError, apply_corrections, build_prompt, guess_doctor
from clinic_scribe.export import export_session
from clinic_scribe.live_monitor import LiveActivityMonitor
from clinic_scribe.live_transcriber import LiveTranscriber
from clinic_scribe.local_llm import LocalLLM
from clinic_scribe.records import (
    INTERRUPTED,
    RecordError,
    RecordSaver,
    RecordStore,
    Trimmed,
    apply_data,
    record_matches,
    record_row,
    remap_result,
    session_data,
    session_from_row,
    trim_silence,
)
from clinic_scribe.recorder import InputDevice, Recorder, RecorderError, demo_device, list_input_devices
from clinic_scribe.session import Session, TranscriptRun
from clinic_scribe.speakers import PATIENT, SpeakerEncoder, SpeakerError, VoiceBook, label_segments, label_turns, main_doctor
from clinic_scribe.summarizer import ClaudeConfig, LocalConfig, SummaryError, summarize, summarize_local
from clinic_scribe.ui import theme
from clinic_scribe.ui.diction_dialog import CorrectionDialog
from clinic_scribe.ui.settings_dialog import SettingsDialog
from clinic_scribe.ui.setup_dialog import SetupDialog
from clinic_scribe.ui.simple_view import HistoryRow, SimpleView, day_label
from clinic_scribe.ui.record_overlay import RecordOverlay, play_cue
from clinic_scribe.ui.tasks import Job, Task, TaskRunner
from clinic_scribe.ui.updates import UpdateDialog, UpdateWatcher
from clinic_scribe.ui.voice_dialog import VoiceDialog
from clinic_scribe.ui.widgets import (
    BrandMark,
    LevelMeter,
    PipelineView,
    TimelineView,
    WaveformView,
    commit_typing,
    set_update_chip,
    update_chip,
)
from clinic_scribe.whisper import (
    TranscriptionResult,
    WhisperConfig,
    WhisperError,
    WhisperRunner,
    assign_regions,
    format_time,
    merge_phrases,
)

PRIVACY_NOTES = {
    "claude": ("녹음과 전사(whisper.cpp)는 이 컴퓨터에서만 처리됩니다. "
               "요약을 요청하면 전사문이 Claude Code CLI 를 통해 Claude 서비스(Anthropic)로 전송됩니다. "
               "진료 기록(무음을 줄인 녹음·전사문·요약)은 이 컴퓨터에만 저장되고 진료 기록에서 지울 수 있습니다. "
               "가상 진료 데이터로만 테스트하세요."),
    "local": ("녹음·전사(whisper.cpp)·요약(로컬 모델)을 모두 이 컴퓨터에서 처리하고 밖으로 보내지 않습니다. "
              "진료 기록(무음을 줄인 녹음·전사문·요약)은 이 컴퓨터에만 저장되고 진료 기록에서 지울 수 있습니다. "
              "가상 진료 데이터로만 테스트하세요."),
}
FOOTNOTES = {
    "claude": "녹음·전사·진료 기록은 이 컴퓨터에만 · 요약을 만들 때 전사문이 Claude 서비스로 전송됩니다",
    "local": "녹음·전사·요약·진료 기록 모두 이 컴퓨터에서만 (로컬 모델 · 밖으로 보내지 않음)",
}
DRAFT_PLACEHOLDER = "녹음을 마치거나 오디오 파일을 불러오면 이곳에 요약이 표시됩니다."
# 간단 화면 진료 기록의 상태 칩 색. 그 밖의 짧은 사유(너무 짧음, 요약 취소 등)는 주의 색.
HISTORY_KINDS = {
    "전사 중…": "busy",
    "요약 중…": "summary",
    "요약 준비됨": "done",
    "요약 실패": "error",
    "전사 실패": "error",
    "전사 완료": "idle",
    "대기": "idle",
}
VIEW_SIZES = {"simple": (1040, 640), "developer": (1400, 900)}  # 화면을 바꿀 때 창 크기
LIVE_PAD_S = 0.3  # 실시간 미리보기 클립 앞뒤 여유
KEEP_AUDIO_SESSIONS = 5  # 녹음(16 kHz)을 메모리에 남겨 둘 최근 진료 수 (나머지는 필요할 때 저장소에서 다시 읽는다)
EMR_RECHECK_S = 20  # 진료를 시작하고 이만큼 뒤에 씨차트 환자 이름을 한 번 더 읽는다 (그 사이 환자가 바뀌었는지)


def _card(layout_type: type[QBoxLayout], margins: tuple[int, int, int, int] = (14, 12, 14, 12)) -> tuple[QFrame, QBoxLayout]:
    """둥근 카드(QFrame#card)와 그 안의 레이아웃."""
    card = QFrame()
    card.setObjectName("card")
    layout = layout_type(card)
    layout.setContentsMargins(*margins)
    return card, layout


def _name_key(name: str) -> str:
    """환자 이름 비교용: 띄어쓰기를 뺀다 ('김 민준' = '김민준')."""
    return "".join(name.split())


def _mmss(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


@dataclass
class LiveEntry:
    clip_id: int  # -1 이면 진료 구분선
    start: float
    end: float | None = None
    status: str = "listening"  # listening / pending / done / off / stopped / error / split
    text: str = ""


class _PermissionBridge(QObject):
    result = Signal(bool)


class MainWindow(QMainWindow):
    # 전사/요약 처리가 끝날 때마다 "transcribe" / "summary" 를 보낸다 (자동 검증용).
    activity_finished = Signal(str)
    _record_error = Signal(str)  # 진료 기록 저장 실패 (저장 스레드에서 보낸다)
    hotkey_pressed = Signal()  # 녹음 버튼 (Windows 는 단축키 스레드에서 보낸다)

    def __init__(self, settings: Settings, demo_file: Path | None = None, demo_speed: float = 1.0) -> None:
        super().__init__()
        self.settings = settings
        self.interactive = True  # False 면 확인 대화상자를 건너뛴다 (자동 검증용)
        # 녹음 버튼: 씨차트를 보며 누를 때 화면 구석 표시창으로 상태를 알린다 (_refresh_controls 가 맞춘다)
        self.overlay = RecordOverlay()
        self._hotkey: GlobalHotkey | None = None
        self.hotkey_issue = ""  # 단축키를 등록하지 못한 이유 (설정 창에 보인다)
        self._hotkey_session: Session | None = None  # 버튼으로 마지막에 시작·종료한 진료
        self._last_hotkey = 0.0
        self.recorder = Recorder()
        self.live = LiveActivityMonitor()
        self.live_transcriber = LiveTranscriber(self)
        self.live_transcriber.finished.connect(self._on_live_result)
        self._live_entries: list[LiveEntry] = []
        self._live_runner: WhisperRunner | None = None
        self._live_issue: str | None = None
        self.tasks = TaskRunner(self)
        self.tasks.finished.connect(self._on_task_finished)
        self.emr = TaskRunner(self)  # 씨차트 연결 (emr_link): 환자 이름 읽기 · 요약 넣기
        self.emr.finished.connect(self._on_emr_finished)
        self._emr_jobs: dict[Task, tuple[Session, bool]] = {}  # 작업 → (진료, 요약을 클립보드에도 복사했나)
        self._permission = _PermissionBridge(self)
        self._permission.result.connect(self._on_permission_result)
        # 진료 기록 저장소: 앱을 다시 열어도 기록이 남는다. 목록용 정보만 먼저 읽고 내용은 고를 때 읽는다.
        self._records_issue: str | None = None
        try:
            self.store: RecordStore | None = RecordStore(records_dir())
            stored = self.store.load_index()
        except RecordError as exc:
            self.store, stored = None, []
            self._records_issue = f"진료 기록 저장소를 열지 못해 이번 기록은 저장되지 않습니다: {exc}"
        self.saver = RecordSaver(self.store, self._record_error.emit)
        self._record_error.connect(self._on_record_error)
        self.sessions: list[Session] = [session_from_row(row) for row in stored]  # 진료 기록 (오래된 것부터)
        self.session: Session | None = None  # 화면에 보이는 진료
        self._recording_session: Session | None = None  # 지금 소리를 받는 진료
        self._recording_id = 0  # 연속 녹음 번호 (실시간 전사 결과 구분)
        self._recording_started: datetime | None = None
        self._session_counter = max((s.session_id for s in self.sessions), default=0)
        self._tasks: dict[tuple[int, str], Task] = {}  # (진료 번호, "transcribe"/"summary") → 작업
        self._waiting_permission = False
        self._silent_warned = False
        self._last_record_second = -1
        self._devices: list[InputDevice] = []
        self._loading_devices = False
        self._status: tuple[str, str] = ("대기 중", "idle")  # 스페이스바 안내는 간단 화면에 따로 있다
        self._demo_device: InputDevice | None = None
        self._demo_error: str | None = None
        # 등록한 의사 목소리 (말한 사람 표시용). 화자 인식 모델은 처음 쓸 때 연다.
        self._voices_error: str | None = None
        try:
            self.voices = VoiceBook.load(voices_path())
        except SpeakerError as exc:
            self.voices = VoiceBook(voices_path())
            self._voices_error = str(exc)
        # 의사별 딕션 사전 (용어 목록·교정)
        try:
            self.diction = DictionBook.load(diction_path())
        except DictionError as exc:
            self.diction = DictionBook(diction_path())
            self._voices_error = "\n".join(filter(None, [self._voices_error, str(exc)]))
        self._speaker_encoder_cache: SpeakerEncoder | None = None
        self._speaker_issue: str | None = None
        self._shown_session_error: str | None = None  # 고른 기록의 전사 실패 안내 (다른 진료를 고르면 지운다)
        self._local_llm: LocalLLM | None = None  # 로컬 요약 모델 서버 (처음 요약할 때 띄운다)
        if demo_file is not None:
            try:
                self._demo_device = demo_device(demo_file, demo_speed)
            except Exception as exc:
                self._demo_error = f"데모 입력 파일을 열지 못했습니다 ({demo_file}): {exc}"

        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(50)
        self._rec_timer.timeout.connect(self._on_rec_tick)
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(500)
        self._busy_timer.timeout.connect(self._refresh_status_badge)

        self.setWindowTitle("clinic-scribe — 녹음 · 전사 · 요약")
        self._build_ui()
        self._load_devices(refresh=False)
        if self._demo_error:
            self._show_error(self._demo_error)
        if self._voices_error:
            self._show_error(self._voices_error, warn=True)
        if self._records_issue:
            self._show_error(self._records_issue, warn=True)
        self._refresh_voices_button()
        self._refresh_mic_status()
        self._refresh_tool_status()
        self._refresh_engine_texts()
        self._refresh_auto_split_note()
        self._show_session()
        self._render_live()
        # 스페이스바 = 녹음 시작/종료 (어느 화면에서든)
        QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("devPage")
        central.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        central.setStyleSheet(theme.stylesheet("#devPage"))
        page = QVBoxLayout(central)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)

        # 위 막대: 앱 표시 · 입력 장치 · 권한 · 설정 · 간단히 보기
        bar = QWidget()
        bar.setObjectName("topBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        device_row = QHBoxLayout(bar)
        device_row.setContentsMargins(18, 10, 14, 10)
        device_row.setSpacing(10)
        device_row.addWidget(BrandMark())
        brand = QLabel("clinic-scribe")
        brand.setObjectName("brand")
        device_row.addWidget(brand)
        chip = QLabel("개발자 보기")
        chip.setObjectName("chip")
        device_row.addWidget(chip)
        device_row.addStretch(1)
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(280)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_row.addWidget(self.device_combo)
        self.refresh_devices_btn = QPushButton("장치 새로고침")
        self.refresh_devices_btn.setObjectName("ghost")
        self.refresh_devices_btn.clicked.connect(lambda: self._load_devices(refresh=True))
        device_row.addWidget(self.refresh_devices_btn)
        self.default_device_label = QLabel()
        self.default_device_label.setObjectName("subtle")
        device_row.addWidget(self.default_device_label)
        self.mic_status_label = QLabel()
        self.mic_status_label.setObjectName("micStatus")
        device_row.addWidget(self.mic_status_label)
        self.update_btn = update_chip()
        self.update_btn.clicked.connect(self._open_updates)
        device_row.addWidget(self.update_btn)
        self.voices_btn = QPushButton("의사 목소리")
        self.voices_btn.setObjectName("ghost")
        self.voices_btn.setToolTip("의사 목소리를 등록하면 전사문의 발화마다 의사 이름이나 '환자'가 붙습니다.")
        self.voices_btn.clicked.connect(self._open_voices)
        device_row.addWidget(self.voices_btn)
        settings_btn = QPushButton("설정")
        settings_btn.setObjectName("ghost")
        settings_btn.clicked.connect(self._open_settings)
        device_row.addWidget(settings_btn)
        simple_btn = QPushButton("간단히 보기")
        simple_btn.setObjectName("ghost")
        simple_btn.setToolTip("녹음 버튼과 상태만 보이는 기본 화면으로 돌아갑니다.")
        simple_btn.clicked.connect(lambda: self._set_view("simple"))
        device_row.addWidget(simple_btn)
        page.addWidget(bar)

        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)
        page.addWidget(body, 1)
        self.demo_warning = QLabel(
            "데모 입력이 골라져 있습니다. 녹음 버튼을 누르면 데모 오디오 파일이 들어가고 마이크 소리는 녹음되지 않습니다. "
            "내 목소리로 하려면 위 입력 장치를 'MacBook Pro 마이크' 같은 마이크로 바꾸세요."
        )
        self.demo_warning.setWordWrap(True)
        self.demo_warning.setStyleSheet(theme.callout("note"))
        self.demo_warning.hide()
        root.addWidget(self.demo_warning)

        # 녹음 카드: 버튼 · 경과 시간 · 입력 파형 · 상태 · 파일
        record_card, record_row = _card(QHBoxLayout)
        record_row.setSpacing(14)
        self.record_btn = QPushButton()
        self.record_btn.setMinimumSize(210, 64)
        self.record_btn.clicked.connect(self._on_record_clicked)
        record_row.addWidget(self.record_btn)
        self.cancel_rec_btn = QPushButton("녹음 취소")
        self.cancel_rec_btn.setToolTip("지금 녹음 중인 진료를 버리고 마이크를 놓습니다.")
        self.cancel_rec_btn.clicked.connect(self._on_cancel_recording)
        record_row.addWidget(self.cancel_rec_btn)

        meter_col = QVBoxLayout()
        meter_col.setSpacing(4)
        self.elapsed_label = QLabel("00:00")
        self.elapsed_label.setFont(theme.timer_font(32))
        self.elapsed_label.setStyleSheet("color: #f1f3f7;")
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meter_col.addWidget(self.elapsed_label)
        self.level_meter = LevelMeter()
        self.level_meter.setMinimumWidth(150)
        self.level_meter.setMaximumWidth(190)
        meter_col.addWidget(self.level_meter)
        record_row.addLayout(meter_col)

        self.waveform = WaveformView()
        record_row.addWidget(self.waveform, 1)

        state_col = QVBoxLayout()
        state_col.setSpacing(6)
        self.status_badge = QLabel()
        self.status_badge.setMinimumWidth(240)
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setWordWrap(True)
        state_col.addWidget(self.status_badge)
        self.live_indicator = QLabel()
        self.live_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_col.addWidget(self.live_indicator)
        record_row.addLayout(state_col)

        file_col = QVBoxLayout()
        file_col.setSpacing(6)
        self.open_btn = QPushButton("오디오 파일 불러오기…")
        self.open_btn.setToolTip("WAV 등 녹음 파일로 같은 전사·요약 과정을 실행합니다 (마이크 없이 테스트).")
        self.open_btn.clicked.connect(self._on_open_file)
        file_col.addWidget(self.open_btn)
        self.cancel_task_btn = QPushButton("처리 취소")
        self.cancel_task_btn.setToolTip("보고 있는 진료의 전사·요약을 멈춥니다. 녹음과 전사문은 남습니다.")
        self.cancel_task_btn.clicked.connect(self._on_cancel_tasks)
        file_col.addWidget(self.cancel_task_btn)
        record_row.addLayout(file_col)
        root.addWidget(record_card)

        # 진행 카드: 단계 + 음성 구간 타임라인
        progress_card, progress = _card(QVBoxLayout)
        progress.setSpacing(10)
        self.pipeline = PipelineView()
        progress.addWidget(self.pipeline)
        self.timeline = TimelineView()
        progress.addWidget(self.timeline)
        root.addWidget(progress_card)

        # 옵션 카드: VAD · 실시간 전사 미리보기
        options_card, vad_row = _card(QHBoxLayout, (14, 8, 14, 8))
        vad_row.setSpacing(8)
        v = self.settings.vad
        self.vad_check = QCheckBox("음성 구간 자동 분리 (Silero VAD)")
        self.vad_check.setChecked(self.settings.vad_enabled)
        self.vad_check.toggled.connect(self._on_vad_options_changed)
        vad_row.addWidget(self.vad_check)
        self.silence_spin = self._spin(v.min_silence_ms, 100, 5000, 50, "이보다 긴 무음에서 구간을 나눕니다. 짧은 쉼은 합쳐집니다.")
        self.pad_spin = self._spin(v.speech_pad_ms, 0, 1000, 10, "구간 앞뒤에 더하는 여유. 문장 첫소리·끝소리 잘림을 막습니다.")
        self.min_speech_spin = self._spin(v.min_speech_ms, 30, 2000, 10, "이보다 짧은 소리는 발화로 보지 않습니다. 짧은 대답이 빠지면 줄이세요.")
        for label, spin in (("무음 분리 기준", self.silence_spin), ("구간 앞뒤 여유", self.pad_spin), ("최소 발화", self.min_speech_spin)):
            caption = QLabel(label)
            caption.setObjectName("muted")
            vad_row.addSpacing(6)
            vad_row.addWidget(caption)
            vad_row.addWidget(spin)
        vad_row.addSpacing(6)
        self.retranscribe_btn = QPushButton("같은 녹음 다시 전사")
        self.retranscribe_btn.setToolTip("현재 VAD 설정으로 같은 녹음을 다시 전사합니다. 이전 결과는 '비교' 탭에서 볼 수 있습니다.")
        self.retranscribe_btn.clicked.connect(self._on_retranscribe)
        vad_row.addWidget(self.retranscribe_btn)
        vad_row.addStretch(1)
        self.live_check = QCheckBox("실시간 전사 미리보기")
        self.live_check.setToolTip(
            "녹음 중 말이 끊길 때마다 그 부분을 whisper.cpp 로 바로 전사해 보여 줍니다(이 컴퓨터).\n"
            "녹음이 끝나면 전체 녹음으로 다시 전사한 결과가 '전사문' 탭에 나옵니다.\n"
            "자동 진료 나누기의 신호 단어 감지도 이 실시간 전사를 씁니다."
        )
        self.live_check.setChecked(self.settings.live_preview)
        self.live_check.toggled.connect(self._on_live_toggled)
        vad_row.addWidget(self.live_check)
        root.addWidget(options_card)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error_label.hide()
        root.addWidget(self.error_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(12)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_transcript_panel())
        splitter.addWidget(self._build_draft_panel())
        splitter.setSizes([660, 660])
        root.addWidget(splitter, 1)

        # 화면 두 개: 간단 화면(기본) / 개발자 대시보드. 녹음·진료 상태는 함께 쓴다.
        self.dev_page = central
        self.simple = SimpleView()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.simple)
        self.stack.addWidget(self.dev_page)
        self.setCentralWidget(self.stack)
        self.simple.device_combo.setModel(self.device_combo.model())
        self.device_combo.currentIndexChanged.connect(self.simple.device_combo.setCurrentIndex)
        self.simple.device_combo.currentIndexChanged.connect(self.device_combo.setCurrentIndex)
        self.simple.device_combo.about_to_open.connect(self._refresh_devices_before_popup)
        self.simple.record_clicked.connect(self._on_record_clicked)
        self.simple.retry_clicked.connect(self._on_retry_clicked)
        self.simple.delete_requested.connect(self._on_delete_requested)
        self.simple.copy_clicked.connect(self._on_copy)
        self.simple.settings_clicked.connect(self._open_settings)
        self.simple.voices_clicked.connect(self._open_voices)
        self.simple.developer_clicked.connect(lambda: self._set_view("developer"))
        self.simple.history_selected.connect(self._on_history_selected)
        self.simple.update_clicked.connect(self._open_updates)
        self.simple.patient_requested.connect(self._edit_patient)
        self.simple.search_changed.connect(self._on_history_search)
        self.simple.group_changed.connect(self._on_history_group)
        self._history_query = ""
        self.simple.set_group(self.settings.history_group)
        # 자동 업데이트: 새 버전(테스트 통과)을 받아 두면 위쪽 막대에 '새 버전' 단추가 나온다
        self.updates = UpdateWatcher(self.settings, self)
        self.updates.changed.connect(self._refresh_update_chip)
        self._refresh_update_chip()
        self.overlay.clicked.connect(self._bring_to_front)
        self.hotkey_pressed.connect(self._on_hotkey)
        QApplication.instance().applicationStateChanged.connect(lambda _state: self._refresh_overlay())
        self._apply_hotkey()
        # 전사문에서 틀린 말을 골라 오른쪽 클릭 → 고쳐 쓰기 (딕션 사전에 쌓인다)
        for view in (self.transcript_view, self.simple.transcript_view):
            view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            view.customContextMenuRequested.connect(lambda pos, v=view: self._transcript_menu(v, pos))

        self.tool_status_label = QLabel()
        self.statusBar().setStyleSheet(
            "QStatusBar { background: #13161d; border-top: 1px solid #20242e; }"
            " QLabel { color: #6f7787; font-size: 12px; padding: 2px 8px; }"
        )
        self.statusBar().addWidget(self.tool_status_label, 1)
        self._on_vad_options_changed()
        self._set_view(self.settings.view_mode)

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
        panel, layout = _card(QVBoxLayout)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("전사")
        title.setObjectName("cardTitle")
        header.addWidget(title)
        self.source_label = QLabel()
        self.source_label.setObjectName("subtle")
        header.addWidget(self.source_label)
        header.addStretch(1)
        result_caption = QLabel("결과")
        result_caption.setObjectName("muted")
        header.addWidget(result_caption)
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(300)
        self.run_combo.currentIndexChanged.connect(lambda _i: self._show_run(self._current_run()))
        header.addWidget(self.run_combo)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.live_view = QTextBrowser()
        self.tabs.addTab(self.live_view, "실시간 전사")
        self.transcript_view = QTextBrowser()
        self.tabs.addTab(self.transcript_view, "전사문")
        for view in (self.live_view, self.transcript_view):
            view.document().setDefaultStyleSheet(theme.DOCUMENT_STYLE)

        regions = QWidget()
        rl = QVBoxLayout(regions)
        rl.setContentsMargins(0, 8, 0, 0)
        self.regions_info = QLabel()
        self.regions_info.setWordWrap(True)
        self.regions_info.setObjectName("subtle")
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
        layout.addWidget(self.tabs, 1)
        return panel

    def _build_draft_panel(self) -> QWidget:
        panel, layout = _card(QVBoxLayout)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.draft_title = QLabel()
        self.draft_title.setObjectName("cardTitle")
        header.addWidget(self.draft_title)
        header.addStretch(1)
        mode_caption = QLabel("요약 종류")
        mode_caption.setObjectName("muted")
        header.addWidget(mode_caption)
        self.summary_mode_combo = QComboBox()
        for mode, label in SUMMARY_MODES.items():
            self.summary_mode_combo.addItem(label, mode)
        self.summary_mode_combo.setCurrentIndex(max(0, self.summary_mode_combo.findData(self.settings.summary_mode)))
        self.summary_mode_combo.setToolTip(
            "주요 내용 요약: 한 줄 요약과 주제별 주요 내용·할 일. 진료 차트 초안: 주호소·진찰·평가/계획·확인 필요.\n"
            "바꾼 뒤 '요약 다시 시도'를 누르면 같은 전사문으로 다시 요약합니다(다시 녹음할 필요 없음)."
        )
        self.summary_mode_combo.currentIndexChanged.connect(self._on_summary_mode_changed)
        header.addWidget(self.summary_mode_combo)
        layout.addLayout(header)
        self.privacy_label = QLabel(PRIVACY_NOTES["claude"])  # 요약 방식에 따라 _refresh_engine_texts 가 바꾼다
        self.privacy_label.setWordWrap(True)
        self.privacy_label.setStyleSheet(theme.callout("note"))
        layout.addWidget(self.privacy_label)
        self.draft_basis_label = QLabel()
        self.draft_basis_label.setObjectName("subtle")
        self.draft_basis_label.setWordWrap(True)
        layout.addWidget(self.draft_basis_label)
        self.summary_error_label = QLabel()
        self.summary_error_label.setWordWrap(True)
        self.summary_error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_error_label.setStyleSheet(theme.callout("error"))
        self.summary_error_label.hide()
        layout.addWidget(self.summary_error_label)
        self.draft_view = QTextBrowser()
        self.draft_view.document().setDefaultStyleSheet(theme.DOCUMENT_STYLE)
        layout.addWidget(self.draft_view, 1)
        buttons = QHBoxLayout()
        self.summary_btn = QPushButton("요약 요청")
        self.summary_btn.setObjectName("primary")
        self.summary_btn.setToolTip("선택한 전사 결과로 요약을 (다시) 만듭니다. 다시 녹음할 필요가 없습니다.")
        self.summary_btn.clicked.connect(self._on_summary_clicked)
        buttons.addWidget(self.summary_btn)
        self.evidence_check = QCheckBox("근거 문장 표시")
        self.evidence_check.setChecked(self.settings.show_evidence)
        self.evidence_check.toggled.connect(lambda _c: self._show_draft())
        buttons.addWidget(self.evidence_check)
        buttons.addStretch(1)
        self.copy_btn = QPushButton("복사")
        self.copy_btn.setToolTip("요약을 머리말과 함께 클립보드에 복사합니다.")
        self.copy_btn.clicked.connect(self._on_copy)
        buttons.addWidget(self.copy_btn)
        self.export_btn = QPushButton("내보내기…")
        self.export_btn.setToolTip("고른 위치에 전사문·요약(.md)과 선택 시 녹음(.wav)을 저장합니다.")
        self.export_btn.clicked.connect(self._on_export)
        buttons.addWidget(self.export_btn)
        layout.addLayout(buttons)
        self._refresh_draft_title()
        return panel

    def _set_view(self, mode: str) -> None:
        self.settings.view_mode = mode if mode in ("simple", "developer") else "simple"
        simple = self.settings.view_mode == "simple"
        current = self.simple if simple else self.dev_page
        # 가려진 화면의 최소 크기가 창 크기를 붙잡지 않게 한다 (간단 화면 창을 작게 쓰기 위해).
        for page in (self.simple, self.dev_page):
            policy = QSizePolicy.Policy.Preferred if page is current else QSizePolicy.Policy.Ignored
            page.setSizePolicy(policy, policy)
        self.stack.setCurrentWidget(current)
        self.statusBar().setVisible(not simple)
        if not (self.isMaximized() or self.isFullScreen()):
            self.resize(*VIEW_SIZES[self.settings.view_mode])

    # -------------------------------------------------------- 스페이스바 토글
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(watched, QWidget):
            commit_typing(watched)  # 한글 마지막 글자가 조합 중인 채로 단추가 눌리지 않게 (모든 창)
        if (
            event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Space
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self._space_toggles_recording(watched)
        ):
            if not event.isAutoRepeat():  # 길게 눌러도 한 번만
                self._on_record_clicked()
            return True
        return super().eventFilter(watched, event)

    def _space_toggles_recording(self, watched: QObject) -> bool:
        if QApplication.activeModalWidget() is not None:  # 확인 창·설정 창이 떠 있으면 그 창이 키를 쓴다
            return False
        if not isinstance(watched, QWidget) or watched.window() is not self:
            return False
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox)):
            return False
        if isinstance(focus, (QTextEdit, QPlainTextEdit)) and not focus.isReadOnly():
            return False
        return self.recorder.active or self.record_btn.isEnabled()

    # ------------------------------------------------------------ 상태 표시
    def _set_status(self, text: str, kind: str = "idle") -> None:
        self._status = (text, kind)
        self._refresh_status_badge()

    def _task(self, session: Session | None, kind: str) -> Task | None:
        return self._tasks.get((session.session_id, kind)) if session is not None else None

    def _refresh_status_badge(self) -> None:
        s = self.session
        background = {sid for sid, _kind in self._tasks if s is None or sid != s.session_id}
        transcribing = self._task(s, "transcribe")
        summarizing = self._task(s, "summary")
        if self.recorder.active:
            rec = self._recording_session
            demo = self.recorder.device is not None and self.recorder.device.demo_file
            text = "● 녹음 중" + (" (데모 입력)" if demo else "")
            if rec is not None and len(self.sessions) > 1:
                text += f" · 진료 #{rec.session_id}"
            if background:
                text += f" · 이전 진료 처리 중 {len(background)}건"
            kind = "recording"
        elif self._waiting_permission:
            text, kind = "마이크 권한 요청 중…", "busy"
        elif transcribing or summarizing:
            parts = []
            if transcribing:
                parts.append(f"전사 중… {transcribing.elapsed:.0f}초")
            if summarizing:
                parts.append(f"요약 중… {summarizing.elapsed:.0f}초")
            text, kind = " · ".join(parts), ("busy" if transcribing else "summary")
        elif background:
            text, kind = f"이전 진료 처리 중 {len(background)}건", "busy"
        else:
            text, kind = self._status
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(theme.pill(kind, 14))
        self.simple.set_status(text, kind)
        self._refresh_pipeline()
        busy = bool(self._tasks)
        if busy and not self._busy_timer.isActive():
            self._busy_timer.start()
        elif not busy and self._busy_timer.isActive():
            self._busy_timer.stop()

    def _refresh_pipeline(self) -> None:
        """진행 단계는 보이는 진료의 상태에서 계산한다 (진료가 여러 개라 이벤트마다 설정하지 않는다)."""
        s, p = self.session, self.pipeline
        if s is None:
            p.reset()
            return
        if s is self._recording_session and self.recorder.active:
            demo = self.recorder.device is not None and self.recorder.device.demo_file
            p.set_step("record", "active", _mmss(self.recorder.elapsed - s.offset_s) + (" · 데모 입력" if demo else ""))
        elif s.source.startswith("파일"):
            p.set_step("record", "done", s.source)
        elif s.duration_s > 0:
            reason = f" · {s.end_reason}" if s.end_reason else ""
            p.set_step("record", "done", f"{_mmss(s.duration_s)} 녹음{reason}")
        else:
            p.set_step("record", "idle")
        task = self._task(s, "transcribe")
        if task is not None:
            p.set_step("transcribe", "active", f"처리 중… {task.elapsed:.0f}초")
        elif s.transcribe_error:
            p.set_step("transcribe", "error", "실패 · 위 안내 참고")
        elif s.runs:
            r = s.runs[-1].result
            regions = f" · 구간 {len(r.regions)}개" if r.regions is not None else ""
            p.set_step("transcribe", "done", f"{s.transcribe_seconds:.1f}초 · 문장 {len(r.segments)}개{regions}")
        elif s.summary_short and not s.runs:
            p.set_step("transcribe", "skipped", s.summary_short)
        else:
            p.set_step("transcribe", "idle")
        task = self._task(s, "summary")
        if task is not None:
            p.set_step("summary", "active", f"{self._engine_name()} 응답 대기… {task.elapsed:.0f}초")
        elif s.summary_error:
            p.set_step("summary", "error", "실패 · '요약 다시 시도' 가능")
        elif s.draft is not None:
            p.set_step("summary", "done", f"{s.summary_seconds:.1f}초 · {s.draft.kind_label}")
        elif s.summary_short and s.runs:
            p.set_step("summary", "skipped", s.summary_short)
        else:
            p.set_step("summary", "idle")
        if s.draft is not None:
            p.set_step("review", "active", f"{s.draft.kind_label} 확인")
        else:
            p.set_step("review", "idle")

    def _show_error(self, message: str, warn: bool = False) -> None:
        self.error_label.setStyleSheet(theme.callout("warn" if warn else "error"))
        self.error_label.setText(message)
        self.error_label.show()
        self.simple.set_error(message, warn)

    def _clear_error(self) -> None:
        self.error_label.clear()
        self.error_label.hide()
        self.simple.set_error(None)

    def _refresh_controls(self) -> None:
        recording = self.recorder.active
        s = self.session
        live_session = s is not None and s is self._recording_session and recording
        has_audio = s is not None and (s.pcm16 is not None or s.stored_audio)
        run = self._current_run()
        summarizing = self._task(s, "summary") is not None
        transcribing = self._task(s, "transcribe") is not None

        self.record_btn.setText("■  종료하고 요약" if recording else "●  녹음 시작")
        self.record_btn.setStyleSheet(theme.record_button(recording))
        self.record_btn.setEnabled(not self._waiting_permission and (recording or bool(self._devices)))
        self.cancel_rec_btn.setEnabled(recording)
        for button in (self.voices_btn, self.simple.voices_btn):
            button.setEnabled(not recording)  # 목소리 등록은 진료 녹음 중이 아닐 때
        self.device_combo.setEnabled(not recording)
        self.refresh_devices_btn.setEnabled(not recording)
        self.open_btn.setEnabled(not recording and not self._waiting_permission)
        self.cancel_task_btn.setEnabled(transcribing or summarizing)
        self.retranscribe_btn.setEnabled(has_audio and not live_session and not transcribing)
        can_summarize = run is not None and not run.result.is_empty and not summarizing and not live_session
        self.summary_btn.setEnabled(can_summarize)
        self.summary_btn.setText("요약 다시 시도" if s is not None and (s.draft or s.summary_error) else "요약 요청")
        self.copy_btn.setEnabled(s is not None and s.draft is not None)
        self.export_btn.setEnabled(s is not None and s.has_results)
        self.play_btn.setEnabled(has_audio and not recording)
        self.simple.set_recording(recording, self.record_btn.isEnabled())
        self.simple.device_combo.setEnabled(not recording)
        self.simple.set_results(
            has_transcript=run is not None and not live_session,
            has_summary=s is not None and not live_session and (s.draft is not None or bool(s.summary_note)),
            can_copy=s is not None and s.draft is not None,
            can_delete=s is not None and not live_session,
            can_name=s is not None,
        )
        self._refresh_overlay()
        stale = s is not None and s.summary_stale and s.draft is not None and not s.summary_error
        # 전사가 실패했거나 전사 도중 앱이 꺼진 기록은 저장된 녹음으로 다시 전사한다
        retranscribe = (s is not None and not s.runs and bool(s.transcribe_error) and has_audio
                        and not transcribing and not live_session)
        if retranscribe:
            self.simple.retry_btn.setText("전사 다시 시도")
        else:
            self.simple.retry_btn.setText("고친 전사문으로 다시 요약" if stale else "요약 다시 시도")
        self.simple.retry_btn.setVisible(
            retranscribe or (can_summarize and s is not None and (bool(s.summary_error) or stale))
        )
        self._refresh_history()
        self._refresh_status_badge()

    def _refresh_history(self) -> None:
        """진료 기록 목록: 최신순으로, 날짜별(기본) 또는 환자별(이름 가나다순, 이름 없는 진료는 끝)로 묶는다. 찾는 중이면 찾은 것만."""
        by_patient = self.settings.history_group == "patient"
        sessions = list(reversed(self.sessions))
        found = self._search_ids(self._history_query) if self._history_query else None
        if found is not None:
            sessions = [sess for sess in sessions if sess.session_id in found]
        groups: dict[str, str] = {}
        if by_patient:  # 띄어쓰기만 다른 이름은 같은 환자로. 묶음 제목은 가장 최근에 적은 모양
            sessions.sort(key=lambda sess: (not sess.patient, _name_key(sess.patient)))  # 같은 환자 안에서는 최신순 그대로
            for sess in sessions:
                groups.setdefault(_name_key(sess.patient), sess.patient or "이름 없는 진료")
        rows = [self._history_row(sess, groups.get(_name_key(sess.patient))) for sess in sessions]
        self.simple.set_history(rows, self.session.session_id if self.session else None,
                                total=len(self.sessions) if found is not None else None)

    def _session_doctor(self, sess: Session) -> str | None:
        if sess.loaded:
            return main_doctor(sess.runs[-1].result.segments) if sess.runs else None
        return sess.stored_doctor

    def _search_ids(self, query: str) -> set[int]:
        """검색어의 낱말이 모두 맞는 진료: 환자·의사 이름, 날짜(11월 12일·11/12·오늘), 요약·대화 내용, 진료 번호(#12)."""
        today = date.today()
        found: set[int] | None = None
        for term in query.split():
            content: set[int] = set()
            if self.store is not None:
                try:
                    content = self.store.search_ids(term)
                except RecordError:
                    pass
            matched = set()
            for sess in self.sessions:
                text = ""
                if sess.loaded and sess.session_id not in content:  # 아직 저장 전인 내용(방금 끝난 진료)
                    text = "\n".join(seg.text for seg in sess.runs[-1].result.segments) if sess.runs else ""
                if (sess.session_id in content or term.lower() in text.lower() or term == f"#{sess.session_id}"
                        or record_matches(term, sess.patient, self._session_doctor(sess), sess.created_at, today)):
                    matched.add(sess.session_id)
            found = matched if found is None else found & matched
        return found or set()

    def _on_history_search(self, query: str) -> None:
        self._history_query = query
        self._refresh_history()

    def _on_history_group(self, mode: str) -> None:
        self.settings.history_group = mode
        self._refresh_history()

    def _edit_patient(self, session_id: int) -> None:
        """진료에 환자 이름을 적는다 (-1: 지금 고른 진료). 이 컴퓨터 의 진료 기록에만 저장하고 요약 요청에는 넣지 않는다."""
        sess = self.session if session_id < 0 else self._session_by_id(session_id)
        if sess is None:
            return
        dialog = QInputDialog(self)
        dialog.setStyleSheet(theme.stylesheet("QInputDialog"))
        dialog.setWindowTitle("환자 이름")
        dialog.setLabelText(f"진료 #{sess.session_id} ({sess.created_at.month}월 {sess.created_at.day}일 "
                            f"{sess.created_at:%H:%M}) 환자 이름\n이 컴퓨터 의 진료 기록에만 저장합니다. 비우면 지웁니다.")
        dialog.setTextValue(sess.patient)
        if self.interactive and not dialog.exec():
            return
        self.set_patient(sess, dialog.textValue())

    def set_patient(self, sess: Session, name: str) -> None:
        sess.patient = " ".join(name.split())[:60]
        if sess.loaded:
            self._persist(sess)
        else:
            self.saver.set_patient(sess.session_id, sess.patient)  # 목록만 읽은 기록: 내용을 읽지 않고 이름만
        self._refresh_history()

    def _history_row(self, sess: Session, group: str | None = None) -> HistoryRow:
        """group: 환자별로 묶을 때 묶음 제목(환자 이름). 그때는 줄에 이름 대신 날짜·시각을 쓴다."""
        start = sess.created_at.strftime("%H:%M")
        if group is not None:
            day = group
            on = f"{sess.created_at.month}월 {sess.created_at.day}일 "
        else:
            day = day_label(sess.created_at.date(), date.today())
            on = ""
        patient = "" if group is not None else sess.patient
        if sess is self._recording_session and self.recorder.active:
            return HistoryRow(sess.session_id, f"{on}{start} ~", "녹음 중", "recording", day=day, patient=patient)
        if sess.source.startswith("파일"):
            when = f"{on}{start} · {sess.source}"
        elif sess.duration_s > 0:
            end = (sess.created_at + timedelta(seconds=sess.duration_s)).strftime("%H:%M")
            when = f"{on}{start}–{end} · {_mmss(sess.duration_s)}"
        else:
            when = f"{on}{start}"
        status = self._session_status(sess)
        doctor = self._session_doctor(sess)
        reason = sess.end_reason if sess.end_reason and sess.end_reason != "직접 종료" else ""
        note = " · ".join(part for part in (f"의사 {doctor}" if doctor else "", reason) if part)
        return HistoryRow(sess.session_id, when, status, HISTORY_KINDS.get(status, "warn"), note, day, patient)

    def _session_status(self, sess: Session) -> str:
        if not sess.loaded:
            return sess.stored_status or "대기"
        if self._task(sess, "transcribe"):
            return "전사 중…"
        if self._task(sess, "summary"):
            return "요약 중…"
        if sess.draft is not None:
            return "요약 준비됨"
        if sess.summary_error:
            return "요약 실패"
        if sess.transcribe_error:
            return "전사 실패"
        if sess.summary_short:
            return sess.summary_short
        if sess.runs:
            return "전사 완료"
        return "대기"

    def _refresh_mic_status(self) -> None:
        status = mic_permission.authorization_status()
        label = mic_permission.STATUS_LABEL.get(status, status)
        color = {"authorized": "#4ade80", "denied": "#f87171", "restricted": "#f87171"}.get(status, "#8b93a1")
        tip = mic_permission.permission_tooltip()
        for target in (self.mic_status_label, self.simple.mic_status):
            target.setText(f"마이크 권한: {label}")
            target.setStyleSheet(f"color: {color};")
            target.setToolTip(tip)

    def _refresh_tool_status(self) -> None:
        s = self.settings
        cli = s.whisper_cli_path()
        model = s.whisper_model_path()
        vad = s.vad_model_path()
        claude = s.claude_cli_path()

        def mark(ok: bool) -> str:
            return "✓" if ok else "✗ 없음"

        if self._local_engine_selected():
            local = self._selected_local_model()
            local_path = installer.local_model_path(local)
            ready = installer.llama_server_path().is_file() and installer.is_valid(local.file, local_path)
            engine, engine_tip = f"요약: 로컬 {local.label} {mark(ready)}", f"로컬 모델: {local_path}"
        else:
            engine, engine_tip = f"요약: Claude CLI {mark(bool(claude))} ({s.claude_model})", f"claude: {claude}"
        self.tool_status_label.setText(
            f"whisper-cli {mark(bool(cli))} · Whisper 모델 {model.name} {mark(model.is_file())} · "
            f"VAD {vad.name} {mark(vad.is_file())} · {engine}"
        )
        self.tool_status_label.setToolTip(f"whisper-cli: {cli}\nWhisper 모델: {model}\nVAD 모델: {vad}\n{engine_tip}")

    def _refresh_auto_split_note(self) -> None:
        s = self.settings
        if s.auto_split:
            cues = parse_cues(s.split_cues_next)
            first = f"'{cues[0]}' 같은 말이 나오거나 " if cues else ""
            text = f"자동 나누기 켜짐 · {first}{s.split_silence_s}초 동안 말이 없으면 다음 진료로 넘어갑니다"
            tip = "다음 환자 신호: " + ", ".join(cues) + "\n끝인사: " + ", ".join(parse_cues(s.split_cues_end))
        else:
            text, tip = "자동 나누기 꺼짐 · 스페이스바로만 진료를 끝냅니다", ""
        self.simple.auto_split_note.setText(text)
        self.simple.auto_split_note.setToolTip(tip)

    # ------------------------------------------------------------ 입력 장치
    def _load_devices(self, refresh: bool) -> None:
        try:
            devices = list_input_devices(refresh=refresh)
        except RecorderError as exc:
            devices = []
            self._show_error(str(exc))
        self._devices = ([self._demo_device] if self._demo_device else []) + devices
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for device in self._devices:
            self.device_combo.addItem(device.label(), device.index)
        if not self._devices:
            self.device_combo.addItem("입력 장치 없음 — 오디오 파일 불러오기로 테스트하세요", -1)
        lid_note = None
        if self._demo_device:
            preferred = 0
        else:
            preferred = next((i for i, d in enumerate(self._devices) if d.name == self.settings.input_device), None)
            if preferred is None:
                preferred = next((i for i, d in enumerate(self._devices) if d.is_default), 0)
            # 덮개를 닫으면(클램쉘) 내장 마이크는 0 만 들어온다: 다른 입력 장치가 있으면 그것을 고른다 (저장한 선택은 그대로)
            if self._devices and mic_permission.is_builtin_mic(self._devices[preferred].name) and mic_permission.lid_closed():
                others = [i for i, d in enumerate(self._devices) if not mic_permission.is_builtin_mic(d.name)]
                if others:
                    preferred = others[0]
                    lid_note = (f"맥북 덮개가 닫혀 있어(클램쉘 모드) 내장 마이크 대신 "
                                f"{mic_permission.quoted(self._devices[preferred].name)} 을(를) 골랐습니다. "
                                "덮개를 닫으면 내장 마이크는 하드웨어로 꺼집니다.")
                else:
                    lid_note = mic_permission.lid_closed_message([])
        self.device_combo.setCurrentIndex(preferred)
        self.device_combo.blockSignals(False)
        self._loading_devices = True  # 자동으로 고른 장치는 설정에 저장하지 않는다
        self.simple.device_combo.setCurrentIndex(preferred)  # 같은 목록(모델)을 쓰는 간단 화면 쪽 선택 맞추기
        self._loading_devices = False
        if lid_note:
            self._show_error(lid_note, warn=True)
        default = next((d for d in devices if d.is_default), None)
        self.default_device_label.setText(f"시스템 기본 입력: {default.name if default else '없음'}")
        if not self._devices:
            self._show_error("사용할 수 있는 입력 장치가 없습니다. 마이크를 연결한 뒤 '장치 새로고침'을 누르거나, 오디오 파일을 불러와 테스트하세요.", warn=True)
        self._refresh_demo_warning()
        self._refresh_controls()

    def _refresh_devices_before_popup(self) -> None:
        # 간단 화면에는 '장치 새로고침' 버튼이 없으므로, 목록을 펼칠 때 새로 꽂은 마이크까지 다시 읽는다.
        if not self.recorder.active:
            self._load_devices(refresh=True)

    def _selected_device(self) -> InputDevice | None:
        index = self.device_combo.currentData()
        return next((d for d in self._devices if d.index == index), None)

    def _on_device_changed(self, _i: int) -> None:
        device = self._selected_device()
        if device is not None and not device.demo_file and not self._loading_devices:
            self.settings.input_device = device.name
        self._refresh_demo_warning()

    def _refresh_demo_warning(self) -> None:
        device = self._selected_device()
        demo = bool(device is not None and device.demo_file)
        self.demo_warning.setVisible(demo)
        self.simple.demo_warning.setVisible(demo)

    # ---------------------------------------------------------------- 진료
    def _confirm(self, title: str, text: str) -> bool:
        if not self.interactive:
            return True
        answer = QMessageBox.question(
            self, title, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _create_session(self, source: str, offset_s: float = 0.0, started: datetime | None = None) -> Session:
        session = Session(session_id=self._new_session_id(), source=source, offset_s=offset_s)
        if started is not None:
            session.created_at = started
        self.sessions.append(session)
        self._trim_history()
        return session

    def _new_session_id(self) -> int:
        """새 진료 번호. 저장소가 번호를 매겨 앱을 다시 열어도 이어지고, 지운 번호는 다시 쓰지 않는다."""
        session_id = self._session_counter + 1
        if self.store is not None:
            try:
                session_id = max(session_id, self.store.allocate_id())
            except RecordError:
                pass  # 이번 실행 안에서만 번호를 이어 간다 (저장 오류는 저장할 때 알린다)
        self._session_counter = session_id
        return session_id

    def _trim_history(self) -> None:
        """오래된 진료의 녹음은 메모리에서 놓는다 (저장소에 있으니 필요할 때 다시 읽는다)."""
        busy = {sid for sid, _kind in self._tasks}
        for old in self.sessions[:-KEEP_AUDIO_SESSIONS]:
            if old.session_id not in busy and old is not self.session and not old.audio_dirty:
                old.pcm16 = None

    def _session_by_id(self, session_id: int) -> Session | None:
        return next((s for s in self.sessions if s.session_id == session_id), None)

    def _select_session(self, session: Session | None) -> None:
        self.session = session
        self._show_session()

    def _on_history_selected(self, session_id: int) -> None:
        session = self._session_by_id(session_id)
        if session is not None and session is not self.session:
            self._select_session(session)

    def _cancel_session_tasks(self, session: Session) -> None:
        for kind in ("transcribe", "summary"):
            task = self._tasks.pop((session.session_id, kind), None)
            if task is not None:
                task.cancel.set()

    def _current_run(self) -> TranscriptRun | None:
        if self.session is None:
            return None
        return self.session.run(self.run_combo.currentData())

    def _ensure_loaded(self, session: Session | None) -> None:
        """앱을 열 때 목록용 정보만 읽은 기록이면 전사문·요약을 마저 읽는다."""
        if session is None or session.loaded or self.store is None:
            return
        try:
            apply_data(session, self.store.load_data(session.session_id))
        except RecordError as exc:  # 기록은 그대로 두고(덮어쓰지 않음) 알린다. 지울 수는 있다.
            self._show_error(str(exc), warn=True)

    def _session_audio(self, session: Session) -> np.ndarray | None:
        """진료 녹음. 메모리에서 놓았으면 저장소에서 다시 읽는다."""
        if session.pcm16 is None and session.stored_audio and self.store is not None:
            self.saver.flush()  # 아직 쓰는 중인 녹음(무음을 줄인 것)이 있으면 기다린다
            try:
                session.pcm16 = self.store.read_audio(session.session_id)
            except RecordError as exc:
                self._show_error(str(exc), warn=True)
        return session.pcm16

    def _persist(self, session: Session | None) -> None:
        """진료 기록을 저장한다 (백그라운드 스레드). 녹음이 바뀌었으면 녹음 파일도 새로 쓴다."""
        if session is None or not session.loaded or self._session_by_id(session.session_id) is not session:
            return
        audio = session.pcm16 if session.audio_dirty else None
        if audio is None and not (session.stored_audio or session.has_results):
            return  # 녹음도 결과도 없는 진료(너무 짧음 등)는 남기지 않는다
        pending = next((kind for kind in ("transcribe", "summary") if self._task(session, kind)), "")
        self.saver.save(record_row(session, self._session_status(session), pending), session_data(session), audio)
        if audio is not None and self.store is not None:
            session.audio_dirty, session.stored_audio = False, True

    def _on_record_error(self, message: str) -> None:
        self._show_error(f"진료 기록 저장 오류: {message}", warn=True)

    def _on_delete_requested(self, session_id: int) -> None:
        """진료 기록을 지운다 (-1: 지금 고른 진료). 녹음·전사문·요약을 저장소에서도 지운다."""
        session = self.session if session_id < 0 else self._session_by_id(session_id)
        if session is None:
            return
        if session is self._recording_session and self.recorder.active:
            self._show_error("녹음 중인 진료는 지울 수 없습니다. 녹음을 끝내거나 취소한 뒤 지우세요.", warn=True)
            return
        when = f"{session.created_at.month}월 {session.created_at.day}일 {session.created_at:%H:%M}"
        if session.patient:
            when = f"{session.patient} · {when}"
        if not self._confirm_delete(f"진료 #{session.session_id} ({when}) 기록을 삭제할까요?",
                                    "녹음·전사문·요약이 이 컴퓨터 에서 지워지며 되돌릴 수 없습니다."):
            return
        self._cancel_session_tasks(session)
        self.sessions = [s for s in self.sessions if s is not session]
        self.saver.delete(session.session_id)
        if self.session is session:
            self._select_session(None)
            self.simple.collapse_results()
        self._set_status(f"진료 #{session.session_id} 기록을 삭제했습니다", "done")
        self._refresh_controls()

    def _confirm_delete(self, text: str, detail: str) -> bool:
        if not self.interactive:
            return True
        box = QMessageBox(QMessageBox.Icon.Warning, "기록 삭제", text, parent=self)
        box.setInformativeText(detail)
        delete = box.addButton("삭제", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(box.addButton("취소", QMessageBox.ButtonRole.RejectRole))
        box.exec()
        return box.clickedButton() is delete

    def _show_session(self) -> None:
        self._ensure_loaded(self.session)
        s = self.session
        self.source_label.setText(f"· 진료 #{s.session_id} · {s.source}" if s else "")
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
        # 전사가 안 된 기록을 고르면 그 이유와 '전사 다시 시도' 버튼을 보여 준다
        failed = s.transcribe_error if s is not None and not s.runs and not self._task(s, "transcribe") else None
        if failed:
            self._show_error(failed, warn=True)
            self._shown_session_error = failed
        elif self._shown_session_error and self.error_label.text() == self._shown_session_error:
            self._clear_error()
            self._shown_session_error = None

    # ---------------------------------------------------------------- 녹음
    def _on_record_clicked(self) -> None:
        if self.recorder.active:
            self._stop_recording_and_process("직접 종료")
            return
        device = self._selected_device()
        if device is None:
            self._show_error("입력 장치가 없습니다. 마이크를 연결하거나 오디오 파일을 불러오세요.")
            return
        if device.demo_file:  # 데모 입력은 마이크를 쓰지 않는다
            self._start_recording(device)
            return
        if mic_permission.is_builtin_mic(device.name) and mic_permission.lid_closed():
            # 덮개를 닫으면 내장 마이크는 0 만 들어온다: 무음을 녹음하지 않고 바로 알린다
            others = [d.name for d in self._devices if not d.demo_file and not mic_permission.is_builtin_mic(d.name)]
            self._show_error(mic_permission.lid_closed_message(others))
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
        self._stop_playback()
        try:
            self.recorder.start(device)
        except RecorderError as exc:
            self._show_error(f"{exc}\n장치를 다시 고르거나 '장치 새로고침' 후 다시 시도하세요.")
            self._set_status("녹음 시작 실패", "error")
            self._refresh_controls()
            return
        source = f"데모 입력 · {Path(device.demo_file).name}" if device.demo_file else f"마이크 · {device.name}"
        self._recording_id += 1
        self._recording_started = datetime.now()
        session = self._create_session(source, 0.0, self._recording_started)
        self._recording_session = session
        self.live_transcriber.cancel_all()
        self.live = LiveActivityMonitor(
            min_speech_ms=self.min_speech_spin.value(), min_silence_ms=self.silence_spin.value()
        )
        self._live_entries = []
        self._live_runner = None
        self._live_issue = None
        self._silent_warned = False
        self._last_record_second = -1
        self.waveform.clear()
        self.timeline.clear()
        self.elapsed_label.setText("00:00")
        self.simple.elapsed.setText("00:00")
        self.simple.collapse_results()
        self._clear_error()
        self._set_status("녹음 중", "recording")
        self._select_session(session)
        self._read_emr_patient(session)
        self.tabs.setCurrentWidget(self.live_view)
        self._render_live()
        self._rec_timer.start()
        self._refresh_controls()

    def _on_rec_tick(self) -> None:
        blocks = self.recorder.drain()
        rate = self.recorder.samplerate
        peak = 0.0
        if blocks:
            mono = to_mono(np.concatenate(blocks))
            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            level = rms_dbfs(mono)
            self.level_meter.set_level(level)
            self.simple.level.set_level(level)
            for block in blocks:
                for event in self.live.feed(to_mono(block), rate):
                    self._on_live_event(event)
        self.waveform.push(peak, self.live.in_speech)
        total = self.recorder.elapsed
        rec = self._recording_session
        consult = total - (rec.offset_s if rec is not None else 0.0)
        if rec is not None and not rec.emr_rechecked and consult >= EMR_RECHECK_S:
            rec.emr_rechecked = True  # 자동으로 나뉜 진료는 이때 처음 읽는다 (다음 환자 차트를 열 시간)
            self._read_emr_patient(rec)
        self.elapsed_label.setText(_mmss(consult))
        self.simple.elapsed.setText(_mmss(consult))
        self.timeline.set_data(
            total, [(c.start, c.end if c.end is not None else self.live.time) for c in self.live.clips],
            live=True, cursor=total,
        )
        if int(total) != self._last_record_second:
            self._last_record_second = int(total)
            self._refresh_pipeline()
        speaking = self.live.in_speech
        self.live_indicator.setText("● 말하는 중" if speaking else "○ 조용함")
        self.live_indicator.setStyleSheet("color: #4ade80; font-weight: bold;" if speaking else "color: gray;")
        if not self._silent_warned and self.recorder.looks_digitally_silent():
            self._silent_warned = True
            device = self.recorder.device
            self._show_error(mic_permission.silent_input_message(device.name if device else ""), warn=True)
        if self.recorder.active and not self.recorder.stream_alive():
            self._show_error("입력 장치에서 소리가 더 들어오지 않습니다(연결이 끊겼을 수 있음). 지금까지 녹음한 내용으로 전사합니다.", warn=True)
            self._stop_recording_and_process("입력 끊김")
            return
        self._check_auto_split()

    # ------------------------------------------------------- 자동 진료 나누기
    def _last_speech_end(self, after: float | None = None) -> float | None:
        ends = [c.end for c in self.live.clips if c.end is not None and (after is None or c.end > after + 0.05)]
        return max(ends) if ends else None

    def _check_auto_split(self) -> None:
        s = self.settings
        rec = self._recording_session
        if rec is None or not self.recorder.active or self.live.in_speech:
            return
        now = self.live.time
        last = self._last_speech_end()
        if s.auto_stop_idle_s > 0 and now - (last if last is not None else 0.0) >= s.auto_stop_idle_s:
            self._stop_recording_and_process(f"말 없음 {s.auto_stop_idle_s}초 · 녹음 자동 멈춤")
            return
        if not s.auto_split:
            return
        session_end = self._last_speech_end(after=rec.offset_s)
        if session_end is not None and now - session_end >= s.split_silence_s:
            self._split_recording(min(now, session_end + 0.5), f"무음 {s.split_silence_s}초")

    def _check_cues(self, segments) -> None:
        """실시간 전사에서 다음 환자 신호·끝인사를 찾으면 진료를 나눈다."""
        s = self.settings
        rec = self._recording_session
        if not (s.auto_split and rec is not None and self.recorder.active):
            return
        next_cues, end_cues = parse_cues(s.split_cues_next), parse_cues(s.split_cues_end)
        for seg in segments:
            cue = find_cue(seg.text, next_cues, end_cues)
            if cue is None:
                continue
            boundary = seg.start if cue.kind == "next" else seg.end + 0.2
            if boundary <= rec.offset_s or boundary > self.recorder.elapsed:
                continue  # 이미 나뉜 앞 진료의 말
            if boundary - rec.offset_s < s.min_session_s:
                continue  # 진료가 너무 짧다 (예: 새 진료 첫머리의 "들어오세요")
            if not self._cue_from_doctor(seg):
                continue  # 환자가 "들어오세요"를 따라 말한 경우 등
            label = "다음 환자 신호" if cue.kind == "next" else "끝인사"
            self._split_recording(boundary, f"{label} '{cue.phrase}'")
            return

    def _cue_from_doctor(self, seg) -> bool:
        """등록한 의사가 있고 '의사 목소리일 때만'이 켜져 있으면, 신호 단어를 말한 목소리가 등록한 의사인지 본다.

        확인할 수 없으면(모델 없음, 너무 짧은 발화) 예전처럼 나눈다.
        """
        if not (self.settings.split_doctor_only and self.voices.voices):
            return True
        encoder = self._speaker_encoder_or_none()
        if encoder is None:
            return True
        try:
            clip = to_whisper_pcm(self.recorder.snapshot(seg.start, seg.end), self.recorder.samplerate)
        except Exception:  # 앞부분이 이미 메모리에서 지워진 경우 등
            return True
        if len(clip) < WHISPER_SR // 2:
            return True
        name, _score = self.voices.identify(encoder.embed(clip), self.settings.speaker_threshold)
        return name is not None

    # ------------------------------------------------------------ 말한 사람 (화자)
    def _speaker_encoder(self) -> SpeakerEncoder:
        """화자 인식 모델 (처음 쓸 때 연다). 없으면 SpeakerError."""
        if self._speaker_encoder_cache is None:
            self._speaker_encoder_cache = SpeakerEncoder(self.settings.speaker_model_path())
        return self._speaker_encoder_cache

    def _speaker_encoder_or_none(self) -> SpeakerEncoder | None:
        try:
            return self._speaker_encoder()
        except SpeakerError as exc:
            self._speaker_issue = str(exc)
            return None

    def _refresh_voices_button(self) -> None:
        count = len(self.voices.voices)
        for button in (self.voices_btn, self.simple.voices_btn):
            button.setText(f"의사 목소리 {count}" if count else "의사 목소리")

    def _open_voices(self) -> None:
        if self.recorder.active:
            return
        runner, params = self._whisper_runner(), self._vad_params()

        def transcribe(pcm16, prompt=None):  # 따라 읽기 분석용 (진료 전사와 같은 설정)
            return runner.transcribe(pcm16, use_vad=True, params=params, prompt=prompt)

        dialog = VoiceDialog(self.voices, self._speaker_encoder, self._selected_device(), self, diction=self.diction,
                             transcribe=transcribe, base_prompt=self.settings.whisper_prompt)
        dialog.setStyleSheet(theme.stylesheet("VoiceDialog"))
        dialog.exec()
        for old, new in dialog.renamed:
            self._rename_doctor(old, new)
        self._refresh_voices_button()

    def _rename_doctor(self, old: str, new: str) -> None:
        """의사 이름을 고쳤다: 지금 열린 진료와 저장된 진료 기록의 말한 사람·의사 표시도 새 이름으로 바꾼다."""
        for session in self.sessions:
            if not session.loaded:
                if session.stored_doctor == old:
                    session.stored_doctor = new
                continue
            for run in session.runs:
                if run.result.doctor == old:
                    run.result.doctor = new
                for seg in run.result.segments:
                    if seg.speaker == old:
                        seg.speaker = new
        self.saver.rename_speaker(old, new)
        self._show_session()

    # ------------------------------------------------------------ 딕션 사전 (고쳐 쓰기)
    def _transcript_menu(self, view: QTextBrowser, pos) -> None:
        menu = view.createStandardContextMenu()
        selected = view.textCursor().selectedText().strip()
        action = menu.addAction("선택한 말 고쳐 쓰기…")
        action.setEnabled(bool(selected) and " " not in selected and self._current_run() is not None)
        action.triggered.connect(lambda: self._correct_selection(selected))
        menu.exec(view.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def _diction_scopes(self) -> list[str]:
        return list(dict.fromkeys(self.voices.names() + self.diction.doctors()))

    def _correct_selection(self, selected: str) -> None:
        run = self._current_run()
        if run is None:
            return
        doctor = run.result.doctor or main_doctor(run.result.segments)
        dialog = CorrectionDialog(selected, self._diction_scopes(), doctor or COMMON, self)
        dialog.setStyleSheet(theme.stylesheet("CorrectionDialog"))
        if dialog.exec():
            self._add_correction(*dialog.values())

    def _add_correction(self, wrong: str, right: str, scope: str, add_term: bool = True) -> bool:
        """교정을 딕션 사전에 저장하고, 보고 있는 진료의 전사문에 바로 반영한다 (다음 전사부터는 자동)."""
        try:
            rule = self.diction.add_correction(scope, wrong, right, add_term)
            self.diction.save()
        except (DictionError, OSError) as exc:
            self._show_error(f"교정을 저장하지 못했습니다: {exc}", warn=True)
            return False
        s = self.session
        run = self._current_run()
        if run is not None:
            before = [seg.text for seg in run.result.segments]
            run.result.segments = apply_corrections(run.result.segments, self.diction)
            if s is not None and s.draft is not None and before != [seg.text for seg in run.result.segments]:
                s.summary_stale = True  # 고친 전사문으로 다시 요약할 수 있게 한다
            self._show_run(run)
            self._persist(s)
        who = "공통 사전" if scope == COMMON else f"{scope} 사전"
        self._set_status(f"{who}에 저장: {rule.wrong} → {rule.right} · 다음 전사부터 자동으로 고칩니다", "done")
        self._refresh_controls()
        return True

    def _split_recording(self, boundary: float, reason: str) -> None:
        """녹음은 이어 가고, boundary 앞까지를 한 진료로 떼어 전사·요약한다."""
        rec = self._recording_session
        if rec is None or not self.recorder.active:
            return
        boundary = min(max(boundary, rec.offset_s), self.recorder.elapsed)
        rate = self.recorder.samplerate
        audio = self.recorder.snapshot(rec.offset_s, boundary)
        rec.duration_s = boundary - rec.offset_s
        rec.end_reason = reason
        rec.live_clips = [c for c in self.live.clips if rec.offset_s - 0.05 <= c.start < boundary]
        started = (self._recording_started or datetime.now()) + timedelta(seconds=boundary)
        nxt = self._create_session(rec.source, boundary, started)
        nxt.auto_continued = True
        nxt.via_hotkey = rec.via_hotkey  # 버튼으로 시작한 녹음에서 나뉜 진료도 요약이 끝나면 복사한다
        self._recording_session = nxt
        self._live_entries.append(LiveEntry(
            clip_id=-1, start=boundary, status="split",
            text=f"진료 #{rec.session_id} 끝 ({reason}) → 진료 #{nxt.session_id}",
        ))
        self._render_live()
        self.recorder.discard_before(boundary - LIVE_PAD_S - 1.0)
        if len(audio) >= 0.3 * max(1, rate):
            rec.original_rate, rec.original_channels = rate, audio.shape[1]
            self._start_transcription(rec, source_audio=(audio, rate), auto_summary=self.settings.auto_summarize)
        else:
            rec.summary_note, rec.summary_short = "녹음이 너무 짧아 전사하지 않았습니다.", "너무 짧음"
        if self.session is rec:
            self._show_session()  # 끝난 진료의 처리 상황을 계속 보여 준다
        self._set_status(f"진료 #{rec.session_id} 끝 ({reason}) · 진료 #{nxt.session_id} 녹음 중", "recording")
        self._refresh_controls()

    # ------------------------------------------------------- 실시간 전사 미리보기
    def _on_live_event(self, event: str) -> None:
        clips = self.live.clips
        if event == "start":
            self._live_entries.append(LiveEntry(clip_id=len(clips) - 1, start=clips[-1].start))
        elif event == "end":
            self._close_live_clip(len(clips) - 1)
        elif event == "split":
            self._close_live_clip(len(clips) - 2)
            self._live_entries.append(LiveEntry(clip_id=len(clips) - 1, start=clips[-1].start))
        self._render_live()

    def _live_entry(self, clip_id: int) -> LiveEntry | None:
        return next((e for e in self._live_entries if e.clip_id == clip_id), None)

    def _live_transcription_enabled(self) -> bool:
        # 미리보기를 꺼도 신호 단어로 진료를 나누려면 실시간 전사가 필요하다.
        s = self.settings
        cues = parse_cues(s.split_cues_next) or parse_cues(s.split_cues_end)
        return self.live_check.isChecked() or (s.auto_split and bool(cues))

    def _ensure_live_runner(self) -> WhisperRunner | None:
        if not self._live_transcription_enabled():
            return None
        if self._live_runner is None and self._live_issue is None:
            runner = self._whisper_runner()
            issues = runner.problems(need_vad=True)
            if issues:
                self._live_issue = f"실시간 전사를 할 수 없습니다: {issues[0]}"
            else:
                self._live_runner = runner
        return self._live_runner

    def _close_live_clip(self, clip_id: int) -> None:
        clip = self.live.clips[clip_id]
        entry = self._live_entry(clip_id)
        if entry is None:
            return
        entry.end = clip.end
        runner = self._ensure_live_runner()
        if runner is None:
            entry.status = "off"
            return
        offset = max(0.0, clip.start - LIVE_PAD_S)
        audio = self.recorder.snapshot(offset, (clip.end or self.live.time) + LIVE_PAD_S)
        if len(audio) < 0.2 * max(1, self.recorder.samplerate):
            self._live_entries.remove(entry)
            return
        entry.status = "pending"
        self.live_transcriber.submit(
            self._recording_id, clip_id, offset, audio, self.recorder.samplerate, runner, self._vad_params()
        )

    @Slot(int, int, object, object)
    def _on_live_result(self, recording_id: int, clip_id: int, segments, error) -> None:
        if recording_id != self._recording_id:
            return
        entry = self._live_entry(clip_id)
        if entry is None:
            return
        if error is not None:
            entry.status, entry.text = "error", str(error)
        elif not segments:
            self._live_entries.remove(entry)  # Silero VAD 가 말소리가 아니라고 본 클립
        else:
            entry.status, entry.text = "done", " ".join(s.text for s in segments)
        self._render_live()
        if segments:
            self._check_cues(segments)

    def _render_live(self) -> None:
        note = (
            '<p style="color:#8b93a1;">실시간 미리보기 · 말이 끊길 때마다 그 부분만 whisper.cpp 로 전사합니다(이 컴퓨터). '
            "앞뒤 문맥 없이 짧게 전사해 정확도가 낮을 수 있으며, 진료가 끝나면 그 진료 녹음 전체로 다시 전사한 결과가 "
            "'전사문' 탭에 나옵니다.</p>"
        )
        rows = [note]
        if self._live_issue:
            rows.append(f'<p style="color:#f87171;">{html.escape(self._live_issue)}</p>')
        if not self.live_check.isChecked():
            rows.append('<p style="color:#fdba74;">실시간 전사 미리보기가 꺼져 있습니다.</p>')
        if not self._live_entries:
            message = ("말을 하면 여기에 바로 나타납니다…" if self.recorder.active
                       else "‘● 녹음 시작’(또는 스페이스바)을 누르고 말하면, 말이 끊길 때마다 그 부분이 여기에 바로 전사됩니다.")
            rows.append(f'<p style="color:#8b93a1;">{message}</p>')
        for entry in self._live_entries:
            if entry.status == "split":
                rows.append(f'<p style="color:#8b93a1;border-top:1px solid #2a2f3a;">── {html.escape(entry.text)} ──</p>')
                continue
            end = f" → {format_time(entry.end)}" if entry.end is not None else ""
            stamp = f'<span style="color:#8b93a1;font-family:Menlo,Consolas,DejaVu Sans Mono,monospace;">[{format_time(entry.start)}{end}]</span>'
            body = {
                "listening": '<span style="color:#4ade80;">● 말하는 중…</span>',
                "pending": '<span style="color:#93a8f8;">인식 중…</span>',
                "off": '<span style="color:#8b93a1;">(미리보기 꺼짐)</span>',
                "stopped": '<span style="color:#8b93a1;">녹음 종료 — 최종 전사에서 처리</span>',
                "error": f'<span style="color:#f87171;">미리보기 실패: {html.escape(entry.text)}</span>',
            }.get(entry.status, html.escape(entry.text))
            rows.append(f"<p>{stamp}&nbsp; {body}</p>")
        self.live_view.setHtml("".join(rows))
        bar = self.live_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_live_toggled(self, checked: bool) -> None:
        self.settings.live_preview = checked
        if not self._live_transcription_enabled():
            self.live_transcriber.cancel_all()
            for entry in self._live_entries:
                if entry.status == "pending":
                    entry.status = "off"
        self._render_live()

    def _stop_live_preview(self) -> None:
        self.live_transcriber.cancel_all()
        for entry in self._live_entries:
            if entry.status in ("listening", "pending"):
                entry.status = "stopped"
                if entry.end is None:
                    entry.end = self.live.time
        self._render_live()

    # ------------------------------------------------------------ 녹음 종료
    def _stop_recording_and_process(self, reason: str = "직접 종료") -> None:
        self._rec_timer.stop()
        rec = self._recording_session
        self.recorder.drain()
        end = self.recorder.elapsed
        rate = self.recorder.samplerate
        audio = self.recorder.snapshot(rec.offset_s, end) if rec is not None else None
        try:
            self.recorder.stop(keep_audio=False)
        except Exception as exc:
            self._show_error(f"녹음을 정상적으로 끝내지 못했습니다: {exc}")
        self._recording_session = None
        self.level_meter.set_level(-120.0)
        self.simple.level.set_level(-120.0)
        self.live_indicator.setText("")
        if self.live.clips and self.live.clips[-1].end is None:
            self.live.clips[-1].end = self.live.time
        self._stop_live_preview()
        if rec is None:
            self._refresh_controls()
            return
        rec.duration_s = max(0.0, end - rec.offset_s)
        rec.end_reason = reason
        rec.live_clips = [c for c in self.live.clips if c.start >= rec.offset_s - 0.05]
        if rec.auto_continued and not rec.live_clips:
            # 자동 나누기로 이어진 뒤 아무 말도 없던 진료는 기록에 남기지 않는다.
            self.sessions.remove(rec)
            if self.session is rec:
                self._select_session(self.sessions[-1] if self.sessions else None)
            self._set_status(f"녹음을 멈췄습니다 ({reason})", "idle")
            self._refresh_controls()
            return
        if audio is None or len(audio) < 0.3 * max(1, rate):
            rec.summary_note, rec.summary_short = "녹음이 너무 짧아 전사하지 않았습니다.", "너무 짧음"
            self._set_status("녹음이 너무 짧아 전사하지 않았습니다.", "warn")
            self._show_session()
            return
        rec.original_rate, rec.original_channels = rate, audio.shape[1]
        if reason != "직접 종료":
            self._set_status(f"녹음을 멈췄습니다 ({reason})", "warn")
        self._start_transcription(rec, source_audio=(audio, rate), auto_summary=self.settings.auto_summarize)

    def _on_cancel_recording(self) -> None:
        if not self.recorder.active:
            return
        self._rec_timer.stop()
        self.recorder.abort()
        self.live_transcriber.cancel_all()
        rec, self._recording_session = self._recording_session, None
        if rec is not None and rec in self.sessions:
            self.sessions.remove(rec)
        self.level_meter.set_level(-120.0)
        self.simple.level.set_level(-120.0)
        self.elapsed_label.setText("00:00")
        self.simple.elapsed.setText("00:00")
        self.live_indicator.setText("")
        self.waveform.clear()
        self.timeline.clear()
        self._live_entries = []
        self.session = self.sessions[-1] if self.sessions else None
        self._show_session()
        self._render_live()
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
        """파일을 새 진료로 추가해 전사·요약한다. 이전 진료의 처리는 그대로 이어진다."""
        if self.recorder.active:
            return
        session = self._create_session(f"파일 · {path.name}")
        self._live_entries = []
        self.waveform.clear()
        self.simple.collapse_results()
        self._clear_error()
        self._set_status("대기 중", "idle")
        self._select_session(session)
        self._render_live()
        self._start_transcription(session, source_path=path, auto_summary=self.settings.auto_summarize)

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
        self.settings.vad = self._vad_params()
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

    def _start_transcription(self, session: Session, *, source_audio=None, source_path: Path | None = None,
                             auto_summary: bool) -> None:
        use_vad = self.vad_check.isChecked()
        params = self._vad_params()
        runner = self._whisper_runner()
        job = Job("transcribe", session.session_id, session.new_run_id(), auto_summary)
        pending = {"audio": source_audio}
        # 등록한 의사가 있으면 전사 뒤에 발화마다 말한 사람을 붙인다 (모델이 없으면 건너뜀)
        voices = self.voices.copy()
        diction = self.diction.copy()
        encoder = self._speaker_encoder_or_none() if voices.voices else None
        threshold = self.settings.speaker_threshold
        base_prompt = self.settings.whisper_prompt
        trim = not session.audio_trimmed
        saver = self.saver

        def work(cancel) -> tuple[TranscriptionResult, Trimmed | None]:
            if session.pcm16 is None:
                if source_path is not None:
                    data, rate = load_audio_file(source_path)
                    session.original_rate, session.original_channels = rate, data.shape[1]
                else:
                    data, rate = pending.pop("audio")
                session.pcm16 = to_whisper_pcm(data, rate)  # 원본 버퍼는 여기서 놓는다
                del data
                # 전사 도중 앱이 꺼져도 녹음이 남도록 받은 그대로 먼저 저장한다 (전사가 끝나면 무음을 줄인 녹음으로 바꾼다)
                if saver.store is not None:
                    session.stored_audio = True
                    saver.save(record_row(session, "전사 중…", "transcribe"), session_data(session), session.pcm16)
            # 의사별 용어 목록이 있으면 이 진료의 의사를 먼저 알아내 그 의사의 용어를 Whisper 힌트에 넣는다
            doctor = None
            if encoder is not None and diction.doctors():
                doctor = guess_doctor(session.pcm16, encoder, voices, threshold)
            prompt = None if diction.empty else build_prompt(base_prompt, diction, doctor)
            result = runner.transcribe(session.pcm16, use_vad=use_vad, params=params, cancel=cancel, prompt=prompt)
            if encoder is not None and result.segments:
                if result.phrases:  # 말 덩어리마다 말한 사람을 정하고(쉼 없이 끼어든 말은 덩어리를 나눠), 사람이 바뀌는 곳에서 줄을 나눠 다시 잇는다
                    labeled = label_turns(result.phrases, session.pcm16, encoder, voices, threshold, cancel)
                    result.segments = merge_phrases(labeled)
                    assign_regions(result.segments, result.regions, result.vad_enabled)
                else:
                    result.segments = label_segments(result.segments, session.pcm16, encoder, voices, threshold, cancel)
            result.segments = apply_corrections(result.segments, diction)  # 잘못 들린 말 → 바른 말
            result.doctor = doctor
            trimmed = None
            if trim and result.regions:  # 긴 무음을 줄인 녹음으로 바꾸고 전사 시각도 그 녹음 기준으로 옮긴다
                trimmed = trim_silence(session.pcm16, result.regions)
                result = remap_result(result, trimmed.spans)
            return result, trimmed

        session.transcribe_error = None
        self._tasks[(session.session_id, "transcribe")] = self.tasks.start(job, work)
        self._persist(session)  # 이미 저장된 진료를 다시 전사하는 경우: '전사 중'으로 표시해 둔다
        if session is self.session:
            self._clear_error()
            if voices.voices and encoder is None:
                self._show_error(f"말한 사람 표시를 건너뜁니다: {self._speaker_issue}", warn=True)
        self._refresh_controls()

    def _on_retranscribe(self) -> None:
        s = self.session
        if s is None or (s is self._recording_session and self.recorder.active) or self._session_audio(s) is None:
            return
        self._start_transcription(s, auto_summary=False)

    def _on_retry_clicked(self) -> None:
        """간단 화면의 '다시 시도': 전사가 안 된 기록은 저장된 녹음으로 다시 전사하고, 아니면 요약을 다시 한다."""
        s = self.session
        if s is not None and not s.runs and s.transcribe_error:
            if self._session_audio(s) is not None:
                self._start_transcription(s, auto_summary=self.settings.auto_summarize)
            return
        self._on_summary_clicked()

    def _on_cancel_tasks(self) -> None:
        if self.session is None:
            return
        self._cancel_session_tasks(self.session)
        self._persist(self.session)
        self._set_status("처리를 취소했습니다. 전사문·녹음은 남아 있습니다.", "warn")
        self._show_draft()

    @Slot(object, object, object)
    def _on_task_finished(self, task: Task, result, error) -> None:
        job = task.job
        key = (job.session_id, job.kind)
        if self._tasks.get(key) is task:
            del self._tasks[key]
        session = self._session_by_id(job.session_id)
        if session is None or task.cancel.is_set():
            self._refresh_controls()
            return
        if job.kind == "transcribe":
            self._on_transcribed(session, job, task, result, error)
        else:
            self._on_summarized(session, job, task, result, error)
        self._persist(session)
        self._refresh_controls()
        self.activity_finished.emit(job.kind)

    def _on_transcribed(self, session: Session, job: Job, task: Task,
                        output: tuple[TranscriptionResult, Trimmed | None] | None, error) -> None:
        shown = session is self.session
        if error is not None:
            if isinstance(error, procutil.Cancelled):
                if shown:
                    self._set_status("전사를 취소했습니다.", "warn")
                return
            if isinstance(error, (WhisperError, AudioFileError)):
                message = str(error)
            else:
                message = f"전사 중 예기치 못한 오류: {type(error).__name__}: {error}"
            if session.pcm16 is not None:
                message += "\n녹음은 남아 있으니 문제를 고친 뒤 '같은 녹음 다시 전사'를 누르세요."
            session.transcribe_error = message
            if shown:
                self._show_error(message)
                self._set_status("전사 실패", "error")
            return
        result, trimmed = output
        if trimmed is not None:
            # 녹음을 무음을 줄인 것으로 바꾸고, 앞선 전사도 같은 시각으로 옮긴다 (저장할 때 녹음 파일도 바꾼다)
            for earlier in session.runs:
                earlier.result = remap_result(earlier.result, trimmed.spans)
            session.pcm16 = trimmed.pcm16
            session.audio_trimmed = session.audio_dirty = True
        run = TranscriptRun(job.run_id, result)
        session.runs.append(run)
        session.transcribe_seconds = task.elapsed
        if shown:
            self._show_session()
            if len(session.runs) == 1 and not self.recorder.active:
                self.tabs.setCurrentWidget(self.transcript_view)  # 실시간 미리보기 대신 최종 전사
        if not job.auto_summary:
            if shown:
                self._set_status(f"전사 완료 · {result.elapsed:.1f}초 · {len(result.segments)}개 문장", "done")
            return
        blocked = self._summary_block_reason(run)
        if blocked:
            session.summary_note, session.summary_short = blocked
            if shown:
                self._show_draft()
                self._set_status(f"전사 완료 · {blocked[0]}", "warn")
            return
        self._start_summary(session, run)

    def _summary_block_reason(self, run: TranscriptRun) -> tuple[str, str] | None:
        """(화면 안내, 짧은 표시). 요약을 요청하지 않을 이유가 없으면 None."""
        r = run.result
        if r.no_speech_detected:
            return "음성이 감지되지 않아 요약을 요청하지 않았습니다.", "음성 없음"
        if r.is_empty:
            return "전사된 발화가 없어 요약을 요청하지 않았습니다.", "전사 없음"
        if self._local_engine_selected():
            model = self._selected_local_model()
            if not installer.llama_server_path().is_file() or not installer.is_valid(model.file, installer.local_model_path(model)):
                return (f"로컬 모델({model.label})이 아직 없습니다. 설정 → '설치 · 요약 방식…' 에서 받으세요.", "모델 없음")
            return None
        if not self.settings.claude_cli_path():
            return ("Claude Code CLI(claude)를 찾지 못해 요약하지 않았습니다. 설정 → '설치 · 요약 방식…' 에서 설치하거나 "
                    "경로를 지정하세요.", "claude 없음")
        return None

    # ---------------------------------------------------------------- 요약
    def _summary_mode(self) -> str:
        return self.summary_mode_combo.currentData() or "general"

    def _refresh_draft_title(self) -> None:
        general = self._summary_mode() == "general"
        self.draft_title.setText("주요 내용 요약" if general else "차트 초안 (검토용 초안)")
        self.evidence_check.setVisible(not general)  # 근거 문장은 진료 차트 초안에만 있다

    def _on_summary_mode_changed(self, _i: int) -> None:
        self.settings.summary_mode = self._summary_mode()
        self._refresh_draft_title()
        self._show_draft()

    def _on_summary_clicked(self) -> None:
        run = self._current_run()
        if self.session is None or run is None or run.result.is_empty:
            return
        if run.result.no_speech_detected and not self._confirm(
            "요약 요청", "VAD 가 음성을 찾지 못한 녹음입니다. 전사문이 무음에서 생긴 환각일 수 있습니다. 그래도 요약을 요청할까요?"
        ):
            return
        self._start_summary(self.session, run)

    def _start_summary(self, session: Session, run: TranscriptRun) -> None:
        s = self.settings
        mode = self._summary_mode()
        segments = run.result.segments
        based_on = f"진료 #{session.session_id} · {run.label}"
        engine = self._engine_name()
        if self._local_engine_selected():
            model = self._selected_local_model()
            local = LocalConfig(f"{model.label} (이 컴퓨터)", max(s.claude_timeout_s, 600), mode)
            llm = self._local_engine(model)
            label = local.label

            def ask(cancel):
                return summarize_local(segments, local, llm, cancel)
        else:
            cfg = ClaudeConfig(
                cli=s.claude_cli_path(),
                model=s.claude_model.strip() or DEFAULT_CLAUDE_MODEL,
                timeout_s=s.claude_timeout_s,
                effort=s.claude_effort,
                mode=mode,
            )
            label = cfg.model

            def ask(cancel):
                return summarize(segments, cfg, cancel)

        def work(cancel) -> ChartDraft | ConversationSummary:
            structured, _meta = ask(cancel)
            kind = ConversationSummary if mode == "general" else ChartDraft
            try:
                draft = kind.from_structured(structured, model=label, based_on=based_on)
            except ValueError as exc:
                raise SummaryError("parse", f"{engine} 응답이 {kind.kind_label} 형식과 다릅니다: {exc} 다시 시도하세요.") from exc
            if isinstance(draft, ChartDraft):
                verify_draft(draft, segments)
            return draft

        session.summary_error = None
        session.summary_note = None
        session.summary_short = None
        self._tasks[(session.session_id, "summary")] = self.tasks.start(
            Job("summary", session.session_id, run.run_id), work
        )
        self._persist(session)
        if session is self.session:
            self._show_draft()
        self._refresh_controls()

    def _on_summarized(self, session: Session, job: Job, task: Task, draft, error) -> None:
        shown = session is self.session
        if error is not None:
            if isinstance(error, SummaryError) and error.kind == "cancelled":
                session.summary_note, session.summary_short = "요약을 취소했습니다.", "요약 취소"
                if shown:
                    self._set_status("요약 취소 · 전사문은 그대로 있습니다", "warn")
            else:
                session.summary_error = (
                    str(error) if isinstance(error, SummaryError) else f"요약 중 예기치 못한 오류: {type(error).__name__}"
                )
                if shown:
                    self._set_status("요약 실패 · 전사문은 그대로 있습니다", "warn")
                if session.via_hotkey:
                    self.overlay.show_status("error", "요약하지 못했습니다", "clinic-scribe 화면을 확인하세요", 20)
        else:
            session.draft = draft
            session.draft_run_id = job.run_id
            session.summary_seconds = task.elapsed
            session.summary_stale = False
            if shown:
                self._set_status(f"완료 · {draft.kind_label} 준비됨", "done")
            self._deliver_hotkey_summary(session)
        if shown:
            self._show_draft()

    def _show_draft(self) -> None:
        s = self.session
        if s is not None and s.summary_error:
            self.summary_error_label.setText(f"요약 실패: {s.summary_error}\n전사문은 그대로 있습니다. '요약 다시 시도'로 같은 전사문을 다시 요약할 수 있습니다.")
            self.summary_error_label.show()
            self.simple.set_error(f"요약 실패: {s.summary_error.splitlines()[0]}  (대화 내용은 남아 있습니다)", warn=True)
        else:
            self.summary_error_label.hide()
            if self.simple.error.text().startswith("요약 실패"):
                self.simple.set_error(None)
        if s is not None and s.draft is not None:
            run = s.run(s.draft_run_id)
            basis = f"요약 기준: 진료 #{s.session_id} · {run.label if run else s.draft.based_on} · {s.draft.kind_label}"
            if isinstance(s.draft, ConversationSummary) != (self._summary_mode() == "general"):
                basis += " — 요약 종류를 바꿨습니다. '요약 다시 시도'를 누르면 같은 전사문으로 다시 요약합니다."
            self.draft_basis_label.setText(basis)
            content = s.draft.to_html(include_evidence=self.evidence_check.isChecked())
            self.draft_view.setHtml(content)
            self.simple.summary_view.setHtml(content)
        else:
            self.draft_basis_label.setText("")
            if self._task(s, "summary") is not None:
                message = f"요약 중… {self._engine_name()} 응답을 기다리고 있습니다. 전사문은 왼쪽에서 바로 확인할 수 있습니다."
            elif s is not None and s.summary_note:
                message = s.summary_note
            else:
                message = DRAFT_PLACEHOLDER
            placeholder = f'<p style="color:#8b93a1;">{html.escape(message)}</p>'
            self.draft_view.setHtml(placeholder)
            self.simple.summary_view.setHtml(placeholder)
        self._refresh_controls()

    def _on_copy(self) -> None:
        if self.session is None or self.session.draft is None:
            return
        QGuiApplication.clipboard().setText(self.session.draft.to_text(include_evidence=self.evidence_check.isChecked()))
        self._set_status(f"진료 #{self.session.session_id} 요약을 클립보드에 복사했습니다", "done")

    def _on_export(self) -> None:
        s = self.session
        if s is None or not s.has_results:
            return
        default = Path.home() / "Documents" / f"clinic-scribe-{s.created_at:%Y%m%d-%H%M}-{s.session_id}.md"
        path, _ = QFileDialog.getSaveFileName(self, "결과 내보내기", str(default), "Markdown (*.md)")
        if not path:
            return
        has_audio = s.pcm16 is not None or s.stored_audio
        include_audio = has_audio and self._confirm("녹음 저장", "녹음(16 kHz WAV)도 같은 이름으로 저장할까요?")
        if include_audio:
            self._session_audio(s)
        run = s.run(s.draft_run_id) or self._current_run()
        try:
            written = export_session(Path(path), s, run, s.draft, include_audio, self.evidence_check.isChecked())
        except OSError as exc:
            self._show_error(f"내보내기 실패: {exc}")
            return
        s.exported = True
        self._persist(s)
        self._set_status("내보냄: " + ", ".join(p.name for p in written), "done")

    # ------------------------------------------------------------ 결과 표시
    def _show_run(self, run: TranscriptRun | None) -> None:
        content = self._transcript_html(run)
        self.transcript_view.setHtml(content)
        self.simple.transcript_view.setHtml(content)
        self._fill_regions(run)
        if not self.recorder.active:
            if run is not None:
                self.timeline.set_data(run.result.audio_duration, [(r.start, r.end) for r in run.result.regions or []], live=False)
            else:
                self.timeline.clear()

    def _transcript_html(self, run: TranscriptRun | None) -> str:
        if run is None:
            if self._task(self.session, "transcribe") is not None:
                return '<p style="color:#8b93a1;">전사 중…</p>'
            if self.session is not None and self.session is self._recording_session and self.recorder.active:
                return '<p style="color:#8b93a1;">녹음 중입니다. 진료가 끝나면 이 진료 녹음 전체를 전사해 여기에 보여 줍니다.</p>'
            if self.session is not None and self.session.transcribe_error:
                return f'<p style="color:#fdba74;">{html.escape(self.session.transcribe_error)}</p>'
            return '<p style="color:#8b93a1;">녹음을 마치거나 오디오 파일을 불러오면 타임스탬프가 있는 전사문이 표시됩니다.</p>'
        r = run.result
        vocab = f" · {html.escape(r.doctor)} 용어 목록 사용" if r.doctor else ""
        head = (f'<p style="color:#8b93a1;">{html.escape(run.label)} · 녹음 {format_time(r.audio_duration)} · '
                f"처리 {r.elapsed:.1f}초 · 시각은 {self._time_basis()}{vocab}</p>")
        if not r.segments:
            note = "음성이 감지되지 않았습니다." if r.no_speech_detected else "전사된 발화가 없습니다."
            return head + f'<p style="color:#8b93a1;">{note}</p>'
        hint = ('<p style="color:#6f7787;">틀린 말은 드래그해 고른 뒤 오른쪽 클릭 → \'선택한 말 고쳐 쓰기\'. '
                "고친 말은 딕션 사전에 쌓여 다음 진료부터 자동으로 고쳐집니다.</p>")
        rows = [head, hint]
        for seg in r.segments:
            stamp = f"[{format_time(seg.start)} → {format_time(seg.end)}]"
            text = html.escape(seg.text)
            if not seg.speech_detected:
                rows.append(
                    f'<p style="background:#3a2410;color:#fdba74;">{stamp} ⚠ 음성이 감지되지 않은 곳의 전사(환각 의심): {text}</p>'
                )
            else:
                who = ""
                if seg.speaker:  # 등록한 의사 목소리와 비교한 결과 (틀릴 수 있음)
                    color = "#fbbf24" if seg.speaker == PATIENT else "#a5b4fc"
                    who = f'<span style="color:{color};font-weight:600;">{html.escape(seg.speaker)}</span>&nbsp; '
                fixed = ""
                if seg.corrections:  # 딕션 사전으로 고친 말 (원래 들린 말을 함께 보여 준다)
                    fixed = f'&nbsp;<span style="color:#6f7787;">(고침: {html.escape(", ".join(seg.corrections))})</span>'
                rows.append(f'<p><span style="color:#8b93a1;font-family:Menlo,Consolas,DejaVu Sans Mono,monospace;">{stamp}</span>&nbsp; {who}{text}{fixed}</p>')
        return "".join(rows)

    def _time_basis(self) -> str:
        if self.session is not None and self.session.audio_trimmed:
            return "무음을 줄여 저장한 녹음 기준"
        return "이 진료 녹음 기준"

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
                    f"Silero VAD 음성 구간 {len(r.regions)}개 · {p.describe()} · 시각은 {self._time_basis()}. "
                    f"{p.min_silence_ms}ms 보다 짧은 쉼은 한 구간으로 합쳐집니다. 구간을 재생해 첫소리·끝소리가 잘렸는지 확인하세요."
                )
            else:
                self.regions_info.setText(
                    "이 전사는 VAD 없이 녹음 전체를 처리했습니다. 아래 구간은 같은 설정의 Silero VAD 로 찾은 음성 구간(비교용)이며, "
                    "⚠ 줄은 음성이 없는 곳에서 나온 전사입니다."
                )
            rows = [(str(i + 1), reg.start, reg.end, r.region_text(i), False) for i, reg in enumerate(r.regions)]
            rows += [("⚠", seg.start, seg.end, seg.text, True) for seg in r.segments if seg.region is None]
            rows.sort(key=lambda row: row[1])
        warn_bg = QColor("#3a2410")
        for label, start, end, text, suspicious in rows:
            row = table.rowCount()
            table.insertRow(row)
            values = (label, format_time(start), format_time(end), f"{end - start:.1f}초", text or "(전사 없음)")
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, (start, end))
                if suspicious:
                    item.setBackground(warn_bg)
                    item.setForeground(QColor("#fdba74"))
                table.setItem(row, col, item)
        table.resizeRowsToContents()

    def _refresh_compare(self, *_args) -> None:
        s = self.session
        for combo, view in ((self.compare_a, self.compare_view_a), (self.compare_b, self.compare_view_b)):
            run = s.run(combo.currentData()) if s else None
            view.setHtml(self._transcript_html(run) if run else '<p style="color:#8b93a1;">같은 녹음을 VAD 설정을 바꿔 다시 전사하면 여기서 나란히 비교할 수 있습니다.</p>')

    def _play_selected_region(self) -> None:
        s = self.session
        items = self.regions_table.selectedItems()
        if s is None or not items or self.recorder.active or self._session_audio(s) is None:
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

    # -------------------------------------------------------------- 설정
    # -------------------------------------------------------------- 요약 방식 · 설치
    def _local_engine_selected(self) -> bool:
        return self.settings.summary_engine == "local"

    def _engine_name(self) -> str:
        return "로컬 모델" if self._local_engine_selected() else "Claude"

    def _selected_local_model(self) -> installer.LocalModel:
        return installer.local_model(self.settings.local_model) or installer.local_model(installer.DEFAULT_LOCAL_MODEL)

    def _local_engine(self, model: installer.LocalModel) -> LocalLLM:
        """고른 로컬 모델의 서버 (모델을 바꾸면 전 서버를 끄고 새로 띄운다)."""
        path = installer.local_model_path(model)
        if self._local_llm is not None and self._local_llm.model != path:
            self._local_llm.close()
            self._local_llm = None
        if self._local_llm is None:
            self._local_llm = LocalLLM(installer.llama_server_path(), path, self.settings.whisper_threads)
        return self._local_llm

    def _refresh_engine_texts(self) -> None:
        engine = "local" if self._local_engine_selected() else "claude"
        self.simple.note.setText(FOOTNOTES[engine])
        self.privacy_label.setText(PRIVACY_NOTES[engine])

    def check_setup(self) -> None:
        """앱을 열 때(app.py): whisper·모델이 없거나 요약 방식을 아직 고르지 않았으면 '설치' 창을 띄운다."""
        if self.interactive and SetupDialog.needed(self.settings):
            self._open_setup()

    def _open_setup(self) -> None:
        if self.recorder.active:
            return
        dialog = SetupDialog(self.settings, self)
        dialog.setStyleSheet(theme.stylesheet("SetupDialog"))
        dialog.exec()
        try:
            self.settings.save()
        except OSError:
            pass
        if self._local_llm is not None and not self._local_engine_selected():
            self._local_llm.close()  # Claude 로 바꿨으면 로컬 모델이 쓰던 메모리를 돌려준다
            self._local_llm = None
        self._speaker_encoder_cache, self._speaker_issue = None, None
        self._live_runner, self._live_issue = None, None
        self._refresh_tool_status()
        self._refresh_engine_texts()
        self._refresh_controls()

    # -------------------------------------------------------------- 녹음 버튼 (전역 단축키)
    def _apply_hotkey(self) -> None:
        """설정대로 녹음 버튼을 (다시) 등록한다. 쓸 수 없는 키면 이유를 hotkey_issue 에 두고 스페이스바만 쓴다."""
        if self._hotkey is not None:
            self._hotkey.stop()
            self._hotkey = None
        self.hotkey_issue = ""
        if not self.settings.hotkey_enabled or os.environ.get("CLINIC_SCRIBE_NO_HOTKEY"):
            self._refresh_hint()
            return
        try:
            hotkey = GlobalHotkey(self.settings.record_hotkey, self.hotkey_pressed.emit)
            hotkey.start()
        except (HotkeyError, ValueError) as exc:
            self.hotkey_issue = str(exc)
        else:
            self._hotkey = hotkey
        self._refresh_hint()

    def _hotkey_label(self) -> str:
        try:
            return describe_key(self.settings.record_hotkey)
        except ValueError:
            return self.settings.record_hotkey

    def _refresh_hint(self) -> None:
        active = self._hotkey is not None
        self.simple.hint.setText(f"녹음 시작 · 종료  ·  버튼 {self._hotkey_label()}" if active else "녹음 시작 · 종료")

    def _on_hotkey(self) -> None:
        """녹음 버튼: 녹음 중이 아니면 시작, 녹음 중이면 끝내고 전사·요약. 버튼이 튀어 두 번 눌린 것(0.8초 안)은 한 번으로."""
        now = time.monotonic()
        if now - self._last_hotkey < 0.8:
            return
        self._last_hotkey = now
        sounds = self.settings.hotkey_sounds
        if self.recorder.active:
            session = self._recording_session
            if session is not None:
                session.via_hotkey = True
                self._hotkey_session = session
            self._on_record_clicked()
            if sounds:
                play_cue("stop")
        else:
            self._on_record_clicked()
            if self.recorder.active and self._recording_session is not None:
                self._recording_session.via_hotkey = True
                self._hotkey_session = self._recording_session
                if sounds:
                    play_cue("start")
            else:
                if sounds:
                    play_cue("error")
                self.overlay.show_status("error", "녹음을 시작하지 못했습니다", "clinic-scribe 화면을 확인하세요", 12)
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        """화면 구석 표시창: 녹음 중이거나 버튼으로 녹음한 진료를 처리하는 동안 (clinic-scribe 창을 보고 있으면 숨는다)."""
        overlay = getattr(self, "overlay", None)
        if overlay is None:
            return
        overlay.set_allowed(QApplication.applicationState() != Qt.ApplicationState.ApplicationActive)
        if self.recorder.active and self._recording_started is not None:
            key = f"{self._hotkey_label()} 버튼을 다시 누르면 끝" if self._hotkey is not None else "Space 를 누르면 끝"
            rec = self._recording_session
            overlay.show_recording(self._recording_started, f"{rec.patient} · {key}" if rec is not None and rec.patient else key)
            return
        inserting = next((sess for task, (sess, _copied) in self._emr_jobs.items() if task.job.kind == "emr_insert"), None)
        if inserting is not None:
            overlay.show_status("busy", "씨차트에 넣는 중…", inserting.patient or f"진료 #{inserting.session_id}")
            return
        sess = self._hotkey_session
        if sess is not None and self._session_by_id(sess.session_id) is sess:
            name = sess.patient or f"진료 #{sess.session_id}"
            if self._task(sess, "transcribe"):
                overlay.show_status("busy", "전사 중…", name)
                return
            if self._task(sess, "summary"):
                overlay.show_status("busy", "요약 중…", name)
                return
        overlay.clear()

    def _bring_to_front(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _deliver_hotkey_summary(self, session: Session) -> None:
        """버튼으로 녹음한 진료의 요약이 끝났다: 클립보드에 복사하고, 씨차트 연결을 정했으면 진료 기록 칸 끝에 넣는다.

        녹음하는 동안 씨차트에서 읽은 환자와 지금 씨차트에 열린 환자가 같을 때만 넣고, 씨차트의 저장은 누르지 않는다.
        """
        if not session.via_hotkey or session.draft is None:
            return
        text = session.draft.to_text(include_evidence=self.evidence_check.isChecked())
        copied = self.settings.auto_copy_summary
        if copied:
            QGuiApplication.clipboard().setText(text)
        name = session.patient or f"진료 #{session.session_id}"
        names = emr_link.usable_names(self.settings.emr_name_places)
        field = emr_link.usable_field(self.settings.emr_note_place)
        if self.settings.emr_insert_summary and emr_link.supported() and names and field is not None:
            expected = emr_link.agreed_name(session.emr_reads)
            if not expected:
                reason = ("녹음하는 동안 씨차트 환자가 바뀌었습니다" if session.emr_reads
                          else "녹음할 때 씨차트에서 환자 이름을 읽지 못했습니다")
                self._emr_not_inserted(session, copied, reason)
                return
            task = self.emr.start(Job("emr_insert", session.session_id, 0),
                                  lambda cancel: emr_link.run_insert(names, field, text, expected, cancel))
            self._emr_jobs[task] = (session, copied)
            self._refresh_overlay()
            return
        if copied:
            self.overlay.show_status("done", "요약을 복사했습니다", f"{name} · 씨차트에서 Ctrl+V", 20)
            if self.settings.hotkey_sounds:
                play_cue("done")

    # -------------------------------------------------------------- 씨차트 연결 (emr_link)
    def _read_emr_patient(self, session: Session) -> None:
        """씨차트에 지금 열린 환자 이름을 읽어 이 진료에 붙인다 (씨차트 연결을 정했을 때. 따로 띄운 프로세스에서)."""
        places = emr_link.usable_names(self.settings.emr_name_places) if emr_link.supported() else []
        if places:
            task = self.emr.start(Job("emr_read", session.session_id, 0), lambda cancel: emr_link.run_read(places, cancel))
            self._emr_jobs[task] = (session, False)

    def _on_emr_finished(self, task: Task, result, error) -> None:
        job = self._emr_jobs.pop(task, None)
        if job is None:  # 앱을 닫는 중
            return
        session, copied = job
        if task.job.kind == "emr_read":
            self._on_emr_read(session, result if error is None and isinstance(result, str) else "")
        elif error is not None:
            self._on_emr_inserted(session, copied, {"status": "error", "error": str(error)})
        else:
            self._on_emr_inserted(session, copied, result)

    def _on_emr_read(self, session: Session, name: str) -> None:
        if not name or self._session_by_id(session.session_id) is not session:
            return
        earlier = session.emr_reads[-1] if session.emr_reads else ""
        session.emr_reads.append(name)
        # 비어 있거나 앞서 씨차트에서 읽은 이름이면 새로 읽은 이름으로 (직접 적은 이름은 그대로 둔다)
        if not session.patient or (earlier and emr_link.same_patient(session.patient, earlier)):
            if not emr_link.same_patient(session.patient, name):
                self.set_patient(session, name)
        self._refresh_overlay()

    def _on_emr_inserted(self, session: Session, copied: bool, result: dict) -> None:
        if result.get("status") != "ok":
            reason = emr_link.REASONS.get(result.get("status"), emr_link.REASONS["error"])
            self._emr_not_inserted(session, copied, reason, result.get("error", ""))
            return
        name = session.patient or f"진료 #{session.session_id}"
        self.overlay.show_status("done", "씨차트에 넣었습니다", f"{name} · 읽고 고친 뒤 씨차트에서 저장하세요", 30)
        self._set_status(f"진료 #{session.session_id} 요약을 씨차트 진료 기록 칸에 넣었습니다 (저장은 씨차트에서)", "done")
        if self.settings.hotkey_sounds:
            play_cue("done")

    def _emr_not_inserted(self, session: Session, copied: bool, reason: str, detail: str = "") -> None:
        backup = "요약은 복사해 두었습니다 (Ctrl+V)" if copied else "요약은 clinic-scribe 에 있습니다"
        self.overlay.show_status("warn", "씨차트에 넣지 않았습니다", f"{reason}\n{backup}", 30)
        self._set_status(f"진료 #{session.session_id} 요약을 씨차트에 넣지 않았습니다: {reason}"
                         + (f" ({detail})" if detail else ""), "warn")
        if self.settings.hotkey_sounds:
            play_cue("error")

    # -------------------------------------------------------------- 업데이트
    def _refresh_update_chip(self) -> None:
        chip = self.updates.chip()
        for button in (self.update_btn, self.simple.update_btn):
            set_update_chip(button, chip)

    def _can_restart(self) -> bool:
        return not self.recorder.active and not self._tasks

    def _open_updates(self) -> None:
        dialog = UpdateDialog(self.updates, self._can_restart, self)
        dialog.setStyleSheet(theme.stylesheet("UpdateDialog"))
        dialog.restart_requested.connect(self._restart_for_update)
        dialog.exec()
        self.updates.acknowledge()

    def _restart_for_update(self) -> None:
        """'다시 열어 적용': 새 창을 띄우고 이 창을 닫는다. 새 창은 이 창이 닫히기를 기다렸다가 새 버전을 적용하고 연다."""
        if not self._can_restart():
            return
        try:
            updater.request_restart(apply=True)
        except OSError as exc:
            QMessageBox.warning(self, "업데이트", f"앱을 다시 열지 못했습니다: {exc}\n앱을 닫았다가 직접 다시 열면 적용됩니다.")
            return
        self.close()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self, hotkey_issue=self.hotkey_issue)
        dialog.setStyleSheet(theme.stylesheet("SettingsDialog"))
        if self._hotkey is not None:  # 설정 창에서 버튼을 눌러 키를 정할 수 있게 잠시 푼다 (닫으면 다시 등록)
            self._hotkey.stop()
            self._hotkey = None
        dialog.setup_requested.connect(lambda: (dialog.reject(), QTimer.singleShot(0, self._open_setup)))
        dialog.update_requested.connect(lambda: (dialog.reject(), QTimer.singleShot(0, self._open_updates)))
        if dialog.exec():
            dialog.apply()
            self.settings.save()
            self._live_runner = None  # 바뀐 경로로 다시 만든다
            self._live_issue = None
            self._speaker_encoder_cache = None
            self.summary_mode_combo.setCurrentIndex(max(0, self.summary_mode_combo.findData(self.settings.summary_mode)))
            self._refresh_tool_status()
            self._refresh_auto_split_note()
        self._apply_hotkey()

    # -------------------------------------------------------------- 종료
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        # 끝난 진료는 저장되어 있으므로 묻지 않는다. 녹음 중이거나 처리 중일 때만 묻는다.
        busy = len({sid for sid, _kind in self._tasks})
        question = ""
        if self.recorder.active:
            question = "녹음 중입니다. 지금 종료하면 이번 녹음은 사라집니다. 종료할까요?"
        elif busy:
            question = (f"전사·요약 중인 진료가 {busy}건 있습니다. 지금 종료하면 처리가 멈춥니다. 녹음·전사문은 저장되어 있어 "
                        "다음에 열면 '다시 시도'로 이어서 처리할 수 있습니다. 종료할까요?")
        if question and not self._confirm("종료", question):
            event.ignore()
            return
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        """마이크를 놓고, 진행 중인 외부 프로세스를 끝내고, 진료 기록과 설정을 저장한다."""
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self._hotkey is not None:
            self._hotkey.stop()
            self._hotkey = None
        self._emr_jobs.clear()  # 씨차트 프로세스는 아래 terminate_all 이 끝낸다
        self.overlay.hide()
        self._rec_timer.stop()
        if self.recorder.active:
            self.recorder.abort()
        self.live_transcriber.close()
        # 처리 중이던 진료는 '중단됨'으로 저장해 다음에 열면 '다시 시도'로 이어서 처리할 수 있게 한다
        interrupted = []
        for (session_id, kind), task in self._tasks.items():
            task.cancel.set()
            session = self._session_by_id(session_id)
            if session is None:
                continue
            if kind == "transcribe" and not session.runs:
                session.transcribe_error = INTERRUPTED["transcribe"]
            elif kind == "summary":
                session.summary_error = INTERRUPTED["summary"]
            interrupted.append(session)
        self._tasks.clear()
        for session in interrupted:
            self._persist(session)
        procutil.terminate_all()
        if self._local_llm is not None:
            self._local_llm.close()
        self._stop_playback()
        self.saver.close()
        self.settings.show_evidence = self.evidence_check.isChecked()
        try:
            self.settings.save()
        except OSError:
            pass
        self.session = None
