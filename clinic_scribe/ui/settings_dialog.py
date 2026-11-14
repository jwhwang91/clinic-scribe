"""설정 창: 실행 파일·모델 경로, Whisper 옵션, 요약(Claude Code CLI 또는 로컬 모델) 옵션, 진료 나누기, 말한 사람 표시,
자동 업데이트, 씨차트 연결 점검(Windows)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from clinic_scribe.config import (
    CLAUDE_EFFORT_CHOICES,
    DEFAULT_CLAUDE_MODEL,
    SUMMARY_MODES,
    Settings,
    default_model_dir,
    settings_path,
)
from clinic_scribe import emr_link, emr_probe, hotkey, installer, updater
from clinic_scribe.ui import theme
from clinic_scribe.ui.emr_probe_dialog import EmrProbeDialog
from clinic_scribe.ui.widgets import PathField


class SettingsDialog(QDialog):
    setup_requested = Signal()  # '설치 · 요약 방식…' 을 눌렀다 (메인 창이 설치 창을 연다)
    update_requested = Signal()  # '업데이트 확인…' 을 눌렀다 (메인 창이 업데이트 창을 연다)

    def __init__(self, settings: Settings, parent=None, hotkey_issue: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.settings = settings
        self._hotkey_issue = hotkey_issue  # 메인 창이 지금 키를 등록하지 못한 이유
        self.emr_insert: QCheckBox | None = None  # 씨차트 탭 (Windows)
        s = settings

        whisper_box = QGroupBox("전사 (whisper.cpp, 이 컴퓨터에서 처리)")
        wf = QFormLayout(whisper_box)
        self.whisper_cli = PathField(s.whisper_cli, f"자동: {s.whisper_cli_path() or '찾지 못함'}", "whisper-cli 선택")
        self.whisper_model = PathField(s.whisper_model, str(s.whisper_model_path()), "Whisper 모델 선택")
        self.vad_model = PathField(s.vad_model, str(s.vad_model_path()), "Silero VAD 모델 선택")
        self.language = QLineEdit(s.language)
        self.prompt = QPlainTextEdit(s.whisper_prompt)
        self.prompt.setFixedHeight(64)
        self.prompt.setToolTip("이 문장에 들어 있는 약명·약어일수록 전사문에 영어 원문 표기로 남습니다.")
        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.10, 0.95)
        self.vad_threshold.setSingleStep(0.05)
        self.vad_threshold.setValue(s.vad.threshold)
        wf.addRow("whisper-cli", self.whisper_cli)
        wf.addRow("Whisper 모델", self.whisper_model)
        wf.addRow("VAD 모델", self.vad_model)
        wf.addRow("언어 코드", self.language)
        wf.addRow("전사 힌트 프롬프트", self.prompt)
        wf.addRow("VAD 음성 판정 임계값", self.vad_threshold)

        if s.summary_engine == "local":
            model = installer.local_model(s.local_model)
            engine = f"로컬 모델 {model.label if model else s.local_model} · 이 컴퓨터에서만 (전송 없음)"
        else:
            engine = "Claude Code CLI · 전사문이 Claude 서비스로 전송됨"
        claude_box = QGroupBox(f"요약 ({engine})")
        cf = QFormLayout(claude_box)
        self.setup_btn = QPushButton("설치 · 요약 방식…")
        self.setup_btn.setToolTip("요약을 Claude Code 로 할지 로컬 모델로 할지 고르고, 필요한 프로그램·모델을 받습니다.")
        self.setup_btn.clicked.connect(self.setup_requested)
        cf.addRow("요약 방식", self.setup_btn)
        self.claude_cli = PathField(s.claude_cli, f"자동: {s.claude_cli_path() or '찾지 못함'}", "claude 실행 파일 선택")
        self.claude_model = QLineEdit(s.claude_model)
        self.claude_model.setPlaceholderText(DEFAULT_CLAUDE_MODEL)
        self.claude_effort = QComboBox()
        for choice in CLAUDE_EFFORT_CHOICES:
            self.claude_effort.addItem(choice or "(CLI 기본값)", choice)
        self.claude_effort.setCurrentIndex(max(0, self.claude_effort.findData(s.claude_effort)))
        self.claude_timeout = QSpinBox()
        self.claude_timeout.setRange(30, 1800)
        self.claude_timeout.setSuffix(" 초")
        self.claude_timeout.setValue(s.claude_timeout_s)
        self.auto_summary = QCheckBox("전사가 끝나면 자동으로 요약 요청")
        self.auto_summary.setChecked(s.auto_summarize)
        self.summary_mode = QComboBox()
        for mode, label in SUMMARY_MODES.items():
            self.summary_mode.addItem(label, mode)
        self.summary_mode.setCurrentIndex(max(0, self.summary_mode.findData(s.summary_mode)))
        cf.addRow("claude 실행 파일", self.claude_cli)
        cf.addRow("모델", self.claude_model)
        cf.addRow("effort", self.claude_effort)
        cf.addRow("제한 시간", self.claude_timeout)
        cf.addRow("요약 종류", self.summary_mode)
        cf.addRow("", self.auto_summary)

        split_box = QGroupBox("자동 진료 나누기 (스페이스바로 끝내는 것을 깜빡했을 때)")
        sf = QFormLayout(split_box)
        self.auto_split = QCheckBox("긴 침묵이나 다음 환자 신호에서 진료를 끝내고 다음 진료로 넘어가기 (녹음은 계속)")
        self.auto_split.setChecked(s.auto_split)
        self.split_silence = QSpinBox()
        self.split_silence.setRange(10, 3600)
        self.split_silence.setSuffix(" 초")
        self.split_silence.setValue(s.split_silence_s)
        self.cues_next = QLineEdit(s.split_cues_next)
        self.cues_next.setToolTip("쉼표로 구분. 이 말이 나오면 그 말부터 새 진료로 봅니다. 띄어쓰기는 무시합니다.")
        self.cues_end = QLineEdit(s.split_cues_end)
        self.cues_end.setToolTip("쉼표로 구분. 이 말이 나오면 그 말까지를 앞 진료로 봅니다. 띄어쓰기는 무시합니다.")
        for edit in (self.cues_next, self.cues_end):
            edit.setMinimumWidth(460)  # 신호 목록 전체가 한눈에 보이게
        self.min_session = QSpinBox()
        self.min_session.setRange(0, 600)
        self.min_session.setSuffix(" 초")
        self.min_session.setValue(s.min_session_s)
        self.auto_stop = QSpinBox()
        self.auto_stop.setRange(0, 7200)
        self.auto_stop.setSuffix(" 초")
        self.auto_stop.setSpecialValueText("끄기")
        self.auto_stop.setValue(s.auto_stop_idle_s)
        sf.addRow("", self.auto_split)
        sf.addRow("침묵 기준", self.split_silence)
        sf.addRow("다음 환자 신호", self.cues_next)
        sf.addRow("끝인사", self.cues_end)
        self.split_doctor_only = QCheckBox("신호 단어는 등록한 의사 목소리일 때만 진료를 나누기 (의사 목소리를 등록했을 때)")
        self.split_doctor_only.setChecked(s.split_doctor_only)
        sf.addRow("", self.split_doctor_only)
        sf.addRow("최소 진료 길이", self.min_session)
        sf.addRow("녹음 자동 멈춤 (말 없음)", self.auto_stop)

        speaker_box = QGroupBox("말한 사람 표시 (등록한 의사 목소리와 비교, 이 컴퓨터 에서 처리)")
        pf = QFormLayout(speaker_box)
        self.speaker_model = PathField(s.speaker_model, str(s.speaker_model_path()), "화자 인식 모델 선택")
        self.speaker_threshold = QDoubleSpinBox()
        self.speaker_threshold.setRange(0.30, 0.90)
        self.speaker_threshold.setSingleStep(0.05)
        self.speaker_threshold.setValue(s.speaker_threshold)
        self.speaker_threshold.setToolTip(
            "등록한 목소리와 이 값 이상 비슷하면 그 의사로 봅니다.\n"
            "의사 말이 '환자'로 나오면 낮추고, 환자 말이 의사로 나오면 높이세요."
        )
        pf.addRow("화자 인식 모델", self.speaker_model)
        pf.addRow("같은 사람으로 볼 유사도", self.speaker_threshold)
        speaker_note = QLabel("의사 목소리 등록·삭제는 위쪽 '의사 목소리' 버튼에서 합니다.")
        speaker_note.setObjectName("subtle")
        pf.addRow("", speaker_note)

        info = QLabel(f"설정 파일: {settings_path()}\n모델 기본 폴더: {default_model_dir()}")
        info.setObjectName("subtle")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        pages = [(whisper_box, "전사"), (claude_box, "요약"), (split_box, "진료 나누기"), (speaker_box, "말한 사람"),
                 (self._button_box(), "녹음 버튼"), (self._update_box(), "업데이트")]
        if emr_probe.supported():  # 씨차트는 Windows 프로그램
            pages.append((self._emr_box(), "씨차트"))
        # 항목이 많아 작은 화면에서도 잘리지 않게 탭으로 나눈다.
        tabs = QTabWidget()
        for box, title in pages:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.addWidget(box)
            page_layout.addStretch(1)
            tabs.addTab(page, title)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addWidget(info)
        layout.addWidget(buttons)

    def _button_box(self) -> QGroupBox:
        s = self.settings
        box = QGroupBox("녹음 버튼 (씨차트를 보면서 버튼 하나로 녹음)")
        form = QFormLayout(box)
        self.hotkey_enabled = QCheckBox("버튼(단축키)으로 녹음 시작·종료 — 씨차트 등 다른 프로그램이 앞에 있어도 동작")
        self.hotkey_enabled.setChecked(s.hotkey_enabled)
        form.addRow("", self.hotkey_enabled)
        key_row = QHBoxLayout()
        self.hotkey_edit = QKeySequenceEdit(QKeySequence(s.record_hotkey))
        self.hotkey_edit.setMaximumSequenceLength(1)
        self.hotkey_edit.setToolTip("이 칸을 누르고 USB 버튼(또는 정할 키)을 누르면 그 키로 정합니다")
        self.hotkey_edit.keySequenceChanged.connect(lambda _seq: self._check_hotkey())
        key_row.addWidget(self.hotkey_edit, 1)
        default_key = QPushButton(f"기본값 ({hotkey.DEFAULT_KEY})")
        default_key.clicked.connect(lambda: self.hotkey_edit.setKeySequence(QKeySequence(hotkey.DEFAULT_KEY)))
        key_row.addWidget(default_key)
        form.addRow("버튼 키", key_row)
        self.hotkey_status = QLabel()
        self.hotkey_status.setWordWrap(True)
        form.addRow("", self.hotkey_status)
        self.hotkey_sounds = QCheckBox("버튼으로 시작·종료할 때 소리 내기 (화면을 보지 않아도 눌렸는지 알게)")
        self.hotkey_sounds.setChecked(s.hotkey_sounds)
        form.addRow("", self.hotkey_sounds)
        self.auto_copy = QCheckBox("버튼으로 녹음한 진료는 요약이 끝나면 클립보드에 복사 (씨차트 진료 기록 칸에서 Ctrl+V)")
        self.auto_copy.setChecked(s.auto_copy_summary)
        form.addRow("", self.auto_copy)
        guide = QLabel(
            "USB 버튼·풋페달은 보통 전용 프로그램으로 누를 때 보낼 키를 정합니다. F13 처럼 키보드에 없는 키로 정하면 다른 "
            "프로그램과 겹치지 않습니다. 버튼으로 녹음하는 동안 화면 오른쪽 위에 작은 표시창이 떠 녹음·요약 상태를 알려 줍니다."
        )
        guide.setObjectName("subtle")
        guide.setWordWrap(True)
        form.addRow("", guide)
        self._check_hotkey()
        return box

    def _hotkey_text(self) -> str:
        return self.hotkey_edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)

    def _check_hotkey(self) -> bool:
        """정한 키를 쓸 수 있는지 바로 보여 준다."""
        text = self._hotkey_text()
        try:
            hotkey.parse(text)
        except ValueError as exc:
            self.hotkey_status.setText(str(exc))
            self.hotkey_status.setStyleSheet("color: #f87171; font-size: 12px;")
            return False
        issue = self._hotkey_issue if text == self.settings.record_hotkey else ""
        self.hotkey_status.setText(issue or f"버튼을 누르면 이 키({hotkey.describe(text)})가 들어오게 정해 두세요.")
        self.hotkey_status.setStyleSheet(f"color: {'#fdba74' if issue else '#8b93a1'}; font-size: 12px;")
        return True

    def _update_box(self) -> QGroupBox:
        box = QGroupBox("업데이트 (GitHub 의 새 버전)")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)
        current = updater.version_of("HEAD")
        version = QLabel(f"지금 버전: {current.describe()} — {current.subject}" if current else "지금 버전: 알 수 없음")
        version.setWordWrap(True)
        layout.addWidget(version)
        self.auto_update = QCheckBox("새 버전을 자동으로 받기 (테스트를 모두 통과한 버전만, 앱을 다시 열 때 적용)")
        self.auto_update.setChecked(self.settings.auto_update)
        layout.addWidget(self.auto_update)
        blocked = updater.blocker()
        note = QLabel(blocked or "앱이 열려 있으면 30분마다 확인합니다. 녹음·처리 중에는 버전을 바꾸지 않고, 새 버전이 열리지 않으면 "
                                 "이전 버전으로 되돌립니다. 진료 기록·설정·의사 목소리는 그대로 남습니다.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        check = QPushButton("업데이트 확인…")
        check.clicked.connect(self.update_requested)
        layout.addWidget(check, 0, Qt.AlignmentFlag.AlignLeft)
        return box

    def _emr_box(self) -> QGroupBox:
        box = QGroupBox("씨차트 연결")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)
        note = QLabel(
            "녹음을 시작하면 씨차트에 열린 환자의 이름을 읽어 진료 기록에 붙이고, 버튼으로 녹음한 진료는 요약이 끝나면 씨차트 "
            "진료 기록 칸 끝에 넣습니다. 넣기 전에 같은 환자인지 확인하고, 이미 적힌 내용은 건드리지 않으며, 저장은 누르지 "
            "않습니다 (읽고 고친 뒤 직접 저장). 환자 이름은 이 컴퓨터 밖으로 보내지 않습니다."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.emr_name_status = QLabel()
        self.emr_name_status.setWordWrap(True)
        form.addRow("환자 이름 자리", self.emr_name_status)
        self.emr_field_status = QLabel()
        self.emr_field_status.setWordWrap(True)
        form.addRow("진료 기록 칸", self.emr_field_status)
        layout.addLayout(form)
        self.emr_insert = QCheckBox("버튼으로 녹음한 진료는 요약을 씨차트 진료 기록 칸 끝에 넣기 (같은 환자일 때만)")
        self.emr_insert.setChecked(self.settings.emr_insert_summary)
        layout.addWidget(self.emr_insert)
        buttons = QHBoxLayout()
        self.emr_probe_btn = QPushButton("씨차트 연결 점검…")
        self.emr_probe_btn.setObjectName("primary")
        self.emr_probe_btn.clicked.connect(self._open_emr_probe)
        buttons.addWidget(self.emr_probe_btn)
        self.emr_clear_btn = QPushButton("연결 지우기")
        self.emr_clear_btn.setToolTip("정해 둔 환자 이름 자리와 진료 기록 칸을 지웁니다 (다시 점검하면 다시 정해집니다)")
        self.emr_clear_btn.clicked.connect(self._clear_emr)
        buttons.addWidget(self.emr_clear_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._refresh_emr()
        return box

    def _refresh_emr(self) -> None:
        names = emr_link.usable_names(self.settings.emr_name_places)
        field = emr_link.usable_field(self.settings.emr_note_place)
        for label, text, ready in (
            (self.emr_name_status, (emr_link.describe_place(names[0]) + (f" 외 {len(names) - 1}곳" if len(names) > 1 else "")
                                    if names else "아직 정하지 않음 — 점검에서 두 환자의 이름을 찾으면 정해집니다"), bool(names)),
            (self.emr_field_status, (emr_link.describe_place(field) if field is not None
                                     else "아직 정하지 않음 — 점검 4단계에서 칸을 찾으면 정해집니다"), field is not None),
        ):
            label.setText(f"✓ {text}" if ready else text)
            label.setStyleSheet(f"color: {'#86efac' if ready else '#8b93a1'};")
        self.emr_clear_btn.setEnabled(bool(names) or field is not None)

    def _open_emr_probe(self) -> None:
        dialog = EmrProbeDialog(self, settings=self.settings)
        dialog.setStyleSheet(theme.stylesheet("EmrProbeDialog", EmrProbeDialog.STYLE))
        dialog.exec()
        self._refresh_emr()

    def _clear_emr(self) -> None:
        self.settings.emr_name_places, self.settings.emr_note_place = [], {}
        try:
            self.settings.save()
        except OSError:
            pass  # 메모리의 설정은 지웠다. 창을 닫을 때 다시 저장한다
        self._refresh_emr()

    def apply(self) -> None:
        s = self.settings
        s.whisper_cli = self.whisper_cli.text()
        s.whisper_model = self.whisper_model.text()
        s.vad_model = self.vad_model.text()
        s.language = self.language.text().strip() or "ko"
        s.whisper_prompt = self.prompt.toPlainText().strip()
        s.vad.threshold = round(self.vad_threshold.value(), 2)
        s.claude_cli = self.claude_cli.text()
        s.claude_model = self.claude_model.text().strip() or DEFAULT_CLAUDE_MODEL
        s.claude_effort = self.claude_effort.currentData() or ""
        s.claude_timeout_s = self.claude_timeout.value()
        s.auto_summarize = self.auto_summary.isChecked()
        s.summary_mode = self.summary_mode.currentData() or "general"
        s.auto_split = self.auto_split.isChecked()
        s.split_silence_s = self.split_silence.value()
        s.split_cues_next = self.cues_next.text().strip()
        s.split_cues_end = self.cues_end.text().strip()
        s.min_session_s = self.min_session.value()
        s.auto_stop_idle_s = self.auto_stop.value()
        s.split_doctor_only = self.split_doctor_only.isChecked()
        s.speaker_model = self.speaker_model.text()
        s.speaker_threshold = round(self.speaker_threshold.value(), 2)
        s.auto_update = self.auto_update.isChecked()
        s.hotkey_enabled = self.hotkey_enabled.isChecked()
        if self._check_hotkey():
            s.record_hotkey = self._hotkey_text()
        s.hotkey_sounds = self.hotkey_sounds.isChecked()
        s.auto_copy_summary = self.auto_copy.isChecked()
        if self.emr_insert is not None:
            s.emr_insert_summary = self.emr_insert.isChecked()
