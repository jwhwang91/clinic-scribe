"""설정 창: 실행 파일·모델 경로, Whisper 옵션, Claude CLI 옵션."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
)

from clinic_scribe.config import (
    CLAUDE_EFFORT_CHOICES,
    DEFAULT_CLAUDE_MODEL,
    SUMMARY_MODES,
    Settings,
    default_model_dir,
    settings_path,
)
from clinic_scribe.ui.widgets import PathField


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.settings = settings
        s = settings

        whisper_box = QGroupBox("전사 (whisper.cpp, 이 Mac 에서 처리)")
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

        claude_box = QGroupBox("차트 초안 요약 (Claude Code CLI · 전사문이 Claude 서비스로 전송됨)")
        cf = QFormLayout(claude_box)
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
        sf.addRow("최소 진료 길이", self.min_session)
        sf.addRow("녹음 자동 멈춤 (말 없음)", self.auto_stop)

        info = QLabel(f"설정 파일: {settings_path()}\n모델 기본 폴더: {default_model_dir()}")
        info.setStyleSheet("color: gray;")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(whisper_box)
        layout.addWidget(claude_box)
        layout.addWidget(split_box)
        layout.addWidget(info)
        layout.addWidget(buttons)

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
