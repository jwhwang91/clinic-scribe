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

from clinic_scribe.config import CLAUDE_EFFORT_CHOICES, DEFAULT_CLAUDE_MODEL, Settings, default_model_dir, settings_path
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
        cf.addRow("claude 실행 파일", self.claude_cli)
        cf.addRow("모델", self.claude_model)
        cf.addRow("effort", self.claude_effort)
        cf.addRow("제한 시간", self.claude_timeout)
        cf.addRow("", self.auto_summary)

        info = QLabel(f"설정 파일: {settings_path()}\n모델 기본 폴더: {default_model_dir()}")
        info.setStyleSheet("color: gray;")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(whisper_box)
        layout.addWidget(claude_box)
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
