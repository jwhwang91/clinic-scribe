"""간단 화면 (기본).

큰 녹음 버튼과 경과 시간·상태만 보인다. 대화 내용과 요약은 숨겨 두고 '보기' 버튼을 눌러야 펼쳐진다.
스페이스바로 녹음을 시작·종료한다(키 처리는 MainWindow 가 한다). 이 화면의 버튼은 키보드 포커스를 받지 않아
스페이스바가 항상 녹음 토글로만 동작한다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from clinic_scribe.ui.widgets import LevelMeter

_STATUS_COLORS = {
    "idle": "#374151",
    "recording": "#b91c1c",
    "busy": "#1d4ed8",
    "summary": "#6d28d9",
    "done": "#166534",
    "warn": "#9a3412",
    "error": "#b91c1c",
}


def _centered(widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(widget)
    row.addStretch(1)
    return row


class SimpleView(QWidget):
    record_clicked = Signal()
    retry_clicked = Signal()
    copy_clicked = Signal()
    settings_clicked = Signal()
    developer_clicked = Signal()
    history_selected = Signal(int)  # 진료 번호

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 14, 24, 14)

        top = QHBoxLayout()
        top.addWidget(QLabel("입력"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        self.device_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        top.addWidget(self.device_combo)
        top.addStretch(1)
        self.mic_status = QLabel()
        top.addWidget(self.mic_status)
        self.settings_btn = QPushButton("설정…")
        self.settings_btn.clicked.connect(self.settings_clicked)
        self.developer_btn = QPushButton("개발자 보기")
        self.developer_btn.setToolTip("전사문·음성 구간·VAD 비교·실시간 전사가 모두 보이는 개발자용 대시보드")
        self.developer_btn.clicked.connect(self.developer_clicked)
        for button in (self.settings_btn, self.developer_btn):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            top.addWidget(button)
        root.addLayout(top)

        self.demo_warning = QLabel("데모 입력이 골라져 있습니다. 마이크 소리 대신 데모 오디오 파일이 들어갑니다.")
        self.demo_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.demo_warning.setStyleSheet(
            "color: #92400e; background: #fef3c7; border: 1px solid #f59e0b; padding: 6px; font-weight: bold;"
        )
        self.demo_warning.hide()
        root.addWidget(self.demo_warning)

        root.addStretch(2)
        self.record_btn = QPushButton()
        self.record_btn.setFixedSize(190, 190)
        self.record_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        font = self.record_btn.font()
        font.setPointSize(20)
        font.setBold(True)
        self.record_btn.setFont(font)
        self.record_btn.clicked.connect(self.record_clicked)
        root.addLayout(_centered(self.record_btn))
        self.hint = QLabel("스페이스바로 녹음 시작 · 종료")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet("color: #6b7280; font-size: 14px;")
        root.addWidget(self.hint)
        self.auto_split_note = QLabel()
        self.auto_split_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.auto_split_note.setWordWrap(True)
        self.auto_split_note.setStyleSheet("color: #9ca3af; font-size: 12px;")
        root.addWidget(self.auto_split_note)
        self.elapsed = QLabel("00:00")
        mono = QFont("Menlo")
        mono.setPointSize(34)
        self.elapsed.setFont(mono)
        self.elapsed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.elapsed)
        self.level = LevelMeter()
        self.level.setFixedWidth(300)
        root.addLayout(_centered(self.level))
        self.status = QLabel()
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.error = QLabel()
        self.error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error.setWordWrap(True)
        self.error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error.hide()
        root.addWidget(self.error)
        self.retry_btn = QPushButton("요약 다시 시도")
        self.retry_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.retry_btn.clicked.connect(self.retry_clicked)
        self.retry_btn.hide()
        root.addLayout(_centered(self.retry_btn))
        root.addStretch(1)

        # 진료 기록: 내용은 숨기고 번호·시간·상태만 보여 준다. 고르면 아래 '보기'·'복사'가 그 진료를 가리킨다.
        self.history_title = QLabel("진료 기록 (이번 실행)")
        self.history_title.setStyleSheet("color: #6b7280; font-weight: bold;")
        root.addWidget(self.history_title)
        self.history = QListWidget()
        self.history.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.history.setMaximumHeight(118)
        self.history.itemClicked.connect(lambda item: self.history_selected.emit(int(item.data(Qt.ItemDataRole.UserRole))))
        root.addWidget(self.history)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.show_transcript_btn = QPushButton("대화 내용 보기")
        self.show_summary_btn = QPushButton("요약 보기")
        self.copy_btn = QPushButton("요약 복사")
        self.copy_btn.setToolTip("요약을 클립보드에 복사합니다 (차트에 옮겨 적을 때).")
        self.copy_btn.clicked.connect(self.copy_clicked)
        for button in (self.show_transcript_btn, self.show_summary_btn):
            button.setCheckable(True)
        self.show_transcript_btn.toggled.connect(self._on_toggle_transcript)
        self.show_summary_btn.toggled.connect(self._on_toggle_summary)
        for button in (self.show_transcript_btn, self.show_summary_btn, self.copy_btn):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setMinimumWidth(130)
            buttons.addWidget(button)
        buttons.addStretch(1)
        root.addLayout(buttons)

        panels = QHBoxLayout()
        self.transcript_view = QTextBrowser()
        self.summary_view = QTextBrowser()
        for view in (self.transcript_view, self.summary_view):
            view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            view.setMinimumHeight(300)  # 펼쳤을 때 위쪽 빈 공간보다 내용이 넓게 보이도록
            view.hide()
            panels.addWidget(view, 1)
        root.addLayout(panels, 12)

        self.note = QLabel("녹음·전사는 이 Mac 에서 처리합니다. 요약을 만들 때 전사문이 Claude 서비스로 전송됩니다.")
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note.setStyleSheet("color: #9ca3af; font-size: 11px;")
        root.addWidget(self.note)
        self.set_recording(False, True)
        self.set_results(False, False, False)

    # ---- 상태 -------------------------------------------------------------
    def set_recording(self, recording: bool, enabled: bool) -> None:
        self.record_btn.setText("■\n녹음 종료" if recording else "●\n녹음 시작")
        color = "#dc2626" if recording else "#2563eb"
        self.record_btn.setStyleSheet(
            f"QPushButton {{ background: {color}; color: white; border-radius: 95px; }}"
            "QPushButton:disabled { background: #9ca3af; }"
        )
        self.record_btn.setEnabled(enabled)

    def set_status(self, text: str, kind: str) -> None:
        self.status.setText(text)
        color = _STATUS_COLORS.get(kind, _STATUS_COLORS["idle"])
        self.status.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")

    def set_error(self, message: str | None, warn: bool = False) -> None:
        if not message:
            self.error.clear()
            self.error.hide()
            return
        color = "#9a3412" if warn else "#b91c1c"
        self.error.setStyleSheet(f"color: {color};")
        self.error.setText(message)
        self.error.show()

    def set_history(self, rows: list[tuple[int, str]], selected_id: int | None) -> None:
        """rows: (진료 번호, 한 줄 설명) 최신순."""
        self.history.blockSignals(True)
        self.history.clear()
        for session_id, text in rows:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, session_id)
            self.history.addItem(item)
            if session_id == selected_id:
                item.setSelected(True)
                self.history.setCurrentItem(item)
        self.history.blockSignals(False)
        self.history_title.setVisible(bool(rows))
        self.history.setVisible(bool(rows))

    # ---- 결과 (기본 숨김) ----------------------------------------------------
    def set_results(self, has_transcript: bool, has_summary: bool, can_copy: bool) -> None:
        self.show_transcript_btn.setEnabled(has_transcript)
        self.show_summary_btn.setEnabled(has_summary)
        self.copy_btn.setEnabled(can_copy)
        if not has_transcript:
            self.show_transcript_btn.setChecked(False)
        if not has_summary:
            self.show_summary_btn.setChecked(False)

    def collapse_results(self) -> None:
        self.show_transcript_btn.setChecked(False)
        self.show_summary_btn.setChecked(False)

    def _on_toggle_transcript(self, checked: bool) -> None:
        self.transcript_view.setVisible(checked)
        self.show_transcript_btn.setText("대화 내용 숨기기" if checked else "대화 내용 보기")

    def _on_toggle_summary(self, checked: bool) -> None:
        self.summary_view.setVisible(checked)
        self.show_summary_btn.setText("요약 숨기기" if checked else "요약 보기")
