"""녹음 표시창: 씨차트를 보며 버튼으로 녹음할 때 화면 구석에 작게 떠서 지금 상태를 알린다
(녹음 중 · 요약 중 · 복사됨 · 씨차트에 넣음 · 넣지 않음 · 오류).

키보드 입력을 가져가지 않아 씨차트에 쓰던 글이 끊기지 않고, clinic-scribe 창을 보고 있을 때는 숨는다.
누르면 clinic-scribe 창을 앞으로 가져온다. 화면을 보지 않아도 버튼이 눌렸는지 알도록 시작·끝 소리도 여기서 낸다.
"""

from __future__ import annotations

import time
from datetime import datetime

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLayout, QVBoxLayout, QWidget

from clinic_scribe.ui.theme import STATUS_COLORS

_DOT = {"recording": "#fb7185", "busy": "#7aa2ff", "done": "#4ade80", "warn": "#fbbf24", "error": "#f87171"}
_CUES = {"start": (660, 880), "stop": (880, 660), "done": (990,), "error": (196, 196)}  # 음 높이(Hz)


def play_cue(kind: str) -> None:
    """버튼 소리: start 올라가는 두 음 · stop 내려가는 두 음 · done 짧은 한 음 · error 낮은 두 음. 소리 장치가 없으면 조용히 넘어간다."""
    try:
        import numpy as np
        import sounddevice as sd

        rate = 44100
        parts = []
        for freq in _CUES[kind]:
            t = np.arange(int(0.09 * rate)) / rate
            fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)  # 딸깍 소리 없게 10ms 페이드
            parts += [0.18 * np.sin(2 * np.pi * freq * t) * fade, np.zeros(int(0.03 * rate))]
        sd.play(np.concatenate(parts).astype(np.float32), rate)
    except Exception:
        pass


class RecordOverlay(QWidget):
    clicked = Signal()

    def __init__(self) -> None:
        flags = (Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.WindowDoesNotAcceptFocus)
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("누르면 clinic-scribe 창을 엽니다")
        self.kind = ""  # "" 숨김 · recording · busy · done · warn · error
        self.text = ""
        self.detail = ""
        self.allowed = True  # clinic-scribe 창을 보고 있으면 False (숨는다)
        self._started: datetime | None = None
        self._until = 0.0  # 잠깐 보이는 알림이 사라질 시각 (time.monotonic)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)  # 글 길이에 맞춰 늘고 준다
        self.card = QFrame()
        self.card.setObjectName("overlayCard")
        row = QHBoxLayout(self.card)
        row.setContentsMargins(14, 9, 16, 9)
        row.setSpacing(10)
        self.dot = QLabel("●")
        self.dot.setStyleSheet("font-size: 15px;")
        row.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignVCenter)
        column = QVBoxLayout()
        column.setSpacing(1)
        self.title = QLabel()
        self.title.setStyleSheet("color: #f1f3f7; font-size: 14px; font-weight: 700;")
        self.subtitle = QLabel()
        self.subtitle.setStyleSheet("color: #a3aab8; font-size: 12px;")
        column.addWidget(self.title)
        column.addWidget(self.subtitle)
        row.addLayout(column, 1)
        outer.addWidget(self.card)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

    # ---- 상태 ---------------------------------------------------------------
    def show_recording(self, started: datetime, detail: str) -> None:
        self.kind, self._started, self.detail, self._until = "recording", started, detail, 0.0
        self._render()

    def show_status(self, kind: str, text: str, detail: str = "", seconds: float = 0.0) -> None:
        """seconds 가 있으면 그만큼 보이고 사라진다 (복사됨·오류 알림)."""
        self.kind, self.text, self.detail, self._started = kind, text, detail, None
        self._until = time.monotonic() + seconds if seconds else 0.0
        self._render()

    def clear(self) -> None:
        if self.kind and not self._until:  # 잠깐 보이는 알림은 제 시간까지 둔다
            self.kind = ""
        self._render()

    def set_allowed(self, allowed: bool) -> None:
        self.allowed = allowed
        self._render()

    def _tick(self) -> None:
        if self._until and time.monotonic() >= self._until:
            self.kind, self._until = "", 0.0
        self._render()

    def _render(self) -> None:
        if self.kind == "recording" and self._started is not None:
            seconds = int((datetime.now() - self._started).total_seconds())
            self.text = f"녹음 중  {seconds // 60:02d}:{seconds % 60:02d}"
        if not self.kind:
            self._timer.stop()
            self.hide()
            return
        self._timer.start()
        background, _ = STATUS_COLORS.get(self.kind, STATUS_COLORS["idle"])
        self.card.setStyleSheet(f"#overlayCard {{ background: #161a22; border: 1px solid {background};"
                                f" border-radius: 14px; }}")
        self.dot.setStyleSheet(f"font-size: 15px; color: {_DOT.get(self.kind, '#a3aab8')};")
        self.title.setText(self.text)
        self.subtitle.setText(self.detail)
        self.subtitle.setVisible(bool(self.detail))
        if not self.allowed:
            self.hide()
            return
        self.layout().activate()
        screen = QGuiApplication.primaryScreen()
        if screen is not None:  # 화면 오른쪽 위 (씨차트 내용을 덜 가리는 곳)
            area = screen.availableGeometry()
            self.move(QPoint(area.right() - self.width() - 20, area.top() + 20))
        if not self.isVisible():
            self.show()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(event)
