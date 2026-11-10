"""작은 위젯들: 입력 레벨, 입력 파형, 음성 구간 타임라인, 진행 단계, 경로 입력칸."""

from __future__ import annotations

import math
from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QWidget


class LevelMeter(QWidget):
    """입력 레벨(dBFS) 막대. -60 dBFS 이하는 비어 있고 0 dBFS 에서 가득 찬다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = -120.0
        self.setMinimumSize(200, 16)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_level(self, db: float) -> None:
        self._db = db
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 이름)
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor("#3f3f46"))
        fraction = min(1.0, max(0.0, (self._db + 60.0) / 60.0))
        color = "#22c55e" if self._db < -12 else ("#eab308" if self._db < -3 else "#ef4444")
        painter.fillRect(QRectF(rect.x(), rect.y(), rect.width() * fraction, rect.height()), QColor(color))
        painter.setPen(QColor("#f4f4f5"))
        label = f"{self._db:.0f} dBFS" if self._db > -120 else "입력 없음"
        painter.drawText(rect.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, label)
        painter.end()


def _amp_fraction(peak: float) -> float:
    """진폭 → 0~1 (-60 dBFS ~ 0 dBFS 를 선형으로)."""
    db = 20.0 * math.log10(max(peak, 1e-6))
    return min(1.0, max(0.0, (db + 60.0) / 60.0))


class WaveformView(QWidget):
    """최근 입력 파형. 50ms 마다 최대 진폭을 받아 오른쪽에서 왼쪽으로 흐르게 그린다. 발화로 판정된 부분은 초록색."""

    def __init__(self, seconds: float = 12.0, tick_s: float = 0.05, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._values: deque[tuple[float, bool]] = deque(maxlen=int(seconds / tick_s))
        self.setMinimumSize(320, 64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def push(self, peak: float, speaking: bool) -> None:
        self._values.append((peak, speaking))
        self.update()

    def clear(self) -> None:
        self._values.clear()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor("#18181b"))
        mid = rect.center().y()
        painter.setPen(QColor("#3f3f46"))
        painter.drawLine(rect.left(), mid, rect.right(), mid)
        if not self._values:
            painter.setPen(QColor("#a1a1aa"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "녹음을 시작하면 입력 파형이 여기에 흐릅니다")
            painter.end()
            return
        bar = rect.width() / self._values.maxlen
        half = rect.height() / 2 - 3
        count = len(self._values)
        for i, (peak, speaking) in enumerate(self._values):
            x = rect.right() - (count - i) * bar
            height = max(1.0, _amp_fraction(peak) * half)
            painter.fillRect(QRectF(x, mid - height, max(1.0, bar - 1), 2 * height),
                             QColor("#22c55e" if speaking else "#60a5fa"))
        painter.end()


class TimelineView(QWidget):
    """녹음 전체 시간축과 음성 구간. 녹음 중에는 실시간 감지 클립, 전사 후에는 Silero VAD 구간을 그린다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.clear()

    def clear(self) -> None:
        self.duration = 0.0
        self.regions: list[tuple[float, float]] = []
        self.live = True
        self.cursor: float | None = None
        self.update()

    def set_data(self, duration: float, regions: list[tuple[float, float]], live: bool, cursor: float | None = None) -> None:
        self.duration, self.regions, self.live, self.cursor = duration, regions, live, cursor
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor("#f4f4f5"))
        track = QRectF(rect.left() + 8, rect.top() + 18, rect.width() - 16, 14)
        painter.fillRect(track, QColor("#e4e4e7"))
        caption = "실시간 감지 클립(미리보기)" if self.live else "Silero VAD 음성 구간 (원본 녹음 기준)"
        painter.setPen(QColor("#52525b"))
        painter.drawText(QRectF(rect.left() + 8, rect.top() + 1, rect.width() - 16, 16),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         f"음성 구간 타임라인 · {caption}")
        scale = max(self.duration, 30.0)
        color = QColor("#4ade80" if self.live else "#3b82f6")
        for start, end in self.regions:
            x0 = track.left() + track.width() * max(0.0, start) / scale
            x1 = track.left() + track.width() * min(scale, max(start, end)) / scale
            painter.fillRect(QRectF(x0, track.top(), max(2.0, x1 - x0), track.height()), color)
        step = next((s for s in (5, 10, 30, 60, 120, 300, 600, 1800) if scale / s <= 10), 3600)
        painter.setPen(QColor("#71717a"))
        t = 0
        while t <= scale:
            x = track.left() + track.width() * t / scale
            painter.drawLine(QPointF(x, track.bottom()), QPointF(x, track.bottom() + 3))
            painter.drawText(QRectF(x - 30, track.bottom() + 2, 60, 12), Qt.AlignmentFlag.AlignCenter,
                             f"{int(t) // 60:02d}:{int(t) % 60:02d}")
            t += step
        if self.cursor is not None:
            x = track.left() + track.width() * min(self.cursor, scale) / scale
            painter.setPen(QPen(QColor("#dc2626"), 2))
            painter.drawLine(QPointF(x, track.top() - 3), QPointF(x, track.bottom() + 3))
        painter.end()


class PipelineView(QWidget):
    """진행 단계: ① 녹음 → ② 전사 → ③ 요약 → ④ 검토. 단계마다 상태와 소요 시간을 보여 준다."""

    STEPS = (
        ("record", "① 녹음"),
        ("transcribe", "② 전사 · whisper.cpp (이 Mac)"),
        ("summary", "③ 요약 · Claude (전송)"),
        ("review", "④ 검토"),
    )
    _STYLE = {
        "idle": ("#f4f4f5", "#71717a", "normal", ""),
        "active": ("#dbeafe", "#1d4ed8", "bold", "● "),
        "done": ("#dcfce7", "#166534", "bold", "✓ "),
        "error": ("#fee2e2", "#b91c1c", "bold", "✗ "),
        "skipped": ("#f4f4f5", "#71717a", "normal", "– "),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._labels: dict[str, QLabel] = {}
        self.states: dict[str, str] = {}
        for i, (key, _title) in enumerate(self.STEPS):
            if i:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #a1a1aa; font-size: 16px;")
                layout.addWidget(arrow)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(44)
            layout.addWidget(label, 1)
            self._labels[key] = label
        self.reset()

    def reset(self) -> None:
        for key, _title in self.STEPS:
            self.set_step(key, "idle")

    def set_step(self, key: str, state: str, detail: str = "") -> None:
        title = dict(self.STEPS)[key]
        bg, fg, weight, icon = self._STYLE[state]
        self.states[key] = state
        label = self._labels[key]
        label.setText(f"{icon}{title}" + (f"\n{detail}" if detail else ""))
        label.setStyleSheet(
            f"background: {bg}; color: {fg}; font-weight: {weight}; border-radius: 6px; padding: 4px 8px;"
        )


class PathField(QWidget):
    """경로 입력칸 + 찾아보기 버튼. 비워 두면 자동 탐색값(placeholder)을 쓴다."""

    def __init__(self, value: str, placeholder: str, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.edit = QLineEdit(value)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setMinimumWidth(420)
        browse = QPushButton("찾아보기…")
        browse.clicked.connect(lambda: self._browse(caption))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)
        layout.addWidget(browse)

    def _browse(self, caption: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, caption, self.edit.text() or self.edit.placeholderText())
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()
