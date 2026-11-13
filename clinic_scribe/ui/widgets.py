"""작은 위젯들: 앱 표시, 입력 레벨, 입력 파형, 음성 구간 타임라인, 진행 단계, 경로 입력칸, 업데이트 단추."""

from __future__ import annotations

import math
from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QWidget

from clinic_scribe.ui import theme


def commit_typing(target: QWidget | None = None) -> None:
    """입력기에서 아직 조합 중인 글자(한글 마지막 글자 등)를 입력란에 확정한다.

    키보드 포커스를 받지 않는 단추(맥 기본값, 간단 화면의 단추)를 누르면 입력란의 조합이 확정되지 않은 채 단추가
    동작해, '황재웅' 을 쓰고 바로 단추를 누르면 '황재' 만 읽힌다. 그래서 입력란 밖을 누르기 직전과 이름을 읽기 직전에
    확정한다. target 이 바로 그 입력란(안)이면 그대로 둔다 (글자 사이를 눌러 커서만 옮기는 경우).
    """
    focus = QApplication.focusWidget()
    if focus is None or not focus.testAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled):
        return
    if target is not None and (target is focus or focus.isAncestorOf(target)):
        return
    method = QGuiApplication.inputMethod()
    method.commit()  # macOS·Linux(IBus): 조합 중인 글자를 확정한다
    method.reset()  # Windows 는 reset 이 조합 중인 글자를 확정한다. 이미 확정했으면 아무 일도 없다


def paint_brand(p: QPainter, size: float, corner: float = 8 / 26) -> None:
    """앱 표시: 둥근 사각형 안의 음성 막대 세 개. 26px 기준 그림을 size 로 늘린다 (corner = 모서리 반지름 / size)."""
    s = size / 26.0
    gradient = QLinearGradient(0, 0, size, size)
    gradient.setColorAt(0.0, QColor("#8b8ef8"))
    gradient.setColorAt(1.0, QColor("#4f46e5"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(gradient)
    p.drawRoundedRect(QRectF(0, 0, size, size), size * corner, size * corner)
    p.setBrush(QColor("#ffffff"))
    for x, height in ((7.0, 8.0), (11.5, 14.0), (16.0, 10.0)):
        p.drawRoundedRect(QRectF(x * s, (13 - height / 2) * s, 3 * s, height * s), 1.5 * s, 1.5 * s)


def brand_image(size: int) -> QImage:
    """앱 아이콘 (Dock·응용 프로그램 폴더). macOS 아이콘 격자(1024 안에 824, 모서리 185)에 맞춘다."""
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    body = size * 824 / 1024
    p.translate((size - body) / 2, (size - body) / 2)
    paint_brand(p, body, corner=185 / 824)
    p.end()
    return image


class BrandMark(QWidget):
    """앱 표시: 둥근 사각형 안의 음성 막대 세 개."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(26, 26)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 이름)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_brand(p, 26)


class LevelMeter(QWidget):
    """입력 레벨(dBFS) 막대. -60 dBFS 이하는 비어 있고 0 dBFS 에서 가득 찬다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = -120.0
        self.setMinimumSize(150, 18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_level(self, db: float) -> None:
        self._db = db
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 이름)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1f2430"))
        painter.drawRoundedRect(rect, 6, 6)
        fraction = min(1.0, max(0.0, (self._db + 60.0) / 60.0))
        if fraction > 0.02:
            color = "#4ade80" if self._db < -12 else ("#fbbf24" if self._db < -3 else "#f87171")
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(QRectF(rect.x(), rect.y(), max(rect.height(), rect.width() * fraction), rect.height()), 6, 6)
        painter.setPen(QColor("#cbd2dc"))
        label = f"{self._db:.0f} dBFS" if self._db > -120 else "입력 없음"
        painter.drawText(rect.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, label)
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#12151c"))
        painter.drawRoundedRect(rect, 10, 10)
        mid = rect.center().y()
        painter.setPen(QColor("#232834"))
        painter.drawLine(QPointF(rect.left() + 10, mid), QPointF(rect.right() - 10, mid))
        if not self._values:
            painter.setPen(QColor("#6f7787"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "녹음을 시작하면 입력 파형이 여기에 흐릅니다")
            painter.end()
            return
        inner = rect.adjusted(10, 6, -10, -6)
        bar = inner.width() / self._values.maxlen
        half = inner.height() / 2
        count = len(self._values)
        painter.setPen(Qt.PenStyle.NoPen)
        for i, (peak, speaking) in enumerate(self._values):
            x = inner.right() - (count - i) * bar
            height = max(1.0, _amp_fraction(peak) * half)
            painter.setBrush(QColor("#4ade80" if speaking else "#818cf8"))
            painter.drawRect(QRectF(x, mid - height, max(1.0, bar - 1), 2 * height))
        painter.end()


class TimelineView(QWidget):
    """녹음 전체 시간축과 음성 구간. 녹음 중에는 실시간 감지 클립, 전사 후에는 Silero VAD 구간을 그린다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(50)
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        track = QRectF(rect.left() + 4, rect.top() + 21, rect.width() - 8, 12)
        caption = "실시간 감지 클립(미리보기)" if self.live else "Silero VAD 음성 구간 (원본 녹음 기준)"
        painter.setPen(QColor("#a3aab8"))
        painter.drawText(QRectF(rect.left() + 4, rect.top(), rect.width() - 8, 18),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         f"음성 구간 타임라인 · {caption}")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#232834"))
        painter.drawRoundedRect(track, 6, 6)
        scale = max(self.duration, 30.0)
        painter.setBrush(QColor("#4ade80" if self.live else "#818cf8"))
        for start, end in self.regions:
            x0 = track.left() + track.width() * max(0.0, start) / scale
            x1 = track.left() + track.width() * min(scale, max(start, end)) / scale
            painter.drawRoundedRect(QRectF(x0, track.top(), max(3.0, x1 - x0), track.height()), 3, 3)
        step = next((s for s in (5, 10, 30, 60, 120, 300, 600, 1800) if scale / s <= 10), 3600)
        painter.setPen(QColor("#6f7787"))
        t = 0
        while t <= scale:
            x = track.left() + track.width() * t / scale
            painter.drawLine(QPointF(x, track.bottom() + 1), QPointF(x, track.bottom() + 4))
            painter.drawText(QRectF(x - 30, track.bottom() + 3, 60, 13), Qt.AlignmentFlag.AlignCenter,
                             f"{int(t) // 60:02d}:{int(t) % 60:02d}")
            t += step
        if self.cursor is not None:
            x = track.left() + track.width() * min(self.cursor, scale) / scale
            painter.setPen(QPen(QColor("#fb7185"), 2))
            painter.drawLine(QPointF(x, track.top() - 3), QPointF(x, track.bottom() + 3))
        painter.end()


class PipelineView(QWidget):
    """진행 단계: ① 녹음 → ② 전사 → ③ 요약 → ④ 검토. 단계마다 상태와 소요 시간을 보여 준다."""

    STEPS = (
        ("record", "① 녹음"),
        ("transcribe", "② 전사 · whisper.cpp (이 컴퓨터)"),
        ("summary", "③ 요약 · Claude (전송)"),
        ("review", "④ 검토"),
    )
    _STYLE = {  # (배경, 글자, 굵기, 앞 표시)
        "idle": ("#1a1e27", "#8b93a1", "600", ""),
        "active": ("#132341", "#7aa2ff", "700", "● "),
        "done": ("#0f2b20", "#4ade80", "700", "✓ "),
        "error": ("#3a1418", "#f87171", "700", "✗ "),
        "skipped": ("#1a1e27", "#6f7787", "600", "– "),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._labels: dict[str, QLabel] = {}
        self.states: dict[str, str] = {}
        for i, (key, _title) in enumerate(self.STEPS):
            if i:
                arrow = QLabel("›")
                arrow.setStyleSheet("color: #4b5263; font-size: 18px;")
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
            f"background: {bg}; color: {fg}; font-weight: {weight}; border: 1px solid {bg}; border-radius: 10px;"
            " padding: 6px 10px; font-size: 13px;"
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


def update_chip() -> QPushButton:
    """위쪽 막대의 업데이트 단추: 새 버전이 준비됐거나 방금 업데이트했을 때만 보인다 (set_update_chip)."""
    button = QPushButton()
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    button.setToolTip("업데이트: 새 버전은 테스트를 통과한 것만 받아 두었다가 앱을 다시 열 때 적용합니다.")
    button.hide()
    return button


def set_update_chip(button: QPushButton, chip: tuple[str, str] | None) -> None:
    if chip is None:
        button.hide()
        return
    text, kind = chip
    button.setText(text)
    button.setStyleSheet(f"QPushButton {{ {theme.pill(kind, 12)} }}")
    button.show()

