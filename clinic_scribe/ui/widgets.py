"""작은 위젯들."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QSizePolicy, QWidget


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
