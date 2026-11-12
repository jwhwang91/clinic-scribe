"""간단 화면 (기본). 어두운 테마, 두 칸 구성.

왼쪽 칸: 원형 녹음 버튼, 경과 시간, 입력 막대, 상태. 오른쪽 칸: 진료 기록 · 요약 · 대화 내용 탭.
대화 내용과 요약은 숨겨 두고 탭을 눌러야 보인다(기본은 진료 기록 탭). 진료 기록은 날짜별로 묶어 보여 주고,
고른 기록은 '삭제' 버튼이나 오른쪽 클릭으로 지운다.
스페이스바로 녹음을 시작·종료한다(키 처리는 MainWindow 가 한다). 이 화면의 버튼은 키보드 포커스를 받지 않아
스페이스바가 항상 녹음 토글로만 동작한다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from clinic_scribe.ui.theme import ACCENT, DOCUMENT_STYLE, STATUS_COLORS, callout, pill, stylesheet, timer_font
from clinic_scribe.ui.widgets import BrandMark

# 간단 화면에만 있는 모양 (공통 모양은 theme)
_EXTRA = """
#timer { color: #f1f3f7; }
#keycap { background: #1b1f29; color: #cbd2dc; border: 1px solid #2f3542; border-bottom: 2px solid #3a4150;
          border-radius: 6px; padding: 1px 8px; font-size: 12px; font-weight: 600; }
#autoSplit { font-size: 12px; color: #6f7787; }
#demoWarning { background: #2a2210; color: #fcd34d; border: 1px solid #6b5313; border-radius: 10px;
               padding: 7px 12px; font-size: 13px; font-weight: 600; }
#empty { font-size: 13px; color: #6f7787; }
#footnote { font-size: 11px; color: #5c6372; }
"""


@dataclass(frozen=True)
class HistoryRow:
    """진료 기록 한 줄. 내용(전사·요약)은 담지 않는다."""

    session_id: int
    when: str  # 예: "10:02–10:09 · 07:12"
    status: str  # 예: "요약 준비됨"
    kind: str  # 상태 색 (STATUS_COLORS 의 키)
    note: str = ""  # 나눈 이유 등
    day: str = ""  # 날짜 묶음 제목 (예: "오늘 · 11월 12일 (목)"). 바뀌는 곳마다 제목 줄이 들어간다

    @property
    def text(self) -> str:
        return " · ".join(part for part in (f"진료 #{self.session_id}", self.when, self.status, self.note) if part)


_ROW_ROLE = Qt.ItemDataRole.UserRole + 1
_WEEKDAYS = "월화수목금토일"


def day_label(day: date, today: date) -> str:
    """진료 기록의 날짜 묶음 제목: 오늘 · 11월 12일 (목) / 어제 · … / 11월 3일 (월) / 2025년 12월 30일 (화)."""
    text = f"{day.month}월 {day.day}일 ({_WEEKDAYS[day.weekday()]})"
    if day.year != today.year:
        text = f"{day.year}년 {text}"
    gap = (today - day).days
    return {0: f"오늘 · {text}", 1: f"어제 · {text}"}.get(gap, text)


def _font(pixel_size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setPixelSize(pixel_size)
    font.setWeight(weight)
    return font


class RecordButton(QAbstractButton):
    """원형 녹음 버튼. 녹음 중에는 빨간색이 되고 바깥으로 퍼지는 고리가 움직인다."""

    RADIUS = 58
    SIZE = 2 * (RADIUS + 28)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._recording = False
        self._phase = 0.0
        self._pulse = QVariantAnimation(self)
        self._pulse.setStartValue(0.0)
        self._pulse.setEndValue(1.0)
        self._pulse.setDuration(1800)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._on_phase)
        self.set_recording(False)

    def set_recording(self, recording: bool) -> None:
        self._recording = recording
        self.setText("녹음 종료" if recording else "녹음 시작")
        self.setAccessibleName(self.text())
        if recording:
            self._pulse.start()
        else:
            self._pulse.stop()
            self._phase = 0.0
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.SIZE, self.SIZE)

    def hitButton(self, pos) -> bool:  # noqa: N802 — 원 바깥 모서리는 누르지 않은 것으로 본다
        offset = QPointF(pos) - QPointF(self.width() / 2, self.height() / 2)
        return (offset.x() ** 2 + offset.y() ** 2) ** 0.5 <= self.RADIUS + 12

    def enterEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().leaveEvent(event)

    def _on_phase(self, value) -> None:
        self._phase = float(value)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = self.RADIUS
        enabled = self.isEnabled()
        if not enabled:
            base, light = QColor("#2a2f3a"), QColor("#363c49")
        elif self._recording:
            base, light = QColor("#e11d48"), QColor("#fb7185")
        else:
            base, light = QColor("#4f46e5"), QColor("#8b8ef8")
        p.setPen(Qt.PenStyle.NoPen)
        if self._recording and enabled:
            for shift in (0.0, 0.5):  # 퍼져 나가며 옅어지는 고리 두 개
                t = (self._phase + shift) % 1.0
                ring = QColor(base)
                ring.setAlphaF(0.32 * (1.0 - t))
                p.setBrush(ring)
                p.drawEllipse(center, radius + 26 * t, radius + 26 * t)
        elif enabled:
            halo = QColor(base)
            halo.setAlphaF(0.24 if self.underMouse() else 0.16)
            p.setBrush(halo)
            p.drawEllipse(center, radius + 12, radius + 12)
        gradient = QLinearGradient(center.x(), center.y() - radius, center.x(), center.y() + radius)
        gradient.setColorAt(0.0, light.lighter(108) if self.underMouse() and enabled else light)
        gradient.setColorAt(1.0, base.darker(115) if self.isDown() else base)
        p.setBrush(gradient)
        p.drawEllipse(center, radius, radius)
        p.setBrush(QColor("#ffffff" if enabled else "#6b7280"))
        icon = QPointF(center.x(), center.y() - 11)
        if self._recording:
            p.drawRoundedRect(QRectF(icon.x() - 10, icon.y() - 10, 20, 20), 5, 5)
        else:
            p.drawEllipse(icon, 11, 11)
        p.setFont(_font(13, QFont.Weight.DemiBold))
        p.setPen(QColor("#ffffff" if enabled else "#8b93a1"))
        p.drawText(QRectF(center.x() - radius, center.y() + 8, 2 * radius, 22), Qt.AlignmentFlag.AlignCenter, self.text())


class LiveBars(QWidget):
    """최근 입력 크기를 막대로 보여 준다 (오른쪽이 최신). 녹음하지 않을 때는 낮은 회색 막대."""

    BARS = 40

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._levels: deque[float] = deque([0.0] * self.BARS, maxlen=self.BARS)
        self._active = False
        self.setFixedSize(self.BARS * 7, 36)

    def set_level(self, db: float) -> None:
        """입력 크기(dBFS). -120 이하면 녹음이 끝난 것으로 보고 비운다."""
        if db <= -119.0:
            self._levels.extend([0.0] * self.BARS)
            self._active = False
        else:
            self._active = True
            self._levels.append(min(1.0, max(0.0, (db + 60.0) / 50.0)) ** 0.9)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#fb7185" if self._active else "#2c323e"))
        height = self.height()
        for i, level in enumerate(self._levels):
            bar = max(4.0, level * height)
            p.drawRoundedRect(QRectF(i * 7 + 1.5, (height - bar) / 2, 4, bar), 2, 2)


class DeviceCombo(QComboBox):
    """목록을 펼치기 직전에 about_to_open 을 보낸다. 그때 장치를 다시 읽으면 앱을 켠 뒤 꽂은 마이크도 보인다."""

    about_to_open = Signal()

    def showPopup(self) -> None:  # noqa: N802
        self.about_to_open.emit()
        super().showPopup()


class _HistoryDelegate(QStyledItemDelegate):
    """진료 기록 한 줄: 번호 배지 · 시간 · 나눈 이유 · 상태 칩. 날짜가 바뀌는 곳에는 날짜 제목 줄."""

    ROW_HEIGHT = 56
    HEADER_HEIGHT = 30

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        header = isinstance(index.data(_ROW_ROLE), str)
        return QSize(max(200, option.rect.width()), self.HEADER_HEIGHT if header else self.ROW_HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:
        row = index.data(_ROW_ROLE)
        if isinstance(row, str):  # 날짜 제목
            painter.save()
            painter.setFont(_font(12, QFont.Weight.DemiBold))
            painter.setPen(QColor("#8b93a1"))
            text_rect = QRectF(option.rect).adjusted(12, 8, -12, 0)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, row)
            painter.restore()
            return
        if not isinstance(row, HistoryRow):
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect).adjusted(2, 2, -2, -2)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.setPen(Qt.PenStyle.NoPen)
        if selected or hovered:
            painter.setBrush(QColor("#1f2338" if selected else "#1a1e28"))
            painter.drawRoundedRect(rect, 10, 10)

        badge_font = _font(13, QFont.Weight.Bold)
        badge_width = max(30, QFontMetrics(badge_font).horizontalAdvance(str(row.session_id)) + 14)
        badge = QRectF(rect.left() + 10, rect.center().y() - 15, badge_width, 30)
        painter.setBrush(QColor(ACCENT if selected else "#1e2140"))
        painter.drawRoundedRect(badge, 8, 8)
        painter.setFont(badge_font)
        painter.setPen(QColor("#ffffff" if selected else "#a5b4fc"))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(row.session_id))

        chip_font = _font(12, QFont.Weight.DemiBold)
        chip_width = QFontMetrics(chip_font).horizontalAdvance(row.status) + 22
        chip = QRectF(rect.right() - 10 - chip_width, rect.center().y() - 12, chip_width, 24)
        background, foreground = STATUS_COLORS.get(row.kind, STATUS_COLORS["idle"])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(background))
        painter.drawRoundedRect(chip, 12, 12)
        painter.setFont(chip_font)
        painter.setPen(QColor(foreground))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, row.status)

        left, right = badge.right() + 12, chip.left() - 12
        align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        title_font = _font(14, QFont.Weight.DemiBold)
        if row.note:
            title = QRectF(left, rect.center().y() - 19, right - left, 20)
            note = QRectF(left, rect.center().y() + 1, right - left, 18)
            note_font = _font(12)
            painter.setFont(note_font)
            painter.setPen(QColor("#8b93a1"))
            elided = QFontMetrics(note_font).elidedText(row.note, Qt.TextElideMode.ElideRight, int(note.width()))
            painter.drawText(note, align, elided)
        else:
            title = QRectF(left, rect.top(), right - left, rect.height())
        painter.setFont(title_font)
        painter.setPen(QColor("#e6e8ee"))
        painter.drawText(title, align, QFontMetrics(title_font).elidedText(row.when, Qt.TextElideMode.ElideRight, int(title.width())))
        painter.restore()


class SimpleView(QWidget):
    record_clicked = Signal()
    retry_clicked = Signal()
    copy_clicked = Signal()
    settings_clicked = Signal()
    voices_clicked = Signal()
    developer_clicked = Signal()
    history_selected = Signal(int)  # 진료 번호
    delete_requested = Signal(int)  # 지울 진료 번호 (-1: 지금 고른 진료)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(stylesheet("SimpleView", _EXTRA))
        self._panel = "history"
        self._history_rows: tuple[HistoryRow, ...] | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_top_bar())

        self.demo_warning = QLabel("데모 입력이 골라져 있습니다. 마이크 소리 대신 데모 오디오 파일이 들어갑니다.")
        self.demo_warning.setObjectName("demoWarning")
        self.demo_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.demo_warning.hide()
        warning_row = QHBoxLayout()
        warning_row.setContentsMargins(16, 12, 16, 0)
        warning_row.addWidget(self.demo_warning)
        outer.addLayout(warning_row)

        body = QHBoxLayout()
        body.setContentsMargins(16, 14, 16, 8)
        body.setSpacing(14)
        body.addWidget(self._build_recorder())
        body.addWidget(self._build_panel(), 1)
        outer.addLayout(body, 1)

        self.note = QLabel("녹음·전사·진료 기록은 이 컴퓨터 에만 · 요약을 만들 때 전사문이 Claude 서비스로 전송됩니다")
        self.note.setObjectName("footnote")
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note.setContentsMargins(0, 0, 0, 10)
        outer.addWidget(self.note)

        self.set_recording(False, True)
        self.set_status("", "idle")
        self.set_history([], None)
        self.set_results(False, False, False)

    # ---- 화면 구성 -------------------------------------------------------------
    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(bar)
        row.setContentsMargins(18, 10, 14, 10)
        row.setSpacing(10)
        row.addWidget(BrandMark())
        brand = QLabel("clinic-scribe")
        brand.setObjectName("brand")
        row.addWidget(brand)
        tagline = QLabel("녹음 · 전사 · 요약")
        tagline.setObjectName("subtle")
        row.addWidget(tagline)
        row.addStretch(1)
        self.device_combo = DeviceCombo()
        self.device_combo.setMinimumWidth(240)
        self.device_combo.setMaximumWidth(320)
        self.device_combo.setToolTip("녹음에 쓸 마이크. 목록을 펼칠 때마다 새로 꽂은 장치까지 다시 읽습니다.")
        self.device_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(self.device_combo)
        self.mic_status = QLabel()
        self.mic_status.setObjectName("micStatus")
        row.addWidget(self.mic_status)
        self.voices_btn = QPushButton("의사 목소리")
        self.voices_btn.setToolTip("의사 목소리를 등록하면 전사문의 발화마다 의사 이름이나 '환자'가 붙습니다.")
        self.voices_btn.clicked.connect(self.voices_clicked)
        self.settings_btn = QPushButton("설정")
        self.settings_btn.clicked.connect(self.settings_clicked)
        self.developer_btn = QPushButton("개발자 보기")
        self.developer_btn.setToolTip("전사문·음성 구간·VAD 비교·실시간 전사가 모두 보이는 개발자용 대시보드")
        self.developer_btn.clicked.connect(self.developer_clicked)
        for button in (self.voices_btn, self.settings_btn, self.developer_btn):
            button.setObjectName("ghost")
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.addWidget(button)
        return bar

    def _build_recorder(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setFixedWidth(340)
        column = QVBoxLayout(card)
        column.setContentsMargins(20, 16, 20, 16)
        column.setSpacing(0)
        center = Qt.AlignmentFlag.AlignHCenter
        column.addStretch(1)

        self.record_btn = RecordButton()
        self.record_btn.clicked.connect(self.record_clicked)
        column.addWidget(self.record_btn, 0, center)

        self.elapsed = QLabel("00:00")
        self.elapsed.setObjectName("timer")
        self.elapsed.setFont(timer_font(44))
        self.elapsed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self.elapsed)
        column.addSpacing(4)
        self.level = LiveBars()
        column.addWidget(self.level, 0, center)
        column.addSpacing(14)

        # 카드 폭을 채운다: 가운데 정렬로 넣으면 Qt 가 글 폭을 좁게 잡아 짧은 문구도 두 줄로 접고 잘라 버린다.
        self.status = QLabel()
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        column.addWidget(self.status)

        self.error = QLabel()
        self.error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error.setWordWrap(True)
        self.error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error.hide()
        column.addSpacing(8)
        column.addWidget(self.error)
        self.retry_btn = QPushButton("요약 다시 시도")
        self.retry_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.retry_btn.clicked.connect(self.retry_clicked)
        self.retry_btn.hide()
        column.addSpacing(6)
        column.addWidget(self.retry_btn, 0, center)

        column.addSpacing(10)
        hint_row = QHBoxLayout()
        hint_row.setSpacing(8)
        hint_row.addStretch(1)
        keycap = QLabel("Space")
        keycap.setObjectName("keycap")
        hint_row.addWidget(keycap)
        self.hint = QLabel("녹음 시작 · 종료")
        self.hint.setObjectName("muted")
        hint_row.addWidget(self.hint)
        hint_row.addStretch(1)
        column.addLayout(hint_row)
        column.addSpacing(8)
        self.auto_split_note = QLabel()
        self.auto_split_note.setObjectName("autoSplit")
        self.auto_split_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.auto_split_note.setWordWrap(True)
        column.addWidget(self.auto_split_note)
        column.addStretch(1)
        return card

    def _build_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        column = QVBoxLayout(card)
        column.setContentsMargins(12, 10, 12, 10)
        column.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(4)
        self.history_tab = QPushButton("진료 기록")
        self.history_tab.setToolTip("진료를 고르면 요약·대화 내용·복사가 그 진료를 가리킵니다")
        self.show_summary_btn = QPushButton("요약")
        self.show_transcript_btn = QPushButton("대화 내용")
        self._tabs = QButtonGroup(self)
        self._tabs.setExclusive(True)
        for button, name in ((self.history_tab, "history"), (self.show_summary_btn, "summary"),
                             (self.show_transcript_btn, "transcript")):
            button.setObjectName("tab")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.toggled.connect(lambda checked, name=name: checked and self._show(name))
            self._tabs.addButton(button)
            header.addWidget(button)
        header.addStretch(1)
        self.delete_btn = QPushButton("삭제")
        self.delete_btn.setObjectName("danger")
        self.delete_btn.setToolTip("고른 진료 기록(녹음·전사문·요약)을 이 컴퓨터 에서 지웁니다")
        self.delete_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(-1))
        header.addWidget(self.delete_btn)
        self.copy_btn = QPushButton("요약 복사")
        self.copy_btn.setObjectName("primary")
        self.copy_btn.setToolTip("요약을 클립보드에 복사합니다 (차트에 옮겨 적을 때).")
        self.copy_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.copy_btn.clicked.connect(self.copy_clicked)
        header.addWidget(self.copy_btn)
        column.addLayout(header)
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        column.addWidget(divider)

        # 진료 기록: 내용은 숨기고 번호·시간·상태만 보여 준다. 날짜가 바뀌는 곳에 날짜 제목 줄.
        self.history = QListWidget()
        self.history.setItemDelegate(_HistoryDelegate(self.history))
        self.history.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.history.setMouseTracking(True)
        self.history.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.history.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history.itemClicked.connect(self._on_history_clicked)
        self.history.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history.customContextMenuRequested.connect(self._history_menu)
        column.addWidget(self.history, 1)
        self.empty = QLabel("아직 진료 기록이 없습니다\nSpace 를 눌러 녹음을 시작하면 진료가 여기에 쌓입니다")
        self.empty.setObjectName("empty")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self.empty, 1)

        self.summary_view = QTextBrowser()
        self.transcript_view = QTextBrowser()
        for view in (self.summary_view, self.transcript_view):
            view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            view.document().setDefaultStyleSheet(DOCUMENT_STYLE)
            view.document().setDocumentMargin(10)
            column.addWidget(view, 1)
        self.history_tab.setChecked(True)
        return card

    # ---- 상태 -------------------------------------------------------------
    def set_recording(self, recording: bool, enabled: bool) -> None:
        self.record_btn.set_recording(recording)
        self.record_btn.setEnabled(enabled)

    def set_status(self, text: str, kind: str) -> None:
        self.status.setText(text)
        self.status.setVisible(bool(text))
        self.status.setStyleSheet(pill(kind))

    def set_error(self, message: str | None, warn: bool = False) -> None:
        if not message:
            self.error.clear()
            self.error.hide()
            return
        self.error.setStyleSheet(callout("warn" if warn else "error"))
        self.error.setText(message)
        self.error.show()

    def set_history(self, rows: list[HistoryRow], selected_id: int | None) -> None:
        """rows: 최신순. 바뀐 것이 없으면 목록을 다시 만들지 않고 고른 줄만 맞춘다 (기록이 많아도 가볍게)."""
        self.history.blockSignals(True)
        if tuple(rows) != self._history_rows:
            self._history_rows = tuple(rows)
            self.history.clear()
            day = None
            for row in rows:
                if row.day and row.day != day:
                    day = row.day
                    header = QListWidgetItem(day)
                    header.setData(_ROW_ROLE, day)
                    header.setFlags(Qt.ItemFlag.NoItemFlags)
                    self.history.addItem(header)
                item = QListWidgetItem(row.text)
                item.setData(Qt.ItemDataRole.UserRole, row.session_id)
                item.setData(_ROW_ROLE, row)
                self.history.addItem(item)
        current = None
        for item in self.history_items():
            chosen = item.data(Qt.ItemDataRole.UserRole) == selected_id
            item.setSelected(chosen)
            if chosen:
                current = item
        if current is not None:
            self.history.setCurrentItem(current)
        else:
            self.history.setCurrentRow(-1)
        self.history.blockSignals(False)
        self.history_tab.setText(f"진료 기록  {len(rows)}" if rows else "진료 기록")
        self._show(self._panel)

    def history_items(self) -> list[QListWidgetItem]:
        """진료 기록 줄 (날짜 제목 줄은 빼고, 최신순)."""
        items = (self.history.item(i) for i in range(self.history.count()))
        return [item for item in items if item.data(Qt.ItemDataRole.UserRole) is not None]

    def _on_history_clicked(self, item: QListWidgetItem) -> None:
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id is not None:
            self.history_selected.emit(int(session_id))

    def _history_menu(self, pos) -> None:
        item = self.history.itemAt(pos)
        session_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if session_id is None:
            return
        menu = QMenu(self.history)
        menu.addAction(f"진료 #{session_id} 기록 삭제…").triggered.connect(lambda: self.delete_requested.emit(int(session_id)))
        menu.exec(self.history.viewport().mapToGlobal(pos))
        menu.deleteLater()

    # ---- 결과 (기본 숨김: 진료 기록 탭) ------------------------------------------
    def set_results(self, has_transcript: bool, has_summary: bool, can_copy: bool, can_delete: bool = False) -> None:
        self.show_transcript_btn.setEnabled(has_transcript)
        self.show_summary_btn.setEnabled(has_summary)
        self.copy_btn.setEnabled(can_copy)
        self.delete_btn.setEnabled(can_delete)
        if (self._panel == "summary" and not has_summary) or (self._panel == "transcript" and not has_transcript):
            self.history_tab.setChecked(True)

    def collapse_results(self) -> None:
        self.history_tab.setChecked(True)

    def _show(self, name: str) -> None:
        self._panel = name
        has_rows = self.history.count() > 0
        self.history.setVisible(name == "history" and has_rows)
        self.empty.setVisible(name == "history" and not has_rows)
        self.summary_view.setVisible(name == "summary")
        self.transcript_view.setVisible(name == "transcript")
