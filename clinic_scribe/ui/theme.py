"""화면 공통 디자인: 어두운 색과 버튼·입력칸·탭·표 모양. 간단 화면과 개발자 보기가 함께 쓴다."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont

ASSETS = Path(__file__).with_name("assets")
ACCENT = "#6366f1"

# 상태 표시 색: (배경, 글자)
STATUS_COLORS = {
    "idle": ("#1c212b", "#a3aab8"),
    "recording": ("#3a1420", "#fb7185"),
    "busy": ("#132341", "#7aa2ff"),
    "summary": ("#241b3f", "#c4b5fd"),
    "done": ("#0f2b20", "#4ade80"),
    "warn": ("#33230f", "#fbbf24"),
    "error": ("#3a1418", "#f87171"),
}

# 안내 상자: (배경, 글자, 테두리)
_CALLOUTS = {
    "ok": ("#0f2219", "#86efac", "#1d4d33"),
    "note": ("#211d12", "#e9c46a", "#4a3c17"),
    "warn": ("#2a1a0e", "#fdba74", "#7c3a12"),
    "error": ("#2a1215", "#fca5a5", "#7f1d1d"),
}

_BASE = """
#topBar { background: #13161d; border-bottom: 1px solid #20242e; }
#brand { font-size: 15px; font-weight: 700; color: #e6e8ee; }
#chip { background: #1e2140; color: #a5b4fc; border: 1px solid #1e2140; border-radius: 9px; padding: 0 8px;
        min-height: 18px; font-size: 11px; font-weight: 700; }
#subtle { font-size: 12px; color: #6f7787; }
#muted { font-size: 13px; color: #a3aab8; }
#micStatus { font-size: 12px; }
#cardTitle { font-size: 14px; font-weight: 700; color: #e6e8ee; }
#card { background: #161a22; border: 1px solid #232834; border-radius: 14px; }
#divider { background: #232834; }
QLabel { color: #cbd2dc; }
QPushButton { background: #1b1f29; color: #dfe3ea; border: 1px solid #2a303c; border-radius: 9px;
              padding: 7px 14px; font-size: 13px; font-weight: 600; }
QPushButton:hover { background: #222734; border-color: #39404f; }
QPushButton:pressed { background: #191d26; }
QPushButton:disabled { background: #161a22; color: #4b5263; border-color: #20242e; }
QPushButton#ghost { background: transparent; border: 1px solid transparent; color: #a3aab8; padding: 6px 10px; }
QPushButton#ghost:hover { background: #1b1f29; color: #e6e8ee; }
QPushButton#ghost:disabled { background: transparent; color: #464c5b; }
QPushButton#tab { background: transparent; border: 1px solid transparent; color: #8b93a1; padding: 6px 12px; }
QPushButton#tab:hover { color: #e6e8ee; }
QPushButton#tab:checked { background: #232838; color: #f1f3f7; border-color: #2f3546; }
QPushButton#tab:disabled { background: transparent; color: #464c5b; border-color: transparent; }
QPushButton#primary { background: #6366f1; color: #ffffff; border-color: #6366f1; }
QPushButton#primary:hover { background: #7376f3; border-color: #7376f3; }
QPushButton#primary:disabled { background: #22253a; border-color: #22253a; color: #5d6380; }
QComboBox { background: #1b1f29; color: #dfe3ea; border: 1px solid #2a303c; border-radius: 9px;
            padding: 5px 30px 5px 12px; font-size: 13px; }
QComboBox:hover { border-color: #39404f; }
QComboBox:disabled { color: #5c6372; }
QComboBox::drop-down { border: none; width: 28px; subcontrol-origin: padding; subcontrol-position: center right; }
QComboBox::down-arrow { image: url({ASSETS}/chevron-down.svg); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: #1b1f29; color: #dfe3ea; border: 1px solid #2a303c; padding: 4px; outline: 0;
                              selection-background-color: #2a2f45; selection-color: #ffffff; }
QAbstractSpinBox { background: #1b1f29; color: #dfe3ea; border: 1px solid #2a303c; border-radius: 8px;
                   padding: 4px 20px 4px 8px; font-size: 13px; }
QAbstractSpinBox:hover { border-color: #39404f; }
QAbstractSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 18px; border: none; }
QAbstractSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 18px; border: none; }
QAbstractSpinBox::up-arrow { image: url({ASSETS}/chevron-up.svg); width: 9px; height: 9px; }
QAbstractSpinBox::down-arrow { image: url({ASSETS}/chevron-down.svg); width: 9px; height: 9px; }
QLineEdit, QPlainTextEdit { background: #1b1f29; color: #dfe3ea; border: 1px solid #2a303c; border-radius: 8px;
                            padding: 5px 8px; font-size: 13px; selection-background-color: #3b3f75; }
QLineEdit:focus, QPlainTextEdit:focus, QAbstractSpinBox:focus { border-color: #6366f1; }
QGroupBox { border: 1px solid #232834; border-radius: 12px; margin-top: 22px; padding: 14px 12px 10px 12px;
            color: #e6e8ee; font-size: 13px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 4px; }
QCheckBox { color: #cbd2dc; spacing: 8px; font-size: 13px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 5px; border: 1px solid #39404f; background: #1b1f29; }
QCheckBox::indicator:checked { background: #6366f1; border-color: #6366f1; image: url({ASSETS}/check.svg); }
QTabWidget::pane { border: none; }
QTabWidget::tab-bar { alignment: left; }
QTabBar { background: transparent; qproperty-drawBase: 0; }
QTabBar::tab { background: transparent; color: #8b93a1; border: 1px solid transparent; border-radius: 8px;
               padding: 6px 12px; margin-right: 4px; font-size: 13px; font-weight: 600; }
QTabBar::tab:selected { background: #232838; color: #f1f3f7; border-color: #2f3546; }
QTabBar::tab:hover { color: #e6e8ee; }
QTableWidget { background: transparent; color: #dfe3ea; border: none; gridline-color: #232834;
               selection-background-color: #2a2f45; selection-color: #ffffff; font-size: 13px; }
QHeaderView::section { background: #13161d; color: #8b93a1; border: none; border-bottom: 1px solid #232834;
                       padding: 6px 8px; font-size: 12px; font-weight: 600; }
QTableCornerButton::section { background: #13161d; border: none; }
QTextBrowser { background: transparent; border: none; font-size: 14px; color: #e6e8ee;
               selection-background-color: #3b3f75; }
QListWidget { background: transparent; border: none; outline: 0; }
QSplitter::handle { background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #2c323e; border-radius: 3px; min-height: 32px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #2c323e; border-radius: 3px; min-width: 32px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QToolTip { background: #1b1f29; color: #e6e8ee; border: 1px solid #2a303c; }
"""

# 요약·전사문 글자 모양 (내용 HTML 은 요약 모듈과 메인 창이 만든다)
DOCUMENT_STYLE = "h3 { font-size: 15px; color: #f1f3f7; } li { margin-bottom: 3px; } p { line-height: 140%; }"


def stylesheet(root: str, extra: str = "") -> str:
    """root(화면 위젯 선택자)의 배경 + 공통 모양 + 화면별 모양."""
    return (f"{root} {{ background: #0f1117; }}\n" + _BASE + extra).replace("{ASSETS}", ASSETS.as_posix())


def pill(kind: str, font_px: int = 13) -> str:
    background, foreground = STATUS_COLORS.get(kind, STATUS_COLORS["idle"])
    # Qt 는 반지름이 높이의 절반을 넘으면 모서리를 아예 둥글리지 않으므로 최소 높이를 함께 정한다.
    return (
        f"background: {background}; color: {foreground}; border: 1px solid {background}; border-radius: 13px;"
        f" padding: 4px 14px; min-height: 20px; font-size: {font_px}px; font-weight: 600;"
    )


def callout(kind: str) -> str:
    background, foreground, border = _CALLOUTS[kind]
    return (
        f"background: {background}; color: {foreground}; border: 1px solid {border}; border-radius: 10px;"
        " padding: 8px 12px; font-size: 12px;"
    )


def record_button(recording: bool) -> str:
    base, hover = ("#e11d48", "#f43f5e") if recording else (ACCENT, "#7376f3")
    return (
        f"QPushButton {{ background: {base}; color: #ffffff; border: none; border-radius: 14px;"
        f" font-size: 17px; font-weight: 700; padding: 0 22px; }}"
        f" QPushButton:hover {{ background: {hover}; }}"
        " QPushButton:disabled { background: #2a2f3a; color: #6b7280; }"
    )


def timer_font(pixel_size: int) -> QFont:
    """경과 시간 글꼴: 가늘고, 숫자 폭이 같아 시간이 바뀔 때 흔들리지 않는다 (Qt 6.7+)."""
    font = QFont()
    font.setPixelSize(pixel_size)
    font.setWeight(QFont.Weight.Light)
    try:
        font.setFeature(QFont.Tag.fromString("tnum"), 1)
    except (AttributeError, TypeError):
        pass
    return font
