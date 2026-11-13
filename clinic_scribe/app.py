"""앱 진입점: python -m clinic_scribe [--open 파일.wav] [--demo [파일.wav]]"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

import shiboken6
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication

from clinic_scribe import mic_permission, updater
from clinic_scribe.config import Settings
from clinic_scribe.ui import theme
from clinic_scribe.ui.main_window import MainWindow
from clinic_scribe.ui.widgets import brand_image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEMO = REPO_ROOT / "samples" / "mock_consult.wav"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinic-scribe", description="모의 진료 녹음 → 로컬 전사 → 검토용 차트 초안")
    parser.add_argument("--open", type=Path, metavar="AUDIO", help="시작하면서 불러올 오디오 파일 (마이크 없이 테스트)")
    parser.add_argument(
        "--demo", nargs="?", const=DEFAULT_DEMO, type=Path, metavar="AUDIO",
        help="입력 장치에 '데모 입력'을 추가한다. 녹음 버튼을 누르면 이 가상 대화 파일을 마이크처럼 실시간으로 흘려 넣는다 "
             "(기본: samples/mock_consult.wav, 없으면 만든다)",
    )
    parser.add_argument("--demo-speed", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--summary", choices=("general", "chart"),
        help="요약 종류: general = 일반 대화 요약, chart = 진료 차트 초안 (지정하지 않으면 마지막으로 고른 값)",
    )
    _ensure_stdio()
    args = parser.parse_args(argv)
    if args.open is not None and not args.open.is_file():
        parser.error(f"파일이 없습니다: {args.open}")
    if args.demo is not None and not args.demo.is_file():
        if args.demo == DEFAULT_DEMO and sys.platform != "darwin":
            parser.error("데모 음성(samples/)은 macOS 의 say 로 만듭니다. Mac 에서 만든 samples 폴더를 복사하거나 "
                         "--demo 다음에 가진 오디오 파일을 지정하세요.")
        if args.demo == DEFAULT_DEMO:
            print("가상 진료 대화 음성을 만드는 중입니다 (macOS say, 약 20초)…", file=sys.stderr)
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_test_audio.py")], check=False,
                           stdout=subprocess.DEVNULL)
        if not args.demo.is_file():
            parser.error(f"데모 입력 파일이 없습니다: {args.demo}")

    if sys.platform == "win32":
        _windows_taskbar_identity()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("clinic-scribe")
    app.setDesktopFileName("clinic-scribe")  # Linux: 바로 가기(clinic-scribe.desktop)와 창을 한 앱으로 묶는다
    if sys.platform != "darwin":
        # Windows·Linux: OS 테마(밝음/어두움)와 관계없이 macOS 와 같은 어두운 화면 (확인 창·파일 창 포함)
        app.setStyle("Fusion")
        app.setPalette(theme.dark_palette())
    if not mic_permission.launched_as_app():  # 앱으로 열면 macOS 가 그린 앱 아이콘을 그대로 둔다
        app.setWindowIcon(QIcon(QPixmap.fromImage(brand_image(1024))))  # 터미널에서 띄울 때도 Dock 아이콘을 앱 로고로
    # 화면 색은 어두운 배경 기준이다. 시스템이 밝은 모드여도 설정·확인 창까지 어둡게 맞춘다.
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    settings = Settings.load()
    if args.summary:
        settings.summary_mode = args.summary
    window = MainWindow(settings, demo_file=args.demo, demo_speed=args.demo_speed)  # 창 크기는 화면 종류에 맞춘다
    window.show()
    updater.mark_started()  # 새 버전으로 창이 떴다: 다음에 열 때 되돌리지 않는다
    # 새 컴퓨터 등에서 whisper·모델이 없거나 요약 방식을 아직 고르지 않았으면 '설치' 창을 띄운다
    QTimer.singleShot(0, window.check_setup)

    # 터미널에서 Ctrl+C 를 눌러도 마이크를 놓고 정리한 뒤 끝나게 한다.
    signal.signal(signal.SIGINT, lambda *_: window.close())
    wake = QTimer()
    wake.start(250)
    wake.timeout.connect(lambda: None)

    if args.open is not None:
        QTimer.singleShot(0, lambda: window.open_audio_file(args.open, confirm=False))
    code = app.exec()
    # Python 이 닫히기 전에 Qt 를 정리한다: 창을 붙잡고 있던 Ctrl+C 처리기를 풀고, 창과 QApplication 을 지운다
    signal.signal(signal.SIGINT, signal.default_int_handler)
    wake.stop()
    close_qt()
    return code


def close_qt() -> None:
    """남은 창과 QApplication 을 지금 지운다 (앱을 닫을 때·테스트가 다 끝났을 때, Python 이 닫히기 전에).

    남겨 두면 Python 이 닫히는 도중에 Qt 가 지우면서 이미 정리된 창의 Python 코드를 불러, Windows·Linux 에서 가끔
    비정상 종료한다 (CI 에서 본 일: 모든 테스트가 끝난 뒤 access violation · segmentation fault, Python 코드 밖).
    부모가 있는 창은 부모와 함께 지워지므로, Python 이 만든 부모 없는 창만 먼저 골라 지운다.
    """
    app = QApplication.instance()
    if app is None:
        return
    roots = [w for w in QApplication.topLevelWidgets() if w.parent() is None and shiboken6.createdByPython(w)]
    for widget in roots:
        if shiboken6.isValid(widget):
            shiboken6.delete(widget)
    shiboken6.delete(app)


def _ensure_stdio() -> None:
    """Windows 바로 가기(pythonw)로 열면 표준 출력·오류가 없다. 어떤 출력도 앱을 멈추지 않게 버리는 곳으로 돌린다."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _windows_taskbar_identity() -> None:
    """작업 표시줄에 python 아이콘 대신 이 앱 아이콘이 따로 보이게 한다."""
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("local.clinic-scribe")
    except (AttributeError, OSError):
        pass
