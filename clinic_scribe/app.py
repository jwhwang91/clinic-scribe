"""앱 진입점: python -m clinic_scribe [--open 파일.wav]"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from clinic_scribe.config import Settings
from clinic_scribe.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinic-scribe", description="모의 진료 녹음 → 로컬 전사 → 검토용 차트 초안")
    parser.add_argument("--open", type=Path, metavar="AUDIO", help="시작하면서 불러올 오디오 파일 (마이크 없이 테스트)")
    args = parser.parse_args(argv)
    if args.open is not None and not args.open.is_file():
        parser.error(f"파일이 없습니다: {args.open}")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("clinic-scribe")
    window = MainWindow(Settings.load())
    window.resize(1360, 860)
    window.show()

    # 터미널에서 Ctrl+C 를 눌러도 마이크를 놓고 정리한 뒤 끝나게 한다.
    signal.signal(signal.SIGINT, lambda *_: window.close())
    wake = QTimer()
    wake.start(250)
    wake.timeout.connect(lambda: None)

    if args.open is not None:
        QTimer.singleShot(0, lambda: window.open_audio_file(args.open, confirm=False))
    return app.exec()
