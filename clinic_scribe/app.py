"""앱 진입점: python -m clinic_scribe [--open 파일.wav] [--demo [파일.wav]]"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from clinic_scribe.config import Settings
from clinic_scribe.ui.main_window import MainWindow

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
    args = parser.parse_args(argv)
    if args.open is not None and not args.open.is_file():
        parser.error(f"파일이 없습니다: {args.open}")
    if args.demo is not None and not args.demo.is_file():
        if args.demo == DEFAULT_DEMO:
            print("가상 진료 대화 음성을 만드는 중입니다 (macOS say, 약 20초)…", file=sys.stderr)
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_test_audio.py")], check=False,
                           stdout=subprocess.DEVNULL)
        if not args.demo.is_file():
            parser.error(f"데모 입력 파일이 없습니다: {args.demo}")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("clinic-scribe")
    settings = Settings.load()
    if args.summary:
        settings.summary_mode = args.summary
    window = MainWindow(settings, demo_file=args.demo, demo_speed=args.demo_speed)
    window.resize(1400, 900)
    window.show()

    # 터미널에서 Ctrl+C 를 눌러도 마이크를 놓고 정리한 뒤 끝나게 한다.
    signal.signal(signal.SIGINT, lambda *_: window.close())
    wake = QTimer()
    wake.start(250)
    wake.timeout.connect(lambda: None)

    if args.open is not None:
        QTimer.singleShot(0, lambda: window.open_audio_file(args.open, confirm=False))
    return app.exec()
