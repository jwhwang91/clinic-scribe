"""macOS 마이크 권한(TCC) 확인.

python 은 앱 번들이 아니므로 권한은 앱을 띄운 프로그램(터미널, iTerm, PyCharm, VS Code 등)에 붙는다.
PyObjC(AVFoundation)가 없으면 'unknown' 을 돌려주고, 녹음 중 디지털 무음 감지로 대신 알린다.
"""

from __future__ import annotations

import os
from typing import Callable

_STATUS = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}
STATUS_LABEL = {
    "authorized": "허용됨",
    "not_determined": "아직 묻지 않음 (녹음 시작 시 요청)",
    "denied": "거부됨",
    "restricted": "제한됨 (관리 정책)",
    "unknown": "확인 불가",
}
SETTINGS_PATH = "시스템 설정 → 개인정보 보호 및 보안 → 마이크"

_BUNDLE_NAMES = {
    "com.apple.Terminal": "터미널",
    "com.googlecode.iterm2": "iTerm",
    "com.microsoft.VSCode": "Visual Studio Code",
    "com.todesktop.230313mzl4w4u92": "Cursor",
    "dev.warp.Warp-Stable": "Warp",
    "com.mitchellh.ghostty": "Ghostty",
    "com.jetbrains.pycharm": "PyCharm",
    "com.jetbrains.pycharm.ce": "PyCharm CE",
}


def authorization_status() -> str:
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    except Exception:
        return "unknown"
    try:
        return _STATUS.get(int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)), "unknown")
    except Exception:
        return "unknown"


def request_access(callback: Callable[[bool], None]) -> bool:
    """시스템 권한 대화상자를 띄운다. callback 은 다른 스레드에서 불린다. 요청하지 못하면 False."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeAudio, completion_handler(callback))
        return True
    except Exception:
        return False


def completion_handler(callback: Callable[[bool], object]) -> Callable[[bool], None]:
    """AVFoundation 완료 블록(void 반환)용 감싸개.

    PyObjC 는 블록 함수가 None 이 아닌 값을 돌려주거나 예외를 내면 Objective-C 예외로 바꾸는데,
    이 블록은 TCC 의 백그라운드 스레드에서 불리므로 그 예외가 앱 전체를 중단시킨다(SIGABRT).
    Qt Signal.emit 처럼 값을 돌려주는 함수를 그대로 넘기면 권한 대화상자에 답하는 순간 앱이 꺼진다.
    """

    def handler(granted) -> None:
        try:
            callback(bool(granted))
        except Exception:
            pass
        return None

    return handler


def host_app_name() -> str:
    """마이크 권한을 받아야 하는 앱(이 프로세스를 띄운 앱) 추정."""
    bundle = os.environ.get("__CFBundleIdentifier", "")
    if bundle in _BUNDLE_NAMES:
        return _BUNDLE_NAMES[bundle]
    if bundle.startswith("com.jetbrains."):
        return "JetBrains IDE"
    term = os.environ.get("TERM_PROGRAM", "")
    return {"Apple_Terminal": "터미널", "iTerm.app": "iTerm", "vscode": "Visual Studio Code"}.get(term, bundle or "이 앱을 실행한 터미널/IDE")


def denied_message(status: str) -> str:
    app = host_app_name()
    if status == "restricted":
        return f"마이크 사용이 관리 정책으로 제한되어 있습니다. ({SETTINGS_PATH}) 관리자에게 문의하세요."
    return (
        f"마이크 권한이 거부되었습니다. {SETTINGS_PATH} 에서 '{app}' 을(를) 켠 뒤 "
        f"'{app}' 을(를) 완전히 종료했다가 ./scripts/run.sh 로 다시 실행하세요. "
        "그동안에도 오디오 파일 불러오기로 전사·요약을 확인할 수 있습니다."
    )


def silent_input_message() -> str:
    app = host_app_name()
    return (
        "마이크 입력이 완전히 0(디지털 무음)입니다. 마이크 권한이 거부되었거나 장치가 음소거되었을 수 있습니다. "
        f"{SETTINGS_PATH} 에서 '{app}' 을(를) 허용했는지, 다른 입력 장치를 골라야 하는지 확인하세요."
    )
