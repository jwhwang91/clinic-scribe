"""마이크 권한 확인과 덮개 닫힘(클램쉘 모드) 확인 (macOS · Windows · Linux).

macOS: TCC 권한. python 은 앱 번들이 아니므로 권한은 앱을 띄운 프로그램(터미널, iTerm, PyCharm, VS Code 등)에 붙고,
  scripts/make_launcher.sh 로 만든 clinic-scribe.app 으로 열면 그 앱(clinic-scribe)에 붙는다.
  PyObjC(AVFoundation)가 없으면 'unknown' 을 돌려주고, 녹음 중 디지털 무음 감지로 대신 알린다.
  맥북은 덮개를 닫으면 보안을 위해 내장 마이크를 하드웨어로 끈다(장치는 목록에 남지만 0 만 들어온다).
Windows: 설정 → 개인 정보 및 보안 → 마이크 의 '마이크 액세스'·'데스크톱 앱이 마이크에 액세스하도록 허용'
  (레지스트리 ConsentStore)과 관리 정책을 읽는다. 꺼져 있으면 녹음해도 0 만 들어온다. 묻는 대화상자는 없다.
Linux: 앱별 마이크 권한이 없다 ('해당 없음'). 0 만 들어오면 소리 설정의 입력 장치·음소거를 안내한다.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Callable

from clinic_scribe.config import run_command

_STATUS = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}
STATUS_LABEL = {
    "authorized": "허용됨",
    "not_determined": "아직 묻지 않음 (녹음 시작 시 요청)",
    "denied": "거부됨",
    "restricted": "제한됨 (관리 정책)",
    "unknown": "확인 불가",
    "not_applicable": "해당 없음",  # Linux: 앱별 마이크 권한이 없다
}
SETTINGS_PATH = "시스템 설정 → 개인정보 보호 및 보안 → 마이크"  # macOS
WINDOWS_SETTINGS = "설정 → 개인 정보 및 보안 → 마이크"
WINDOWS_TOGGLES = "'마이크 액세스'와 '데스크톱 앱이 마이크에 액세스하도록 허용'"
_CONSENT_KEY = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
_POLICY_KEY = r"SOFTWARE\Policies\Microsoft\Windows\AppPrivacy"
APP_BUNDLE_ID = "local.clinic-scribe"  # scripts/make_launcher.sh 가 만드는 앱

_BUNDLE_NAMES = {
    APP_BUNDLE_ID: "clinic-scribe",
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
    if sys.platform == "win32":
        return _windows_status()
    if sys.platform != "darwin":
        return "not_applicable"
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    except Exception:
        return "unknown"
    try:
        return _STATUS.get(int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)), "unknown")
    except Exception:
        return "unknown"


def _windows_status() -> str:
    """Windows 개인 정보 설정의 마이크 허용 여부 (관리 정책 → 이 장치 → 이 사용자 → 데스크톱 앱 순)."""
    try:
        winreg = importlib.import_module("winreg")
    except ImportError:
        return "unknown"

    def read(root, path: str, name: str = "Value"):
        try:
            with winreg.OpenKey(root, path) as key:
                return winreg.QueryValueEx(key, name)[0]
        except OSError:
            return None

    if read(winreg.HKEY_LOCAL_MACHINE, _POLICY_KEY, "LetAppsAccessMicrophone") == 2:  # 2 = 관리자가 강제로 막음
        return "restricted"
    values = [
        read(winreg.HKEY_LOCAL_MACHINE, _CONSENT_KEY),  # 이 장치의 마이크 액세스
        read(winreg.HKEY_CURRENT_USER, _CONSENT_KEY),  # 이 사용자의 마이크 액세스
        read(winreg.HKEY_CURRENT_USER, _CONSENT_KEY + r"\NonPackaged"),  # 데스크톱 앱 허용
    ]
    if "Deny" in values:
        return "denied"
    return "authorized" if "Allow" in values else "unknown"


def request_access(callback: Callable[[bool], None]) -> bool:
    """시스템 권한 대화상자를 띄운다 (macOS 만). callback 은 다른 스레드에서 불린다. 요청하지 못하면 False."""
    if sys.platform != "darwin":
        return False
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


def launched_as_app() -> bool:
    """clinic-scribe.app (Finder·Dock·Spotlight) 으로 열었는지."""
    return os.environ.get("__CFBundleIdentifier", "") == APP_BUNDLE_ID


def permission_tooltip() -> str:
    """화면 위쪽 '마이크 권한' 표시에 붙는 설명."""
    if sys.platform == "win32":
        return f"확인 위치: {WINDOWS_SETTINGS}\n{WINDOWS_TOGGLES} 이 켜져 있어야 합니다."
    if sys.platform != "darwin":
        return "Linux 는 앱별 마이크 권한이 없습니다. 소리가 들어오지 않으면 소리 설정의 입력 장치·음소거를 확인하세요."
    return f"권한을 받는 앱: {host_app_name()}\n확인 위치: {SETTINGS_PATH}"


def denied_message(status: str) -> str:
    if sys.platform == "win32":
        if status == "restricted":
            return "마이크 사용이 관리 정책(그룹 정책)으로 막혀 있습니다. 관리자에게 문의하세요."
        return (f"Windows 마이크 권한이 꺼져 있습니다. {WINDOWS_SETTINGS} 에서 {WINDOWS_TOGGLES} 을 켠 뒤 다시 녹음하세요. "
                "그동안에도 오디오 파일 불러오기로 전사·요약을 확인할 수 있습니다.")
    app = host_app_name()
    if status == "restricted":
        return f"마이크 사용이 관리 정책으로 제한되어 있습니다. ({SETTINGS_PATH}) 관리자에게 문의하세요."
    relaunch = "다시 여세요" if launched_as_app() else f"{run_command()} 로 다시 실행하세요"
    return (
        f"마이크 권한이 거부되었습니다. {SETTINGS_PATH} 에서 '{app}' 을(를) 켠 뒤 "
        f"'{app}' 을(를) 완전히 종료했다가 {relaunch}. "
        "그동안에도 오디오 파일 불러오기로 전사·요약을 확인할 수 있습니다."
    )


def silent_input_message(device_name: str = "") -> str:
    if sys.platform == "win32":
        return ("마이크 입력이 완전히 0(디지털 무음)입니다. Windows 마이크 권한이 꺼졌거나(" + WINDOWS_SETTINGS + "), "
                "장치가 음소거되었거나, 다른 프로그램이 마이크를 독점하고 있을 수 있습니다. "
                "확인하거나 위쪽 마이크 목록에서 다른 입력 장치를 고르세요.")
    if sys.platform != "darwin":
        return ("마이크 입력이 완전히 0(디지털 무음)입니다. 소리 설정(PipeWire·PulseAudio)에서 입력 장치가 음소거되었거나 "
                "다른 장치가 골라져 있을 수 있습니다. 확인하거나 위쪽 마이크 목록에서 다른 입력 장치를 고르세요.")
    if is_builtin_mic(device_name) and lid_closed():
        return lid_closed_message([])
    app = host_app_name()
    return (
        "마이크 입력이 완전히 0(디지털 무음)입니다. 마이크 권한이 거부되었거나 장치가 음소거되었을 수 있습니다. "
        f"{SETTINGS_PATH} 에서 '{app}' 을(를) 허용했는지, 다른 입력 장치를 골라야 하는지 확인하세요."
    )


def lid_closed() -> bool:
    """맥북 덮개가 닫혀 있는지(클램쉘 모드). 확인할 수 없거나 맥북이 아니면 False."""
    if sys.platform != "darwin":
        return False
    ioreg = "/usr/sbin/ioreg" if os.path.exists("/usr/sbin/ioreg") else "ioreg"
    try:
        out = subprocess.run([ioreg, "-r", "-k", "AppleClamshellState", "-d", "4"],
                             capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return '"AppleClamshellState" = Yes' in out


def is_builtin_mic(name: str) -> bool:
    """맥북 내장 마이크 ("MacBook Pro 마이크", "MacBook Air Microphone" 등)."""
    return "macbook" in name.lower()


def quoted(name: str) -> str:
    """장치 이름을 따옴표로 감싼다. 이름에 이미 따옴표가 있으면('iPhone' 마이크) 그대로 둔다."""
    return name if any(q in name for q in "'‘’\"“”") else f"'{name}'"


def lid_closed_message(alternatives: list[str]) -> str:
    base = "맥북 덮개가 닫혀 있어(클램쉘 모드) 내장 마이크가 꺼져 있습니다. 맥북은 덮개를 닫으면 보안을 위해 내장 마이크를 하드웨어로 끕니다. "
    if alternatives:
        return base + f"위쪽 마이크 목록에서 {quoted(alternatives[0])} 같은 다른 입력 장치를 고르거나 덮개를 여세요."
    return (base + "덮개를 열거나, 외장 마이크(이어폰·USB 마이크, 모니터·웹캠 마이크, 아이폰 연속성 마이크 등)를 연결한 뒤 "
            "마이크 목록을 다시 펼쳐 고르세요.")
