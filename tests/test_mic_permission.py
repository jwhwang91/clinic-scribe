"""마이크 권한 요청 콜백.

macOS 권한 대화상자에 답하면 AVFoundation 이 완료 블록을 부른다. 그 블록이 None 이 아닌 값을 돌려주면
PyObjC 가 앱을 중단시킨다(SIGABRT, 실제로 발생했던 문제). 여기서 그 계약을 확인한다.
"""

import subprocess
import sys
import textwrap

import pytest

from clinic_scribe import mic_permission


def test_completion_handler_returns_none_and_passes_bool():
    seen = []

    def emit_like(granted):  # Qt Signal.emit 처럼 값을 돌려주는 함수
        seen.append(granted)
        return True

    handler = mic_permission.completion_handler(emit_like)
    assert handler(1) is None
    assert seen == [True]


def test_completion_handler_swallows_callback_errors():
    def broken(_granted):
        raise RuntimeError("UI 가 이미 닫힘")

    assert mic_permission.completion_handler(broken)(False) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS AVFoundation 권한 요청")
def test_real_avfoundation_callback_does_not_abort():
    """실제 AVFoundation 호출. 권한이 이미 정해진 경우에만 실행한다(이때는 대화상자 없이 바로 답이 온다)."""
    status = mic_permission.authorization_status()
    if status not in ("authorized", "denied"):
        pytest.skip(f"마이크 권한이 아직 정해지지 않아({status}) 실행하면 권한 대화상자가 뜨므로 건너뜀")
    code = textwrap.dedent(
        """
        import threading
        from clinic_scribe import mic_permission
        done = threading.Event()
        result = []
        def emit_like(granted):
            result.append(granted)
            done.set()
            return True  # 예전 버그: 값을 돌려주면 SIGABRT
        assert mic_permission.request_access(emit_like)
        assert done.wait(10), "완료 콜백이 오지 않음"
        print("granted", result[0])
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert proc.stdout.strip() == f"granted {status == 'authorized'}"


# ioreg -r -k AppleClamshellState 실제 출력의 해당 줄 (덮개 닫힘)
IOREG_CLOSED = '+-o IOPMrootDomain  <class IOPMrootDomain>\n    "AppleClamshellState" = Yes\n    "AppleClamshellCausesSleep" = No\n'


def test_lid_closed_reads_clamshell_state(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")  # 맥북 덮개 확인 (다른 OS 는 늘 False)
    def fake_run(cmd, **kwargs):
        assert cmd[0].endswith("ioreg") and "AppleClamshellState" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=fake_run.out, stderr="")

    monkeypatch.setattr(mic_permission.subprocess, "run", fake_run)
    fake_run.out = IOREG_CLOSED
    assert mic_permission.lid_closed()
    fake_run.out = IOREG_CLOSED.replace("= Yes", "= No")
    assert not mic_permission.lid_closed()
    fake_run.out = ""  # 데스크톱 맥 등 덮개가 없는 경우
    assert not mic_permission.lid_closed()

    def broken(cmd, **kwargs):
        raise OSError("ioreg 없음")

    monkeypatch.setattr(mic_permission.subprocess, "run", broken)
    assert not mic_permission.lid_closed()


def test_lid_closed_messages(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert mic_permission.is_builtin_mic("MacBook Pro 마이크") and mic_permission.is_builtin_mic("MacBook Air Microphone")
    assert not mic_permission.is_builtin_mic("‘iPhone (2)’ 마이크") and not mic_permission.is_builtin_mic("USB Audio")
    assert "‘iPhone (2)’ 마이크" in mic_permission.lid_closed_message(["‘iPhone (2)’ 마이크"])
    assert "외장 마이크" in mic_permission.lid_closed_message([])
    monkeypatch.setattr(mic_permission, "lid_closed", lambda: True)
    assert "덮개" in mic_permission.silent_input_message("MacBook Pro 마이크")  # 권한 문제가 아니라 덮개 문제로 안내
    assert "권한" in mic_permission.silent_input_message("USB Audio")


def test_app_bundle_launch_names_the_app(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("__CFBundleIdentifier", mic_permission.APP_BUNDLE_ID)  # clinic-scribe.app 으로 열었을 때
    assert mic_permission.launched_as_app()
    assert mic_permission.host_app_name() == "clinic-scribe"
    message = mic_permission.denied_message("denied")
    assert "'clinic-scribe'" in message and "다시 여세요" in message and "run.sh" not in message
    monkeypatch.setenv("__CFBundleIdentifier", "com.apple.Terminal")
    assert not mic_permission.launched_as_app()
    assert "run.sh" in mic_permission.denied_message("denied")
