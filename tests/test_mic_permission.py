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
