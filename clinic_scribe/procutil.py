"""외부 프로그램(whisper-cli, claude) 실행: 취소·시간 제한·앱 종료 시 정리.

출력은 호출한 쪽에만 돌려주고 어디에도 기록하지 않는다(전사문·요약 원문이 들어 있을 수 있음).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

_active: set[subprocess.Popen] = set()
_lock = threading.Lock()


class Cancelled(Exception):
    """사용자가 작업을 취소했거나 새 세션이 시작되었다."""


class ProcessTimeout(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"{seconds:.0f}초 안에 끝나지 않았습니다.")
        self.seconds = seconds


@dataclass
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def run(
    cmd: list[str],
    *,
    stdin_data: bytes | None = None,
    capture_stdout: bool = True,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    cancel: threading.Event | None = None,
) -> ProcessResult:
    """shell=False 로 실행하고, 0.2초마다 취소·시간 제한을 확인한다.

    별도 프로세스 그룹으로 띄워 취소 시 자식 프로세스까지 함께 정리한다.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        shell=False,
        **popen_group_options(),
    )
    with _lock:
        _active.add(proc)
    deadline = time.monotonic() + timeout if timeout else None
    pending_input = stdin_data
    try:
        while True:
            try:
                out, err = proc.communicate(input=pending_input, timeout=0.2)
                return ProcessResult(proc.returncode, out or b"", err or b"")
            except subprocess.TimeoutExpired:
                pending_input = None  # 입력은 첫 communicate 에서 이미 전달됨
            if cancel is not None and cancel.is_set():
                _terminate(proc)
                raise Cancelled()
            if deadline is not None and time.monotonic() > deadline:
                _terminate(proc)
                raise ProcessTimeout(timeout or 0)
    finally:
        with _lock:
            _active.discard(proc)


def popen_group_options() -> dict:
    """자식 프로세스까지 한꺼번에 끌 수 있게 새 프로세스 그룹으로 띄우는 옵션.

    Windows 는 콘솔 창도 만들지 않는다 (창 없이 연 앱이 whisper-cli·claude 를 부를 때마다 검은 창이 뜨지 않게).
    """
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200) | getattr(subprocess, "CREATE_NO_WINDOW", 0x8000000)
        return {"creationflags": flags}
    return {"start_new_session": True}


def _kill_group(proc: subprocess.Popen, force: bool) -> None:
    if sys.platform == "win32":
        # taskkill /T: 자식(claude.cmd 가 띄운 node 등)까지. Windows 콘솔 프로그램에는 SIGTERM 같은 정중한 종료가 없다.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x8000000), check=False)
        return
    os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        _kill_group(proc, force=False)
    except OSError:  # ProcessLookupError·PermissionError 포함
        proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            _kill_group(proc, force=True)
        except OSError:
            proc.kill()
        proc.wait(timeout=2)


def terminate_tree(proc: subprocess.Popen) -> None:
    """오래 켜 두는 외부 프로그램(로컬 요약 서버)을 자식 프로세스까지 끈다."""
    try:
        _terminate(proc)
    except (OSError, subprocess.SubprocessError):
        pass


def terminate_all() -> None:
    """앱 종료 시 남은 외부 프로세스를 정리한다."""
    with _lock:
        procs = list(_active)
    for proc in procs:
        try:
            _terminate(proc)
        except Exception:
            pass


def tail(data: bytes, lines: int = 6) -> str:
    """오류 표시용 마지막 몇 줄 (whisper-cli/claude 의 stderr 는 진단 메시지만 담는다)."""
    text = data.decode("utf-8", errors="replace").strip()
    return "\n".join(text.splitlines()[-lines:])
