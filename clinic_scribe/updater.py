"""자동 업데이트: GitHub 에 올라온 새 버전 가운데 테스트(CI)를 통과한 버전을 받아 두었다가, 앱을 열 때 적용한다.

- 앱이 열려 있는 동안 30분마다(처음은 연 지 1분 뒤) 새 커밋을 받아 두고(git fetch), GitHub Actions 의 'tests' 가
  성공한 가장 새 커밋을 '준비됨'으로 적어 둔다. 화면 위쪽에 '새 버전' 단추가 나온다.
- 앱을 열 때(__main__, 화면 라이브러리를 불러오기 전에) 준비된 버전으로 fast-forward 하고, requirements.txt 가 바뀌었으면
  가상환경 패키지를 맞춘다. 실패하면 원래 버전으로 되돌리고 그대로 연다. 녹음 중에는 버전이 바뀌지 않는다.
- 새 버전이 창을 띄우지 못하고 끝났으면(다음에 열 때 안다) 이전 버전으로 되돌리고, 그 버전은 다시 받지 않는다.
- 건드리지 않는 경우: git 으로 받은 폴더가 아님, main 이 아님, 이 컴퓨터에서 고친 파일이 있음(개발 중인 컴퓨터),
  GitHub 과 기록이 갈라짐, 테스트 결과를 확인할 수 없음.
진료 기록·설정·의사 목소리·모델은 repo 밖 사용자 폴더에 있어 업데이트와 상관없이 그대로 남는다.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from clinic_scribe import config, procutil

REPO = Path(__file__).resolve().parent.parent
BRANCH = "main"
WORKFLOW = "tests"  # .github/workflows/tests.yml 의 name
APP_BUNDLE_ID = "local.clinic-scribe"  # macOS 앱 아이콘 (mic_permission.APP_BUNDLE_ID 와 같게)
CHECK_INTERVAL_S = 30 * 60
FIRST_CHECK_S = 60
FETCH_TIMEOUT_S = 60
API_TIMEOUT_S = 10
PIP_TIMEOUT_S = 20 * 60
WAIT_RESTART_S = 20  # 다시 열 때 앞 창이 닫히기를 기다리는 시간
MAX_VERIFY = 10  # 새 커밋이 많아도 GitHub 에 테스트 결과를 묻는 것은 최근 10개까지
_GITHUB = re.compile(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$")


@dataclass(frozen=True)
class Version:
    commit: str
    date: str  # 커밋 시각 (YYYY-MM-DD HH:MM)
    subject: str

    @property
    def short(self) -> str:
        return self.commit[:7]

    def describe(self) -> str:
        return f"{self.date} · {self.short}"


@dataclass(frozen=True)
class CheckResult:
    ready: Version | None  # 적용할 수 있는 새 버전 (테스트 통과)
    waiting: int  # 아직 테스트 중이거나 통과하지 못한 더 새 커밋 수
    error: str  # 확인하지 못한 이유 ('' 이면 확인함)


# ---------------------------------------------------------------------------- 상태 파일
def state_path() -> Path:
    return config.settings_path().parent / "update.json"


def load_state() -> dict:
    try:
        state = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------- git
def git_executable() -> str | None:
    found = shutil.which("git")
    if found or sys.platform != "win32":
        return found
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramW6432", ""),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        candidate = Path(base) / "Git" / "cmd" / "git.exe"
        if base and candidate.is_file():
            return str(candidate)
    return None


def _git(*args: str, timeout: float = 30) -> tuple[int, str]:
    """(종료 코드, 출력). 로그인 창이나 질문을 띄우지 않는다."""
    git = git_executable()
    if git is None:
        return 127, "git 없음"
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    try:
        done = procutil.run([git, *args], cwd=str(REPO), env=env, timeout=timeout)
    except (OSError, procutil.ProcessTimeout) as exc:
        return 1, str(exc)
    out = done.stdout if done.returncode == 0 else (done.stderr or done.stdout)  # 성공했을 때 stderr(경고)는 결과가 아니다
    return done.returncode, out.decode("utf-8", errors="replace").strip()


def _rev(name: str) -> str:
    code, out = _git("rev-parse", "--verify", "--quiet", f"{name}^{{commit}}")
    return out if code == 0 else ""


def blocker() -> str:
    """자동 업데이트를 할 수 없는 이유 ('' 이면 할 수 있다)."""
    if os.environ.get("CLINIC_SCRIBE_NO_UPDATE"):
        return "이 실행에서는 자동 업데이트를 끕니다 (CLINIC_SCRIBE_NO_UPDATE)"
    if git_executable() is None:
        return "git 이 없습니다. README 의 설치 방법대로 git 을 설치하세요"
    if not (REPO / ".git").exists():
        return "git 으로 받은 폴더가 아닙니다. git clone 으로 받아야 자동 업데이트가 됩니다"
    code, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if code != 0:
        return "git 상태를 읽지 못했습니다"
    if branch != BRANCH:
        return f"{BRANCH} 이 아닌 '{branch}' 에 있습니다"
    code, changed = _git("status", "--porcelain", "--untracked-files=no")
    if code != 0:
        return "git 상태를 읽지 못했습니다"
    if changed:
        return "이 컴퓨터에서 고친 파일이 있어 자동으로 바꾸지 않습니다 (개발 중인 컴퓨터)"
    return ""


def version_of(commit: str = "HEAD") -> Version | None:
    code, out = _git("log", "-1", "--format=%H%x1f%cd%x1f%s", "--date=format:%Y-%m-%d %H:%M", commit)
    if code != 0 or out.count("\x1f") != 2:
        return None
    sha, date, subject = out.split("\x1f")
    return Version(sha, date, subject)


def changes(to: str) -> list[str]:
    """지금 버전에서 to 까지 바뀐 것 (커밋 제목, 새 것부터)."""
    code, out = _git("log", "--first-parent", "--format=%s", f"HEAD..{to}")
    return [line for line in out.splitlines() if line.strip()] if code == 0 else []


def github_repo() -> tuple[str, str] | None:
    code, url = _git("remote", "get-url", "origin")
    match = _GITHUB.search(url) if code == 0 else None
    return (match.group(1), match.group(2)) if match else None


# ---------------------------------------------------------------------------- GitHub 테스트 결과
def _api_json(url: str) -> dict | None:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "clinic-scribe-updater"}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=API_TIMEOUT_S) as response:
            return json.load(response)
    except (OSError, ValueError):
        pass
    curl = shutil.which("curl")  # 인증서 문제로 Python 이 못 읽으면 시스템 curl 로 (macOS·Windows 10+ 기본 포함)
    if curl is None:
        return None
    try:
        done = procutil.run([curl, "-fsSL", "--max-time", str(API_TIMEOUT_S), "-H", f"Accept: {headers['Accept']}",
                             "-A", headers["User-Agent"], url], timeout=API_TIMEOUT_S + 5)
        return json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, ValueError, procutil.ProcessTimeout):
        return None


def ci_result(repo: tuple[str, str], commit: str) -> str:
    """그 커밋의 'tests' 결과: success · failure · pending(아직 테스트 중·시작 전) · error(확인 못 함)."""
    data = _api_json(f"https://api.github.com/repos/{repo[0]}/{repo[1]}/actions/runs?head_sha={commit}&per_page=30")
    if data is None:
        return "error"
    runs = [r for r in data.get("workflow_runs", []) if r.get("name") == WORKFLOW and r.get("head_sha") == commit]
    on_main = [r for r in runs if r.get("head_branch") == BRANCH and r.get("event") == "push"] or runs
    if any(r.get("status") == "completed" and r.get("conclusion") == "success" for r in on_main):
        return "success"
    if not on_main or any(r.get("status") != "completed" for r in on_main):
        return "pending"
    return "failure"


# ---------------------------------------------------------------------------- 확인 (앱이 열려 있을 때, 따로 도는 스레드에서)
def check() -> CheckResult:
    """새 커밋을 받아 두고, 테스트를 통과한 가장 새 커밋을 '준비됨'으로 적는다 (적용은 다음에 열 때)."""
    reason = blocker()
    if reason:
        return CheckResult(None, 0, reason)
    code, _ = _git("fetch", "--quiet", "origin", BRANCH, timeout=FETCH_TIMEOUT_S)
    if code != 0:
        return CheckResult(None, 0, "GitHub 에 연결하지 못했습니다 (인터넷 연결을 확인하세요)")
    target = f"origin/{BRANCH}"
    if _git("merge-base", "--is-ancestor", "HEAD", target)[0] != 0:
        return CheckResult(None, 0, "이 컴퓨터의 기록이 GitHub 과 달라 자동으로 바꾸지 않습니다")
    code, out = _git("rev-list", "--first-parent", f"HEAD..{target}")
    commits = out.split() if code == 0 else []
    state = load_state()
    state["checked_at"] = _now()
    if not commits:
        state["ready"] = ""
        save_state(state)
        return CheckResult(None, 0, "")
    repo = github_repo()
    if repo is None:
        save_state(state)
        return CheckResult(None, len(commits), "GitHub 저장소가 아니라 테스트 결과를 확인할 수 없습니다")
    bad = set(state.get("bad", []))
    known = {sha: result for sha, result in state.get("verified", {}).items() if sha in commits}
    ready, waiting, error = "", 0, ""
    for sha in commits[:MAX_VERIFY]:
        result = "failure" if sha in bad else known.get(sha) or ci_result(repo, sha)
        if result == "error":
            error = "GitHub 에서 테스트 결과를 확인하지 못했습니다"
            break
        if result in ("success", "failure"):
            known[sha] = result  # 끝난 결과는 다시 묻지 않는다
        if result == "success":
            ready = sha
            break
        waiting += 1
    state.update(verified=known, ready=ready)
    save_state(state)
    return CheckResult(version_of(ready) if ready else None, waiting, error)


def ready_version() -> Version | None:
    """받아 두었고 아직 적용하지 않은 새 버전."""
    ready = load_state().get("ready", "")
    return version_of(ready) if ready and ready != _rev("HEAD") else None


# ---------------------------------------------------------------------------- 적용 (앱을 열 때, 화면 라이브러리를 불러오기 전에)
def apply_on_start() -> None:
    """다시 열기를 기다리고, 창을 띄우지 못한 새 버전을 되돌리고, 준비된 버전을 적용한다.

    무엇이 잘못되어도 예외를 내지 않고 지금 버전으로 연다. 결과는 update.json 에 남아 화면이 알려 준다.
    """
    try:
        _apply_on_start()
    except Exception as exc:  # 업데이트 때문에 앱이 열리지 않는 일은 없게
        state = load_state()
        state["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
        save_state(state)


def _auto_update_enabled() -> bool:
    return bool(getattr(config.Settings.load(), "auto_update", True))


def _apply_on_start() -> None:
    state = load_state()
    restart = state.pop("restart", None)
    apply_now = bool(state.pop("apply_now", False))
    if isinstance(restart, dict) and time.time() - float(restart.get("at", 0)) < 120:
        _wait_for_exit(int(restart.get("wait_pid", 0)), WAIT_RESTART_S)
    if blocker():
        save_state(state)
        return
    head = _rev("HEAD")
    pending = state.pop("pending_start", None)
    if isinstance(pending, dict) and pending.get("to") == head and pending.get("from"):
        # 지난번에 적용한 새 버전이 창을 띄우지 못하고 끝났다: 이전 버전으로 되돌리고 그 버전은 다시 받지 않는다
        if _git("reset", "--keep", "--quiet", pending["from"])[0] == 0:
            state["bad"] = sorted(set(state.get("bad", [])) | {head})
            state["rolled_back"] = {"from": head, "to": pending["from"], "at": _now()}
        save_state(state)
        return
    ready = state.get("ready", "")
    if not ready or ready == head or ready in state.get("bad", []) or not (apply_now or _auto_update_enabled()):
        if ready == head:
            state["ready"] = ""
        save_state(state)
        return
    if _git("merge-base", "--is-ancestor", "HEAD", ready)[0] != 0:  # 받아 둔 버전이 지금 버전 뒤에 이어지지 않는다
        state["ready"] = ""
        save_state(state)
        return
    lock = _lock()
    if lock is None:  # 다른 창이 업데이트하는 중
        save_state(state)
        return
    try:
        old_requirements = _git("show", "HEAD:requirements.txt")[1]
        code, out = _git("merge", "--ff-only", "--quiet", ready, timeout=120)
        if code != 0:
            state["last_error"] = f"새 버전을 적용하지 못했습니다: {out[-200:]}"
            return
        if _git("show", "HEAD:requirements.txt")[1] != old_requirements and not _pip_install():
            _git("reset", "--keep", "--quiet", head)
            state["bad"] = sorted(set(state.get("bad", [])) | {ready})
            state["last_error"] = "새 버전에 필요한 구성 요소를 설치하지 못해 이전 버전으로 열었습니다"
            return
        state.update(ready="", pending_start={"from": head, "to": ready},
                     applied={"from": head, "to": ready, "at": _now()})
        state.pop("last_error", None)
    finally:
        save_state(state)
        _unlock(lock)


def mark_started() -> None:
    """창이 떴다 (app.main): 이번 버전은 되돌리지 않는다."""
    state = load_state()
    if state.pop("pending_start", None) is not None:
        save_state(state)


def take_notice() -> dict | None:
    """방금 적용했거나 되돌린 업데이트 (화면에 한 번 알린다): {"kind": "applied"|"rolled_back", "from", "to"}."""
    state = load_state()
    for kind in ("rolled_back", "applied"):
        notice = state.get(kind)
        if isinstance(notice, dict):
            state.pop("applied", None)
            state.pop("rolled_back", None)
            save_state(state)
            return {"kind": kind, **notice}
    return None


def _pip_install() -> bool:
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe" and python.with_name("python.exe").is_file():
        python = python.with_name("python.exe")
    if sys.platform == "darwin":  # 몇십 초 걸릴 수 있어 알림을 띄운다 (창은 아직 없다)
        subprocess.run(["osascript", "-e", 'display notification "새 버전에 필요한 구성 요소를 설치하고 있습니다" '
                        'with title "clinic-scribe 업데이트"'], capture_output=True, check=False)
    try:
        done = procutil.run([str(python), "-m", "pip", "--disable-pip-version-check", "install", "-q",
                             "-r", str(REPO / "requirements.txt")], cwd=str(REPO), capture_stdout=False,
                            timeout=PIP_TIMEOUT_S)
    except (OSError, procutil.ProcessTimeout):
        return False
    return done.returncode == 0


def _lock() -> int | None:
    path = state_path().with_name("update.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime < 30 * 60:
                    return None
                path.unlink()  # 30분 넘은 잠금: 앞 업데이트가 비정상 종료했다
            except OSError:
                return None
    return None


def _unlock(fd: int) -> None:
    os.close(fd)
    try:
        state_path().with_name("update.lock").unlink()
    except OSError:
        pass


def _wait_for_exit(pid: int, seconds: float) -> None:
    """다시 열 때: 앞 창(프로세스)이 닫히기를 기다린다 (같은 파일을 쓰지 않게)."""
    if pid <= 0 or pid == os.getpid():
        return
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE (os.kill 은 Windows 에서 프로세스를 끝낸다)
        if handle:
            kernel32.WaitForSingleObject(handle, int(seconds * 1000))
            kernel32.CloseHandle(handle)
        return
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        time.sleep(0.2)


# ---------------------------------------------------------------------------- 다시 열기 (화면의 '다시 열어 적용')
def restart_command() -> list[str]:
    """지금 앱을 새로 여는 명령. macOS 앱 아이콘으로 열었으면 같은 앱으로 (마이크 권한이 그 앱에 있다)."""
    if sys.platform == "darwin" and os.environ.get("__CFBundleIdentifier", "") == APP_BUNDLE_ID:
        return ["open", "-n", "-b", APP_BUNDLE_ID]
    python = Path(sys.executable)
    if sys.platform == "win32" and python.with_name("pythonw.exe").is_file():
        python = python.with_name("pythonw.exe")  # 콘솔 창 없이
    return [str(python), "-m", "clinic_scribe"]


def request_restart(apply: bool = True) -> None:
    """새 창을 띄운다. 새 창은 이 창이 닫히기를 기다렸다가 (apply 면) 준비된 버전을 적용하고 연다. 이 창은 부른 쪽이 닫는다."""
    state = load_state()
    state["restart"] = {"wait_pid": os.getpid(), "at": time.time()}
    if apply:
        state["apply_now"] = True
    save_state(state)
    options: dict = {"start_new_session": True} if sys.platform != "win32" else {
        "creationflags": getattr(subprocess, "DETACHED_PROCESS", 0x8) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    subprocess.Popen(restart_command(), cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True, **options)
