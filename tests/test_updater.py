"""자동 업데이트: 테스트를 통과한 버전만 받아 앱을 열 때 적용, 창을 못 띄운 버전은 되돌리기, 개발 중인 컴퓨터는 건드리지 않기.

GitHub 대신 이 컴퓨터에 만든 git 저장소(remote)와 그것을 받은 '진료실 컴퓨터'(clinic)로 실제 git 을 돌린다.
GitHub 의 테스트 결과(ci_result)만 가짜로 정한다.
"""

import json
import os
import subprocess
import sys
import time

import pytest

from clinic_scribe import config, updater

IDENTITY = {"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com"}


def git(cwd, *args) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env={**os.environ, **IDENTITY})
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


class Repos:
    """dev(개발하는 컴퓨터) → remote(GitHub 대신) → clinic(자동 업데이트하는 컴퓨터)."""

    def __init__(self, root):
        self.remote, self.dev, self.clinic = root / "remote.git", root / "dev", root / "clinic"
        git(root, "init", "-q", "--bare", "-b", "main", str(self.remote))
        git(root, "init", "-q", "-b", "main", str(self.dev))
        (self.dev / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        self.first = self.commit("첫 버전", app="1")
        git(self.dev, "remote", "add", "origin", str(self.remote))
        git(self.dev, "push", "-q", "origin", "main")
        git(root, "clone", "-q", str(self.remote), str(self.clinic))
        self.results: dict[str, str] = {}  # 커밋 → GitHub 테스트 결과

    def commit(self, subject: str, app: str | None = None, requirements: str | None = None) -> str:
        if app is not None:
            (self.dev / "app.txt").write_text(app, encoding="utf-8")
        if requirements is not None:
            (self.dev / "requirements.txt").write_text(requirements, encoding="utf-8")
        git(self.dev, "add", "-A")
        git(self.dev, "commit", "-q", "-m", subject)
        return git(self.dev, "rev-parse", "HEAD")

    def push(self, subject: str, result: str = "success", **files) -> str:
        sha = self.commit(subject, **files)
        git(self.dev, "push", "-q", "origin", "main")
        self.results[sha] = result
        return sha

    def head(self) -> str:
        return git(self.clinic, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path, monkeypatch):
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git 이 필요합니다")
    monkeypatch.delenv("CLINIC_SCRIBE_NO_UPDATE", raising=False)
    made = Repos(tmp_path)
    monkeypatch.setattr(updater, "REPO", made.clinic)
    monkeypatch.setattr(updater, "github_repo", lambda: ("owner", "clinic-scribe"))
    monkeypatch.setattr(updater, "ci_result", lambda repo, sha: made.results.get(sha, "pending"))
    return made


def test_takes_only_versions_that_passed_tests_and_applies_them_on_start(repos):
    second = repos.push("두 번째: 끼어든 말 나누기", app="2")
    newest = repos.push("세 번째: 아직 테스트 중", app="3", result="pending")
    result = updater.check()
    assert not result.error and result.ready.commit == second and result.waiting == 1
    assert repos.head() == repos.first  # 받아만 두고, 적용은 앱을 열 때
    assert updater.ready_version().commit == second and updater.changes(second) == ["두 번째: 끼어든 말 나누기"]

    updater.apply_on_start()
    assert repos.head() == second and (repos.clinic / "app.txt").read_text(encoding="utf-8") == "2"
    assert updater.load_state()["pending_start"] == {"from": repos.first, "to": second}
    updater.mark_started()  # 창이 떴다
    assert "pending_start" not in updater.load_state()
    notice = updater.take_notice()
    assert notice["kind"] == "applied" and notice["to"] == second and updater.take_notice() is None

    repos.results[newest] = "success"
    assert updater.check().ready.commit == newest
    updater.apply_on_start()
    assert repos.head() == newest
    assert updater.check() == updater.CheckResult(None, 0, "")  # 최신


def test_version_that_could_not_open_is_rolled_back_and_skipped(repos):
    broken = repos.push("창을 못 띄우는 버전", app="2")
    updater.check()
    updater.apply_on_start()
    assert repos.head() == broken
    # mark_started 없이 끝났다 (창을 띄우기 전에 죽음) → 다음에 열 때 되돌린다
    updater.apply_on_start()
    assert repos.head() == repos.first
    state = updater.load_state()
    assert broken in state["bad"] and updater.take_notice()["kind"] == "rolled_back"
    result = updater.check()
    assert result.ready is None and result.waiting == 1  # 테스트를 통과했어도 다시 받지 않는다
    fixed = repos.push("고친 버전", app="3")
    assert updater.check().ready.commit == fixed


def test_new_requirements_are_installed_or_the_update_is_undone(repos, monkeypatch):
    installs = []
    monkeypatch.setattr(updater, "_pip_install", lambda: installs.append(1) or False)
    needs = repos.push("새 구성 요소가 필요한 버전", app="2", requirements="numpy\ncomtypes\n")
    updater.check()
    updater.apply_on_start()
    assert installs and repos.head() == repos.first  # 설치 실패: 이전 버전으로 연다
    state = updater.load_state()
    assert needs in state["bad"] and "구성 요소" in state["last_error"]

    monkeypatch.setattr(updater, "_pip_install", lambda: installs.append(2) or True)
    plain = repos.push("코드만 바뀐 버전", app="3")  # requirements 는 앞 버전과 같다(설치 필요)
    updater.check()
    updater.apply_on_start()
    assert repos.head() == plain and installs == [1, 2]


def test_leaves_a_development_computer_alone(repos, tmp_path, monkeypatch):
    repos.push("새 버전", app="2")
    (repos.clinic / "app.txt").write_text("이 컴퓨터에서 고침", encoding="utf-8")
    assert "고친 파일" in updater.blocker()
    assert "고친 파일" in updater.check().error
    updater.apply_on_start()
    assert repos.head() == repos.first
    git(repos.clinic, "checkout", "-q", "--", "app.txt")

    git(repos.clinic, "checkout", "-q", "-b", "실험")
    assert "'실험'" in updater.blocker()
    git(repos.clinic, "checkout", "-q", "main")
    assert updater.blocker() == ""

    monkeypatch.setenv("CLINIC_SCRIBE_NO_UPDATE", "1")
    assert "CLINIC_SCRIBE_NO_UPDATE" in updater.blocker()
    monkeypatch.delenv("CLINIC_SCRIBE_NO_UPDATE")
    monkeypatch.setattr(updater, "REPO", tmp_path / "zip")  # git 이 아닌 폴더 (zip 으로 받음)
    (tmp_path / "zip").mkdir()
    assert "git clone" in updater.blocker()


def test_auto_update_off_applies_only_when_asked(repos, monkeypatch):
    settings = config.Settings()
    settings.auto_update = False
    settings.save()
    second = repos.push("새 버전", app="2")
    updater.check()
    updater.apply_on_start()
    assert repos.head() == repos.first  # 자동 업데이트를 껐다

    launched = []
    with monkeypatch.context() as m:  # 새 창을 실제로 띄우지 않는다
        m.setattr(updater.subprocess, "Popen", lambda cmd, **kw: launched.append((cmd, kw)))
        updater.request_restart(apply=True)  # 업데이트 창의 '다시 열어 적용'
    assert launched and launched[0][0] == updater.restart_command()
    state = updater.load_state()
    assert state["restart"]["wait_pid"] == os.getpid() and state["apply_now"] is True
    started = time.monotonic()
    updater.apply_on_start()  # 새 창: 앞 창(여기서는 자기 자신)이 닫히기를 기다렸다가 적용
    assert repos.head() == second and time.monotonic() - started < 10


def test_ci_result_reads_the_tests_workflow_on_main(monkeypatch):
    sha = "a" * 40
    runs = [
        {"name": "emr-probe", "head_sha": sha, "head_branch": "main", "event": "push", "status": "completed", "conclusion": "failure"},
        {"name": "tests", "head_sha": sha, "head_branch": "feat/x", "event": "push", "status": "completed", "conclusion": "cancelled"},
        {"name": "tests", "head_sha": sha, "head_branch": "main", "event": "push", "status": "in_progress", "conclusion": None},
    ]
    monkeypatch.setattr(updater, "_api_json", lambda url: {"workflow_runs": runs})
    assert updater.ci_result(("o", "r"), sha) == "pending"
    runs[2].update(status="completed", conclusion="success")
    assert updater.ci_result(("o", "r"), sha) == "success"
    runs[2].update(conclusion="failure")
    assert updater.ci_result(("o", "r"), sha) == "failure"
    monkeypatch.setattr(updater, "_api_json", lambda url: {"workflow_runs": []})
    assert updater.ci_result(("o", "r"), sha) == "pending"  # 아직 테스트가 시작되지 않음
    monkeypatch.setattr(updater, "_api_json", lambda url: None)
    assert updater.ci_result(("o", "r"), sha) == "error"


def test_github_repo_and_restart_command(monkeypatch):
    for url in ("https://github.com/jwhwang91/clinic-scribe.git", "git@github.com:jwhwang91/clinic-scribe.git",
                "https://github.com/jwhwang91/clinic-scribe"):
        monkeypatch.setattr(updater, "_git", lambda *a, url=url, **k: (0, url))
        assert updater.github_repo() == ("jwhwang91", "clinic-scribe")
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "/srv/git/clinic-scribe.git"))
    assert updater.github_repo() is None
    monkeypatch.setenv("__CFBundleIdentifier", updater.APP_BUNDLE_ID)
    if sys.platform == "darwin":  # 앱 아이콘으로 열었으면 같은 앱으로 다시 연다 (마이크 권한이 그 앱에 있다)
        assert updater.restart_command() == ["open", "-n", "-b", updater.APP_BUNDLE_ID]
    monkeypatch.delenv("__CFBundleIdentifier")
    assert updater.restart_command()[-2:] == ["-m", "clinic_scribe"]


# ---------------------------------------------------------------------------- 화면
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("PySide6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


@pytest.mark.gui
def test_update_dialog_shows_changes_and_restarts_only_when_idle(qapp, repos):
    from clinic_scribe.ui.updates import UpdateDialog, UpdateWatcher

    repos.push("끼어든 말 나누기", app="2")
    repos.push("씨차트 연결 점검", app="3")
    watcher = UpdateWatcher(config.Settings())
    assert watcher.chip() is None
    watcher._on_done(updater.check())
    assert watcher.chip() == ("새 버전", "busy")
    busy = [True]
    dialog = UpdateDialog(watcher, lambda: not busy[0])
    text = dialog.changes.toPlainText()
    assert "씨차트 연결 점검" in text and "끼어든 말 나누기" in text and dialog.pill.text() == "새 버전 준비됨"
    assert not dialog.restart_btn.isEnabled()  # 녹음·처리 중
    busy[0] = False
    watcher.changed.emit()
    asked = []
    dialog.restart_requested.connect(lambda: asked.append(1))
    dialog.restart_btn.click()
    assert asked == [1]
    with open(updater.state_path(), encoding="utf-8") as f:
        assert json.load(f)["ready"]
