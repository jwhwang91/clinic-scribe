"""'설치' 창 (오프스크린): 새 컴퓨터에서 빠진 것 찾기, 요약 방식 고르기, 사양에 맞는 로컬 모델, 받기·취소·실패, Claude 로그인.

실제 받기는 가짜로 바꾼다 (인터넷에서 아무것도 받지 않고, Claude Code 도 설치하지 않는다).
"""

import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from clinic_scribe import installer
from clinic_scribe.config import Settings
from clinic_scribe.ui import setup_dialog as sd

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def wait_until(qapp, condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def new_pc(monkeypatch, tmp_path):
    """whisper·모델·Claude 가 하나도 없는 메모리 16GB Windows PC (CPU 만)."""
    monkeypatch.setattr(installer, "system_specs", lambda: installer.Specs("win32", "x64", "Intel Core i5", 12, 16.0, "", False))
    monkeypatch.setenv("CLINIC_SCRIBE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(installer, "claude_logged_in", lambda cli: False)
    settings = Settings()
    settings.whisper_cli = str(tmp_path / "없는-whisper-cli")
    state = {"claude": None}
    settings.claude_cli_path = lambda: state["claude"]  # 설치하면 생긴다
    settings.state = state
    return settings


def test_setup_is_needed_on_a_new_computer_and_not_when_ready(new_pc, monkeypatch, tmp_path):
    assert sd.SetupDialog.needed(new_pc)
    monkeypatch.setattr(installer, "whisper_ready", lambda cli: True)
    monkeypatch.setattr(installer, "is_valid", lambda item, path, check_hash=False: True)
    new_pc.state["claude"] = str(tmp_path / "claude")
    assert sd.SetupDialog.needed(new_pc)  # 요약 방식을 아직 고르지 않았다
    new_pc.summary_engine = "claude"
    assert not sd.SetupDialog.needed(new_pc)
    new_pc.summary_engine = "local"  # 로컬로 바꿨는데 llama-server 가 없다
    assert sd.SetupDialog.needed(new_pc)


def test_models_show_what_this_computer_can_run(qapp, new_pc):
    dialog = sd.SetupDialog(new_pc)
    assert dialog.local_radio.isChecked()  # Claude 가 없으면 로컬을 먼저 보여 준다
    assert dialog.selected_model().id == "qwen3-4b"  # 16GB CPU: 원활한 것 중 가장 큰 모델
    texts = {dialog.models.item(r, 0).data(Qt.ItemDataRole.UserRole): dialog.models.item(r, 2).text()
             for r in range(dialog.models.rowCount())}
    assert texts["qwen3-4b"].startswith("✓ 원활") and texts["qwen3-8b"].startswith("△ 느림")
    assert texts["qwen3-30b-a3b"].startswith("✗ 메모리 부족")
    row_30b = [r for r in range(dialog.models.rowCount()) if dialog.models.item(r, 0).data(Qt.ItemDataRole.UserRole) == "qwen3-30b-a3b"][0]
    assert dialog.models.item(row_30b, 0).flags() == Qt.ItemFlag.NoItemFlags  # 메모리가 모자라는 모델은 고를 수 없다
    assert "추천" in dialog.models.item(0, 0).text()
    assert dialog._plan() == ["whisper", "whisper_model", "vad_model", "speaker_model", "llama", "local_model"]
    assert dialog.install_btn.text() == "설치하고 시작"
    dialog.choose("claude")
    assert dialog._plan()[-1] == "claude" and not dialog.models.isVisibleTo(dialog)
    dialog.reject()


def test_install_local_model_after_ok(qapp, new_pc, monkeypatch, tmp_path):
    calls = []

    def fake_whisper(progress=None, cancel=None):
        calls.append("whisper")
        progress(5, 10)
        progress(10, 10)

    def fake_fetch(item, folder, progress=None, cancel=None):
        calls.append(item.name)
        progress(item.size, item.size)
        return folder / item.name

    monkeypatch.setattr(installer, "install_whisper", fake_whisper)
    monkeypatch.setattr(installer, "fetch", fake_fetch)
    monkeypatch.setattr(installer, "install_llama", lambda progress=None, cancel=None: calls.append("llama"))
    dialog = sd.SetupDialog(new_pc)
    dialog.choose("local", "qwen3-4b")
    assert not calls  # '설치하고 시작' 을 누르기 전에는 아무것도 받지 않는다
    seen = []
    dialog._signals.progress.connect(lambda done, total: seen.append(total))
    dialog.install_btn.click()
    assert wait_until(qapp, lambda: dialog.result() == QDialog.DialogCode.Accepted)
    assert calls == ["whisper", "ggml-large-v3-turbo-q5_0.bin", "ggml-silero-v6.2.0.bin",
                     "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", "llama", "Qwen3-4B-Q4_K_M.gguf"]
    assert new_pc.summary_engine == "local" and new_pc.local_model == "qwen3-4b"
    assert 2497280256 in seen  # 2GB 가 넘는 모델 크기도 진행 표시에 그대로 전해진다


def test_cancel_and_failure_keep_the_window_open(qapp, new_pc, monkeypatch):
    def slow_whisper(progress=None, cancel=None):
        while not cancel.is_set():
            time.sleep(0.02)
        raise installer.InstallCancelled("취소")

    monkeypatch.setattr(installer, "install_whisper", slow_whisper)
    dialog = sd.SetupDialog(new_pc)
    dialog.install_btn.click()
    assert wait_until(qapp, lambda: dialog.cancel_btn.isVisibleTo(dialog))
    dialog.cancel_btn.click()
    assert wait_until(qapp, lambda: "취소" in dialog.status.text() and dialog.install_btn.isEnabled())
    assert dialog.result() != QDialog.DialogCode.Accepted and new_pc.summary_engine == ""

    def broken(progress=None, cancel=None):
        raise installer.InstallError("이 OS·CPU 용 whisper.cpp 공식 빌드가 없습니다.")

    monkeypatch.setattr(installer, "install_whisper", broken)
    dialog.install_btn.click()
    assert wait_until(qapp, lambda: "공식 빌드가 없습니다" in dialog.status.text())
    assert dialog.result() != QDialog.DialogCode.Accepted and dialog.install_btn.isEnabled()
    dialog.reject()


def test_claude_is_installed_then_the_user_logs_in(qapp, new_pc, monkeypatch, tmp_path):
    monkeypatch.setattr(installer, "whisper_ready", lambda cli: True)  # 전사에 필요한 것은 있고 Claude 만 없다
    monkeypatch.setattr(installer, "is_valid", lambda item, path, check_hash=False: True)
    fake_cli = str(tmp_path / ".local" / "bin" / "claude")

    def fake_install(cancel=None):
        new_pc.state["claude"] = fake_cli
        return fake_cli

    monkeypatch.setattr(installer, "install_claude", fake_install)
    opened = []
    monkeypatch.setattr(installer, "open_claude_login", lambda cli: opened.append(cli) or True)
    dialog = sd.SetupDialog(new_pc)
    dialog.choose("claude")
    assert "설치 필요" in dialog.claude_state.text() and dialog._plan() == ["claude"]
    dialog.install_btn.click()
    assert wait_until(qapp, lambda: "로그인 창 열기" in dialog.status.text())  # 설치는 됐고 로그인은 사용자가 한다
    assert dialog.result() != QDialog.DialogCode.Accepted and new_pc.summary_engine == "claude"
    assert dialog.login_btn.isVisibleTo(dialog)
    dialog.login_btn.click()
    assert opened == [fake_cli]
    monkeypatch.setattr(installer, "claude_logged_in", lambda cli: True)  # 브라우저에서 로그인했다
    dialog.recheck_btn.click()
    assert wait_until(qapp, lambda: "로그인됨" in dialog.claude_state.text())
    assert dialog.install_btn.text() == "시작"
    dialog.install_btn.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
