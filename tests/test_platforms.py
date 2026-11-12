"""다른 OS·하드웨어에서도 같게 동작하는지: OS 별로 갈리는 부분을 이 컴퓨터에서 OS 를 바꿔 가며 확인한다.

(실제 Windows·Linux 에서의 실행은 .github/workflows/tests.yml 이 GitHub 의 Windows·Linux·macOS 컴퓨터에서 확인한다.)
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from clinic_scribe import config, mic_permission, procutil, recorder, whisper


@pytest.fixture
def on(monkeypatch):
    """OS 를 바꿔 본다: on("win32") / on("linux") / on("darwin")."""

    def switch(name: str) -> None:
        monkeypatch.setattr(sys, "platform", name)

    return switch


def test_user_folders_follow_each_os(on, monkeypatch, tmp_path):
    monkeypatch.delenv("CLINIC_SCRIBE_SETTINGS", raising=False)
    monkeypatch.delenv("CLINIC_SCRIBE_MODEL_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    on("win32")
    assert config.settings_path() == tmp_path / "Local" / "clinic-scribe" / "settings.json"
    assert config.default_model_dir() == tmp_path / "Local" / "clinic-scribe" / "models"
    assert config.records_dir() == tmp_path / "Local" / "clinic-scribe" / "records"
    assert config.whisper_dir() == tmp_path / "Local" / "clinic-scribe" / "whisper.cpp" / "bin"
    assert config.exe_name("whisper-cli") == "whisper-cli.exe" and config.setup_command() == r"scripts\setup_windows.bat"
    on("linux")
    assert config.settings_path() == tmp_path / "share" / "clinic-scribe" / "settings.json"
    assert config.default_model_dir() == tmp_path / "cache" / "clinic-scribe" / "models"
    assert config.setup_command() == "./scripts/setup_linux.sh" and config.run_command() == "./scripts/run.sh"
    on("darwin")  # 이 Mac 의 기존 위치는 그대로
    assert config.settings_path() == Path.home() / "Library" / "Application Support" / "clinic-scribe" / "settings.json"
    assert config.default_model_dir() == tmp_path / "cache" / "clinic-scribe" / "models"


def test_tool_candidates_per_os(on, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    on("win32")
    assert config.whisper_cli_candidates() == [str(tmp_path / "clinic-scribe" / "whisper.cpp" / "bin" / "whisper-cli.exe")]
    assert config.claude_cli_candidates()[-1] == str(tmp_path / "Roaming" / "npm" / "claude.cmd")
    on("linux")
    assert "/usr/bin/whisper-cli" in config.whisper_cli_candidates()
    on("darwin")
    assert config.whisper_cli_candidates()[0] == "/opt/homebrew/bin/whisper-cli"


def test_vad_tool_found_next_to_cli_with_exe_suffix(on, tmp_path):
    for name in ("whisper-cli.exe", "whisper-vad-speech-segments.exe"):
        (tmp_path / name).write_text("")
        (tmp_path / name).chmod(0o755)
    on("win32")
    assert whisper.find_vad_tool(str(tmp_path / "whisper-cli.exe")) == str(tmp_path / "whisper-vad-speech-segments.exe")


def test_windows_paths_with_korean_become_short_paths(on, monkeypatch, tmp_path):
    """Windows whisper.cpp 는 명령줄 인자를 시스템 코드 페이지로 받아 한글 경로를 못 연다 → 8.3 짧은 경로로 넘긴다."""
    folder = tmp_path / "홍길동"
    folder.mkdir()
    audio = folder / "in.wav"
    audio.write_bytes(b"x")
    short = {str(audio): "C:\\Users\\B1FD~1\\in.wav", str(folder): "C:\\Users\\B1FD~1"}
    monkeypatch.setattr(config, "_short_path", short.get)
    on("win32")
    assert config.native_path(audio) == "C:\\Users\\B1FD~1\\in.wav"
    assert config.native_path(folder / "out").endswith("out") and config.native_path(folder / "out").isascii()
    assert config.native_path(tmp_path / "plain.wav") == str(tmp_path / "plain.wav")  # ASCII 는 그대로
    monkeypatch.setattr(config, "_short_path", lambda path: None)  # 8.3 이름이 없는 드라이브: 원래 경로
    assert config.native_path(audio) == str(audio)
    on("darwin")
    assert config.native_path(audio) == str(audio)


def test_windows_prompt_keeps_ascii_words_only(on, monkeypatch):
    """Windows 가 UTF-8 모드가 아니면 한글 힌트가 깨지므로 영어 약명·검사명만 넘긴다 (교정은 전사 뒤에 그대로)."""
    prompt = "트리메부틴, 알마게이트 현탁액, Almagate. CBC, CT, MRI, omeprazole."
    monkeypatch.setattr(whisper, "_windows_utf8", lambda: False)
    on("win32")
    assert whisper.cli_prompt(prompt) == "Almagate. CBC, CT, MRI, omeprazole."
    assert whisper.cli_prompt("세티리진, 레바미피드.") == ""
    monkeypatch.setattr(whisper, "_windows_utf8", lambda: True)  # '세계 언어 지원 UTF-8' 이 켜져 있으면 그대로
    assert whisper.cli_prompt(prompt) == prompt
    on("darwin")
    assert whisper.cli_prompt(prompt) == prompt


def test_child_processes_have_no_console_window_on_windows_and_are_killed_as_a_tree(on, monkeypatch):
    on("win32")
    flags = procutil.popen_group_options()["creationflags"]
    assert flags & 0x08000000 and flags & 0x200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    calls = []
    monkeypatch.setattr(procutil.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    procutil._kill_group(types.SimpleNamespace(pid=4321), force=False)
    assert calls == [["taskkill", "/PID", "4321", "/T", "/F"]]
    on("linux")
    assert procutil.popen_group_options() == {"start_new_session": True}


@pytest.mark.skipif(sys.platform == "win32", reason="Windows 는 taskkill /T (위에서 확인)")
def test_real_process_is_cancelled_with_its_children(tmp_path):
    """이 컴퓨터에서 실제로: 취소하면 자식 프로세스까지 끝난다."""
    import threading

    script = tmp_path / "parent.py"
    marker = tmp_path / "child.pid"
    script.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    with pytest.raises(procutil.Cancelled):
        procutil.run([sys.executable, str(script)], cancel=cancel, timeout=30)
    import os
    import time

    pid = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("자식 프로세스가 남아 있다")


WINDOWS_DEVICES = [  # 같은 USB 마이크가 MME·DirectSound·WASAPI 로 겹쳐 나오고, MME 는 이름을 31자로 자른다
    {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100},
    {"name": "마이크(Realtek(R) Audio)", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100},
    {"name": "Microphone (USB Conference Spea", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 44100},
    {"name": "스피커(Realtek(R) Audio)", "hostapi": 0, "max_input_channels": 0, "default_samplerate": 44100},
    {"name": "마이크(Realtek(R) Audio)", "hostapi": 1, "max_input_channels": 2, "default_samplerate": 44100},
    {"name": "Microphone (USB Conference Speakerphone)", "hostapi": 2, "max_input_channels": 1, "default_samplerate": 48000},
    {"name": "마이크(Realtek(R) Audio)", "hostapi": 2, "max_input_channels": 2, "default_samplerate": 48000},
]
WINDOWS_APIS = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]


def test_windows_lists_each_microphone_once_with_full_name():
    devices = recorder.select_input_devices(WINDOWS_DEVICES, WINDOWS_APIS, default_index=1, default_api=0, platform="win32")
    assert [d.name for d in devices] == [
        "Microsoft Sound Mapper - Input", "마이크(Realtek(R) Audio)", "Microphone (USB Conference Speakerphone)",
    ]
    assert [d.index for d in devices] == [0, 1, 2]  # 여는 것은 MME 장치 (어떤 샘플레이트든 Windows 가 맞춰 준다)
    assert devices[1].is_default and devices[2].default_samplerate == 44100


def test_linux_hides_alsa_virtual_devices_but_keeps_hardware_and_default():
    devices = [
        {"name": "HDA Intel PCH: ALC892 Analog (hw:0,0)", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100},
        {"name": "USB PnP Sound Device: Audio (hw:1,0)", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000},
        {"name": "sysdefault", "hostapi": 0, "max_input_channels": 128, "default_samplerate": 48000},
        {"name": "dsnoop", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 48000},
        {"name": "pipewire", "hostapi": 0, "max_input_channels": 64, "default_samplerate": 48000},
        {"name": "default", "hostapi": 0, "max_input_channels": 64, "default_samplerate": 48000},
        {"name": "system", "hostapi": 1, "max_input_channels": 2, "default_samplerate": 48000},  # JACK
    ]
    picked = recorder.select_input_devices(devices, [{"name": "ALSA"}, {"name": "JACK Audio Connection Kit"}],
                                           default_index=5, default_api=0, platform="linux")
    assert [d.name for d in picked] == [
        "HDA Intel PCH: ALC892 Analog (hw:0,0)", "USB PnP Sound Device: Audio (hw:1,0)", "pipewire", "default",
    ]
    assert picked[-1].is_default


def test_macos_device_list_is_unchanged():
    devices = [{"name": "MacBook Pro 마이크", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000},
               {"name": "‘iPhone (2)’ 마이크", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000}]
    picked = recorder.select_input_devices(devices, [{"name": "Core Audio"}], 0, 0, platform="darwin")
    assert [d.name for d in picked] == ["MacBook Pro 마이크", "‘iPhone (2)’ 마이크"]


def fake_winreg(values: dict) -> types.ModuleType:
    module = types.ModuleType("winreg")
    module.HKEY_LOCAL_MACHINE, module.HKEY_CURRENT_USER = "HKLM", "HKCU"

    class Key:
        def __init__(self, root, path):
            self.where = (root, path)

        def __enter__(self):
            if not any(k[:2] == self.where for k in values):
                raise OSError("키 없음")
            return self

        def __exit__(self, *exc):
            return False

    module.OpenKey = Key

    def query(key, name):
        try:
            return values[(*key.where, name)], 1
        except KeyError:
            raise OSError("값 없음") from None

    module.QueryValueEx = query
    return module


def test_windows_microphone_privacy_settings(on, monkeypatch):
    consent = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
    on("win32")
    cases = [
        ({("HKLM", consent, "Value"): "Allow", ("HKCU", consent, "Value"): "Allow",
          ("HKCU", consent + r"\NonPackaged", "Value"): "Allow"}, "authorized"),
        ({("HKCU", consent, "Value"): "Allow", ("HKCU", consent + r"\NonPackaged", "Value"): "Deny"}, "denied"),
        ({("HKLM", r"SOFTWARE\Policies\Microsoft\Windows\AppPrivacy", "LetAppsAccessMicrophone"): 2}, "restricted"),
        ({}, "unknown"),
    ]
    for values, expected in cases:
        monkeypatch.setitem(sys.modules, "winreg", fake_winreg(values))
        assert mic_permission.authorization_status() == expected
    assert not mic_permission.request_access(lambda granted: None)  # Windows 는 묻는 대화상자가 없다
    assert "데스크톱 앱이 마이크에 액세스하도록 허용" in mic_permission.denied_message("denied")
    assert "관리 정책" in mic_permission.denied_message("restricted")
    assert "다른 프로그램이 마이크를 독점" in mic_permission.silent_input_message("마이크(Realtek(R) Audio)")
    assert not mic_permission.lid_closed()


def test_linux_has_no_per_app_permission(on):
    on("linux")
    assert mic_permission.authorization_status() == "not_applicable"
    assert mic_permission.STATUS_LABEL["not_applicable"] == "해당 없음"
    assert "PipeWire" in mic_permission.silent_input_message("default")
    assert "Linux" in mic_permission.permission_tooltip()
    assert not mic_permission.lid_closed() and not mic_permission.request_access(lambda granted: None)


def test_shortcut_launch_without_console_has_safe_stdio(monkeypatch):
    """Windows 바로 가기(pythonw)는 표준 출력이 없다: 무엇을 출력해도 앱이 멈추지 않게 한다."""
    from clinic_scribe import app

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    app._ensure_stdio()
    print("출력해도 괜찮다", file=sys.stderr)
    assert sys.stdout is not None and sys.stderr is not None


def test_dark_palette_for_windows_and_linux():
    from PySide6.QtGui import QPalette

    from clinic_scribe.ui import theme

    palette = theme.dark_palette()
    assert palette.color(QPalette.ColorRole.Window).name() == "#0f1117"
    assert palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).name() == "#4b5263"


def test_setup_scripts_for_each_os():
    repo = Path(__file__).resolve().parent.parent
    for name in ("setup_windows.bat", "run.bat"):
        data = (repo / "scripts" / name).read_bytes()
        assert data.isascii()  # cmd.exe 는 배치 파일을 시스템 코드 페이지로 읽는다: 한글은 setup.py 가 출력한다
        assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")  # CRLF
    if sys.platform != "win32":  # Windows 의 bash 는 WSL 일 수 있다
        for name in ("setup_macos.sh", "setup_linux.sh"):
            assert subprocess.run(["bash", "-n", str(repo / "scripts" / name)]).returncode == 0
    assert "eol=crlf" in (repo / ".gitattributes").read_text(encoding="utf-8")


def test_windows_npm_claude_runs_through_node_not_cmd(on, monkeypatch, tmp_path):
    """npm 의 claude.cmd 는 cmd.exe 가 인자(따옴표·꺾쇠·한글)를 다시 해석해 깨뜨린다 → cli.js 를 node 로 바로 실행한다."""
    npm = tmp_path / "npm"
    script = npm / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js"
    script.parent.mkdir(parents=True)
    script.write_text("")
    (npm / "node.exe").write_text("")
    (npm / "claude.cmd").write_text("")
    on("win32")
    assert config.claude_command(str(npm / "claude.cmd")) == [str(npm / "node.exe"), str(script)]
    assert config.claude_command(str(tmp_path / "claude.exe")) == [str(tmp_path / "claude.exe")]  # 공식 설치본은 그대로
    from clinic_scribe.summarizer import ClaudeConfig, build_command

    assert build_command(ClaudeConfig(cli=str(npm / "claude.cmd"), model="m"))[:3] == [str(npm / "node.exe"), str(script), "-p"]
    native = tmp_path / ".local" / "bin" / "claude.exe"  # 공식 설치본이 있으면 claude.cmd 보다 먼저
    native.parent.mkdir(parents=True)
    native.write_text("")
    monkeypatch.setattr(config, "claude_cli_candidates", lambda: [str(native), str(npm / "claude.cmd")])
    monkeypatch.setattr(config.shutil, "which", lambda name: str(npm / "claude.cmd"))
    assert config.Settings().claude_cli_path() == str(native)
    on("darwin")
    assert config.claude_command(str(npm / "claude.cmd")) == [str(npm / "claude.cmd")]
