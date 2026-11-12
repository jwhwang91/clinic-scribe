"""필요한 프로그램·모델 받기 (installer.py): 받는 목록, 컴퓨터 사양에 맞는 모델 판단, 받기·검증·풀기.

받기는 이 컴퓨터 안의 임시 HTTP 서버로 확인한다 (인터넷에서 아무것도 받지 않는다).
"""

import hashlib
import http.server
import importlib.util
import io
import sys
import tarfile
import threading
import zipfile
from functools import partial
from pathlib import Path

import pytest

from clinic_scribe import config, installer
from clinic_scribe.installer import Download, InstallCancelled, InstallError, LocalModel, Specs

REPO = Path(__file__).resolve().parent.parent


def test_everything_downloaded_is_pinned():
    builds = [*installer.WHISPER_BUILDS.values(), *installer.LLAMA_BUILDS.values()]
    items = [*installer.MODELS, *builds, *(m.file for m in installer.LOCAL_MODELS)]
    for item in items:
        assert item.url.startswith("https://") and item.size > 0, item
        assert len(item.sha256) == 64 and int(item.sha256, 16) >= 0, item
    for key in (("win32", "x64"), ("win32", "arm64"), ("linux", "x64"), ("linux", "arm64")):
        assert key in installer.WHISPER_BUILDS and key in installer.LLAMA_BUILDS
    assert ("darwin", "arm64") in installer.LLAMA_BUILDS  # macOS 의 whisper.cpp 는 Homebrew
    ids = [m.id for m in installer.LOCAL_MODELS]
    assert len(set(ids)) == len(ids) and installer.DEFAULT_LOCAL_MODEL in ids
    assert config.Settings().local_model == installer.DEFAULT_LOCAL_MODEL


def test_setup_script_uses_the_same_catalog():
    spec = importlib.util.spec_from_file_location("setup_script", REPO / "scripts" / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.MODELS is installer.MODELS and module.WHISPER_BUILDS is installer.WHISPER_BUILDS


def specs(ram: float, metal: bool = False) -> Specs:
    return Specs("darwin" if metal else "win32", "arm64" if metal else "x64", "CPU", 8, ram, "Apple Metal" if metal else "", metal)


def model(model_id: str) -> LocalModel:
    return installer.local_model(model_id)


def test_model_fit_depends_on_memory_and_gpu():
    office_pc = specs(8)  # 메모리 8GB 사무용 PC
    assert installer.model_fit(model("qwen3-4b"), office_pc).level == "no"  # 5GB 필요 > 8-4
    assert all(installer.model_fit(m, office_pc).level == "no" for m in installer.LOCAL_MODELS)
    pc16 = specs(16)
    assert installer.model_fit(model("qwen3-4b"), pc16).level == "good"
    assert installer.model_fit(model("qwen3-8b"), pc16).level == "slow" and "CPU" in installer.model_fit(model("qwen3-8b"), pc16).text
    assert installer.model_fit(model("qwen3-30b-a3b"), pc16).level == "no"
    assert "필요 약 21GB" in installer.model_fit(model("qwen3-30b-a3b"), pc16).text
    assert installer.recommended_model(pc16).id == "qwen3-4b"
    pc64 = specs(64)
    assert installer.model_fit(model("qwen3-30b-a3b"), pc64).level == "good"  # 전문가 혼합: 크지만 빠르다
    assert installer.recommended_model(pc64).id == "qwen3-30b-a3b"
    mac24 = specs(24, metal=True)
    assert installer.model_fit(model("qwen3-14b"), mac24).level == "good"
    assert installer.recommended_model(mac24).id == "qwen3-14b"
    assert installer.recommended_model(office_pc).id == "qwen3-4b"  # 돌아가는 것이 없으면 가장 가벼운 것


def test_this_computer_specs_are_read():
    found = installer.system_specs()
    assert found.ram_gb > 1 and found.threads >= 1 and found.cpu
    assert ("메모리" in found.describe()) and (found.metal == (sys.platform == "darwin" and found.machine == "arm64"))


@pytest.fixture
def server(tmp_path):
    """tmp_path/www 를 내보내는 임시 HTTP 서버. (주소, 폴더)."""
    www = tmp_path / "www"
    www.mkdir()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(www))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/", www
    httpd.shutdown()


def publish(www: Path, name: str, data: bytes, base: str) -> Download:
    (www / name).write_bytes(data)
    return Download(name, base + name, hashlib.sha256(data).hexdigest(), len(data))


def test_fetch_verifies_reuses_and_reports_progress(server, tmp_path):
    base, www = server
    item = publish(www, "model.bin", b"x" * 3_000_000, base)
    seen = []
    path = installer.fetch(item, tmp_path / "models", lambda done, total: seen.append((done, total)))
    assert path.read_bytes() == b"x" * 3_000_000 and not (tmp_path / "models" / "model.bin.part").exists()
    assert seen[-1] == (3_000_000, 3_000_000) and len(seen) >= 3
    (www / "model.bin").unlink()  # 이미 받은 것은 다시 받지 않는다 (SHA-256 확인만)
    assert installer.fetch(item, tmp_path / "models") == path
    wrong = Download("other.bin", base + "other.bin", "0" * 64, 4)
    (www / "other.bin").write_bytes(b"abcd")
    with pytest.raises(InstallError, match="SHA-256"):
        installer.fetch(wrong, tmp_path / "models")
    assert not (tmp_path / "models" / "other.bin").exists() and not (tmp_path / "models" / "other.bin.part").exists()


def test_fetch_can_be_cancelled_and_keeps_the_partial_file(server, tmp_path):
    base, www = server
    item = publish(www, "big.bin", b"y" * 5_000_000, base)
    cancel = threading.Event()

    def progress(done, total):
        if done >= 1_000_000:
            cancel.set()

    with pytest.raises(InstallCancelled):
        installer.fetch(item, tmp_path, progress, cancel)
    assert 0 < (tmp_path / "big.bin.part").stat().st_size < 5_000_000  # 다음에 이어받는다
    assert installer.fetch(item, tmp_path).read_bytes() == b"y" * 5_000_000


def zip_bytes(files: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buffer.getvalue()


def tar_bytes(files: dict, links: dict | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            t.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            t.addfile(info)
    return buffer.getvalue()


@pytest.mark.skipif(sys.platform == "win32", reason="실행 권한 비트·심볼릭 링크는 POSIX")
def test_unpack_official_layouts_for_windows_and_linux(monkeypatch, tmp_path):
    windows = tmp_path / "whisper-bin-x64.zip"
    windows.write_bytes(zip_bytes({"Release/whisper-cli.exe": b"exe", "Release/whisper-vad-speech-segments.exe": b"exe",
                                   "Release/ggml-cpu-haswell.dll": b"dll", "Release/whisper.dll": b"dll"}))
    monkeypatch.setattr(sys, "platform", "win32")
    cli = installer.unpack(windows, tmp_path / "win", "whisper-cli", ("whisper-vad-speech-segments",))
    assert cli == tmp_path / "win" / "whisper-cli.exe" and (tmp_path / "win" / "ggml-cpu-haswell.dll").is_file()

    linux = tmp_path / "whisper-bin-ubuntu-x64.tar.gz"
    linux.write_bytes(tar_bytes({"whisper-bin-ubuntu-x64/whisper-cli": b"elf", "whisper-bin-ubuntu-x64/whisper-vad-speech-segments": b"elf",
                                 "whisper-bin-ubuntu-x64/libwhisper.so.1.9.5": b"so"},
                                {"whisper-bin-ubuntu-x64/libwhisper.so.1": "libwhisper.so.1.9.5"}))
    monkeypatch.setattr(sys, "platform", "linux")
    cli = installer.unpack(linux, tmp_path / "lin", "whisper-cli", ("whisper-vad-speech-segments",))
    assert cli.stat().st_mode & 0o111 and (tmp_path / "lin" / "libwhisper.so.1").is_symlink()  # $ORIGIN 라이브러리 이름 그대로

    evil = tmp_path / "evil.zip"
    evil.write_bytes(zip_bytes({"../escape.exe": b"x", "Release/whisper-cli.exe": b"x"}))
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(InstallError, match="경로"):
        installer.unpack(evil, tmp_path / "evil", "whisper-cli")
    assert not (tmp_path / "escape.exe").exists()
    missing = tmp_path / "missing.zip"
    missing.write_bytes(zip_bytes({"Release/whisper-cli.exe": b"x"}))
    with pytest.raises(InstallError, match="whisper-vad-speech-segments"):
        installer.unpack(missing, tmp_path / "m", "whisper-cli", ("whisper-vad-speech-segments",))


def test_install_build_reuses_the_same_build(monkeypatch, server, tmp_path):
    base, www = server
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    archive = tar_bytes({f"llama/{config.exe_name('llama-server')}": b"bin", "llama/libllama.so": b"so"})
    build = publish(www, "llama-test.tar.gz", archive, base)
    target = tmp_path / "llama"
    exe = installer.install_build(build, target, "llama-server")
    assert exe.is_file() and (target / ".build").read_text() == build.name
    assert not (config.cache_dir() / "downloads" / build.name).exists()  # 풀고 나면 압축 파일은 지운다
    (www / build.name).unlink()
    assert installer.install_build(build, target, "llama-server") == exe  # 다시 받지 않는다


@pytest.mark.skipif(sys.platform == "win32", reason="shebang 가짜 실행 파일")
def test_claude_login_state_is_read_without_sending_anything(tmp_path):
    for state, expected in (("true", True), ("false", False), ("", None)):
        cli = tmp_path / f"claude-{state or 'broken'}"
        body = f'print(\'{{"loggedIn": {state}}}\')' if state else "print('not json')"
        cli.write_text(f"#!{sys.executable}\nimport sys\nassert sys.argv[1:] == ['auth', 'status', '--json']\n{body}\n")
        cli.chmod(0o755)
        assert installer.claude_logged_in(str(cli)) is expected
    assert installer.claude_logged_in(str(tmp_path / "없음")) is None


def test_validity_check_by_size_then_hash(tmp_path):
    data = b"model"
    item = Download("m.bin", "https://example.invalid/m.bin", hashlib.sha256(data).hexdigest(), len(data))
    path = tmp_path / "m.bin"
    assert not installer.is_valid(item, path)
    path.write_bytes(b"MODEL")  # 크기만 같고 내용이 다르다
    assert installer.is_valid(item, path) and not installer.is_valid(item, path, check_hash=True)
    path.write_bytes(data)
    assert installer.is_valid(item, path, check_hash=True)
    assert installer.human_size(574041195) == "574MB" and installer.human_size(18556685824) == "18.6GB"
