"""scripts/make_launcher.sh: 클릭해서 여는 clinic-scribe.app 만들기 (실제로 컴파일·서명하고 실행해 본다)."""

from __future__ import annotations

import json
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from clinic_scribe import mic_permission

REPO_ROOT = Path(__file__).resolve().parent.parent
MAKE = REPO_ROOT / "scripts" / "make_launcher.sh"

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("xcrun") is None or not (REPO_ROOT / ".venv" / "bin" / "python").exists(),
    reason="macOS 명령행 도구와 repo 가상환경이 필요합니다",
)


def build(script: Path, app: Path) -> None:
    result = subprocess.run([str(script), str(app)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_builds_signed_app_that_starts_clinic_scribe(tmp_path):
    app = tmp_path / "clinic-scribe.app"
    build(MAKE, app)
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == mic_permission.APP_BUNDLE_ID
    assert info["NSMicrophoneUsageDescription"]  # 없으면 macOS 가 마이크 권한을 묻지 못한다
    assert (app / "Contents" / "Resources" / "clinic-scribe.icns").stat().st_size > 0
    assert subprocess.run(["codesign", "--verify", "--strict", str(app)], capture_output=True).returncode == 0

    exe = app / "Contents" / "MacOS" / "clinic-scribe"
    ok = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=120)
    assert ok.returncode == 0, ok.stderr
    assert "usage: clinic-scribe" in ok.stdout  # 가상환경의 PySide6 를 불러와 앱 진입점까지 돌았다
    bad = subprocess.run([str(exe), "--open", str(tmp_path / "없음.wav")], capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2 and "파일이 없습니다" in bad.stderr  # 앱의 종료 코드가 그대로 전해진다


def test_app_runs_module_with_venv_python(tmp_path):
    """가짜 repo 로 만들어 python 설정을 확인: sys.executable·site-packages·인자·현재 폴더·PATH."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("make_launcher.sh", "launcher.c"):
        shutil.copy2(REPO_ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / ".venv").symlink_to(REPO_ROOT / ".venv")
    (repo / "clinic_scribe").mkdir()
    (repo / "clinic_scribe" / "__main__.py").write_text(
        "import json, os, sys\nimport PySide6\n"
        "print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix, 'argv': sys.argv[1:],"
        " 'cwd': os.getcwd(), 'path': os.environ['PATH'], 'pyside': PySide6.__file__}))\n",
        encoding="utf-8",
    )
    app = tmp_path / "probe.app"
    build(repo / "scripts" / "make_launcher.sh", app)

    finder_env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}  # Finder 에서 열 때의 PATH
    result = subprocess.run([str(app / "Contents" / "MacOS" / "clinic-scribe"), "--x", "-psn_0_42", "값"],
                            capture_output=True, text=True, timeout=120, env=finder_env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["executable"] == str(repo / ".venv" / "bin" / "python")  # 하위 프로세스도 가상환경 python 으로
    assert Path(out["prefix"]).resolve() == (REPO_ROOT / ".venv").resolve()
    assert Path(out["pyside"]).resolve().is_relative_to((REPO_ROOT / ".venv").resolve())
    assert out["argv"] == ["--x", "값"]  # 옛 macOS 의 -psn_ 인자는 버린다
    assert Path(out["cwd"]).resolve() == repo.resolve()
    assert out["path"] == f"{Path.home()}/.local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def test_refuses_path_that_is_not_an_app(tmp_path):
    result = subprocess.run([str(MAKE), str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and ".app" in result.stderr
    assert tmp_path.is_dir()  # 엉뚱한 폴더를 지우지 않는다
