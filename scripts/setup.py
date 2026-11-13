#!/usr/bin/env python3
"""clinic-scribe 설치 (macOS · Windows · Linux 공통). 표준 라이브러리만 쓴다.

  1) whisper.cpp (whisper-cli · whisper-vad-speech-segments)
     macOS: Homebrew (brew install whisper-cpp)
     Windows·Linux: GitHub 공식 빌드(WHISPER_BUILD)를 받아 SHA-256 을 확인한 뒤 사용자 캐시에 푼다 (macOS 와 같은 1.9.5)
  2) repo 안 .venv 가상환경과 Python 패키지 (전역 Python 은 건드리지 않는다)
  3) Whisper · Silero VAD · 화자 인식 모델: repo 밖 사용자 캐시에 받아 크기·SHA-256 확인 (끊기면 이어받기)
  4) Claude Code CLI 확인 (요약용. 로그인 상태만 본다 — 전사문은 보내지 않는다)
  5) whisper.cpp 자체 점검 (만든 소리로 실제 전사·음성 구간 탐지 한 번)
  6) Linux: 필요한 시스템 라이브러리·한글 글꼴 확인 (sudo 가 필요한 설치는 명령만 알려 준다)
  7) 클릭해서 여는 바로 가기 (Windows: 시작 메뉴·바탕 화면, Linux: 앱 목록, macOS: ~/Applications 의 앱 아이콘)

막힌 단계가 있어도 나머지는 계속하고, 마지막에 막힌 단계와 직접 실행할 명령을 보여 준다(종료 코드 1).
사용: python scripts/setup.py [--no-launcher]   (보통은 setup_macos.sh · setup_linux.sh · setup_windows.bat 로 실행)
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from clinic_scribe import config, installer  # noqa: E402  (표준 라이브러리만 쓰는 모듈: 앱의 '설치' 창과 같은 규칙)

MIN_PYTHON = (3, 10)
MODELS = installer.MODELS
WHISPER_BUILD, WHISPER_BUILDS = installer.WHISPER_BUILD, installer.WHISPER_BUILDS
WHISPER_SOURCE_HELP = installer.WHISPER_SOURCE_HELP
CLAUDE_INSTALL = {
    "win32": 'powershell -ExecutionPolicy Bypass -c "irm https://claude.ai/install.ps1 | iex"   # 설치 후 claude 를 실행해 로그인',
    "other": "curl -fsSL https://claude.ai/install.sh | bash   # 설치 후 claude 를 실행해 로그인",
}
os_key, machine = installer.os_key, installer.machine


class Report:
    """진행 표시와 막힌 단계 모음."""

    def __init__(self) -> None:
        self.blocked: list[tuple[str, str]] = []

    @staticmethod
    def section(title: str) -> None:
        print(f"\n==> {title}", flush=True)

    @staticmethod
    def ok(text: str) -> None:
        print(f"  ✓ {text}", flush=True)

    @staticmethod
    def info(text: str) -> None:
        print(f"  · {text}", flush=True)

    def block(self, step: str, command: str) -> None:
        self.blocked.append((step, command))
        print(f"  ✗ {step}", flush=True)


def rerun_command() -> str:
    return config.setup_command()


def _progress(done: int, total: int) -> None:
    if sys.stdout.isatty():
        print(f"\r    {done / 1e6:,.1f} / {total / 1e6:,.1f} MB", end="" if done < total else "\n", flush=True)


def fetch(item: installer.Download, folder: Path, report: Report) -> Path | None:
    """크기와 SHA-256 이 모두 맞을 때만 쓴다. 끊기면 다음 실행 때 이어받는다."""
    if installer.is_valid(item, folder / item.name, check_hash=True):
        report.ok(f"이미 받은 파일 재사용: {folder / item.name}")
        return folder / item.name
    report.info(f"받는 중: {item.name} ({installer.human_size(item.size)})")
    try:
        path = installer.fetch(item, folder, _progress)
    except installer.InstallError as exc:
        report.block(str(exc), rerun_command())
        return None
    report.ok(f"다운로드·검증 완료: {path}")
    return path


# ---------------------------------------------------------------- 1) whisper.cpp
def find_whisper_cli() -> str | None:
    return config.find_executable(os.environ.get("CLINIC_SCRIBE_WHISPER_CLI", ""), "whisper-cli",
                                  config.whisper_cli_candidates())


def install_whisper(report: Report) -> str | None:
    found = find_whisper_cli()
    marker = config.whisper_dir() / ".build"
    pinned = marker.is_file() and marker.read_text(encoding="utf-8").strip() == WHISPER_BUILDS.get(
        (os_key(), machine()), installer.Download("", "", "", 0)).name
    if found and (os_key() == "darwin" or pinned or Path(found).parent != config.whisper_dir()):
        report.ok(f"기존 whisper-cli 재사용: {found}")
        return found
    report.info("whisper.cpp 설치: " + ("brew install whisper-cpp" if os_key() == "darwin"
                                        else f"공식 빌드 {WHISPER_BUILD} (SHA-256 확인)"))
    try:
        cli = installer.install_whisper(_progress)
    except installer.InstallError as exc:
        report.block(str(exc), "brew install whisper-cpp" if os_key() == "darwin" else WHISPER_SOURCE_HELP)
        return None
    report.ok(f"설치 완료: {cli}")
    return str(cli)


# ---------------------------------------------------------------- 2) Python 가상환경
def venv_python() -> Path:
    venv = REPO / ".venv"
    return venv / "Scripts" / "python.exe" if os_key() == "win32" else venv / "bin" / "python"


def make_venv(report: Report) -> Path | None:
    py = venv_python()
    report.info(f"기준 Python: {sys.executable} ({platform.python_version()})")
    if py.exists() and subprocess.run([str(py), "-c", "import sys"], check=False).returncode == 0:
        report.ok("기존 가상환경 재사용")
    else:
        made = subprocess.run([sys.executable, "-m", "venv", "--clear", str(REPO / ".venv")],
                              capture_output=True, text=True, check=False)
        if made.returncode != 0:
            if "ensurepip" in (made.stdout + made.stderr):  # Debian·Ubuntu: venv 모듈이 따로 나뉘어 있다
                version = f"{sys.version_info.major}.{sys.version_info.minor}"
                report.block("Python venv 모듈이 없음", f"sudo apt install python{version}-venv && {rerun_command()}")
            else:
                report.block(f"가상환경 생성 실패: {(made.stderr or made.stdout).strip()[-300:]}",
                             f'"{sys.executable}" -m venv "{REPO / ".venv"}"')
            return None
        report.ok("가상환경 생성")
    pip = [str(py), "-m", "pip", "--disable-pip-version-check", "install", "-q",
           "-r", str(REPO / "requirements.txt"), "-r", str(REPO / "requirements-dev.txt")]
    if subprocess.run(pip, check=False).returncode == 0:
        report.ok("Python 패키지 설치 완료")
    else:
        report.block("Python 패키지 설치 실패 (Python 3.10~3.13 권장)",
                     f'"{py}" -m pip install -r requirements.txt -r requirements-dev.txt')
        return None
    return py


# ---------------------------------------------------------------- 4) Claude Code CLI
def check_claude(report: Report) -> None:
    cli = config.find_executable(os.environ.get("CLINIC_SCRIBE_CLAUDE_CLI", ""), "claude", config.claude_cli_candidates())
    if not cli:
        report.block("claude 실행 파일을 찾지 못함 (요약 기능만 영향)", CLAUDE_INSTALL.get(os_key(), CLAUDE_INSTALL["other"]))
        return
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    try:
        version = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=30, env=env).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        version = ""
    report.ok(f"claude 실행 파일: {cli} ({version.splitlines()[0] if version else '버전 확인 안 됨'})")
    logged_in = installer.claude_logged_in(cli)  # 로그인 상태만 확인한다. 어떤 전사문도 보내지 않는다.
    if logged_in is True:
        report.ok("로그인됨")
    elif logged_in is False:
        report.block("Claude Code 에 로그인되어 있지 않음", f'"{cli}" auth login')
    else:
        report.info("로그인 상태를 확인하지 못했습니다. 요약이 실패하면 터미널에서 'claude' 를 실행해 로그인하세요.")


# ---------------------------------------------------------------- 5) 자체 점검
def write_check_tone(path: Path, seconds: float = 2.0, rate: int = 16000) -> None:
    """점검용 소리 (말소리 대역의 화음). 전사 내용은 보지 않고 whisper.cpp 가 끝까지 도는지만 본다."""
    frames = bytearray()
    for i in range(int(seconds * rate)):
        t = i / rate
        value = 0.25 * math.sin(2 * math.pi * 220 * t) + 0.15 * math.sin(2 * math.pi * 660 * t)
        frames += struct.pack("<h", int(value * 32767 * min(1.0, t * 4, (seconds - t) * 4)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def self_test(cli: str, model: Path, vad: Path, report: Report) -> None:
    vad_tool = Path(cli).with_name(config.exe_name("whisper-vad-speech-segments"))
    flags = {"creationflags": 0x08000000} if os_key() == "win32" else {}  # Windows: 콘솔 창 없이
    with tempfile.TemporaryDirectory(prefix="clinic-scribe-check-") as tmp:
        wav, out = Path(tmp) / "check.wav", Path(tmp) / "out"
        write_check_tone(wav)
        whisper = subprocess.run(
            [cli, "-m", config.native_path(model), "-f", config.native_path(wav), "-l", "ko", "-np", "-oj",
             "-of", config.native_path(out), "--vad", "-vm", config.native_path(vad)],
            capture_output=True, text=True, errors="replace", check=False, **flags,
        )
        if whisper.returncode != 0 or not out.with_suffix(".json").is_file():
            tail = "\n".join((whisper.stderr or "").strip().splitlines()[-4:])
            report.block(f"whisper-cli 자체 점검 실패: {tail or whisper.returncode}", WHISPER_SOURCE_HELP
                         if "GLIBC" in (whisper.stderr or "") else f'"{cli}" -m "{model}" -f "{wav}" -l ko')
            return
        report.ok("whisper-cli + Whisper 모델 + Silero VAD 동작 확인")
        if not vad_tool.is_file():
            vad_tool = Path(shutil.which("whisper-vad-speech-segments") or vad_tool)
        regions = subprocess.run([str(vad_tool), "-f", config.native_path(wav), "-vm", config.native_path(vad), "-np"],
                                 capture_output=True, text=True, errors="replace", check=False, **flags) \
            if vad_tool.is_file() else None
        if regions is not None and regions.returncode == 0:
            report.ok("음성 구간 탐지(whisper-vad-speech-segments) 동작 확인")
        else:
            report.block("whisper-vad-speech-segments 를 실행하지 못함 (줄 나누기·무음 줄이기에 필요)", rerun_command())


# ---------------------------------------------------------------- 6) Linux 시스템 라이브러리
LINUX_PACKAGES = {  # 찾을 라이브러리 → (apt, dnf, pacman) 꾸러미 이름
    "portaudio": ("libportaudio2", "portaudio", "portaudio"),  # 마이크 녹음 (sounddevice)
    "xcb-cursor": ("libxcb-cursor0", "xcb-util-cursor", "xcb-util-cursor"),  # Qt 화면 (X11)
    "EGL": ("libegl1", "mesa-libEGL", "libglvnd"),
    "xkbcommon-x11": ("libxkbcommon-x11-0", "libxkbcommon-x11", "libxkbcommon-x11"),
}
LINUX_FONT = ("fonts-noto-cjk", "google-noto-sans-cjk-fonts", "noto-fonts-cjk")


def check_linux(report: Report) -> None:
    import ctypes.util

    missing = [names for lib, names in LINUX_PACKAGES.items() if not ctypes.util.find_library(lib)]
    fc_list = shutil.which("fc-list")
    if fc_list and not subprocess.run([fc_list, ":lang=ko"], capture_output=True, text=True, check=False).stdout.strip():
        missing.append(LINUX_FONT)
    if not missing:
        report.ok("마이크·화면·한글 글꼴에 필요한 시스템 라이브러리 확인")
        return
    manager = next((m for m in ("apt", "dnf", "pacman") if shutil.which(m)), "apt")
    column = {"apt": 0, "dnf": 1, "pacman": 2}[manager]
    names = " ".join(item[column] for item in missing)
    command = {"apt": f"sudo apt install -y {names}", "dnf": f"sudo dnf install -y {names}",
               "pacman": f"sudo pacman -S --needed {names}"}[manager]
    report.block(f"시스템 꾸러미가 없음: {names}", f"{command} && {rerun_command()}")


# ---------------------------------------------------------------- 7) 바로 가기 (macOS 는 앱 아이콘)
def make_launcher(py: Path, report: Report) -> None:
    if os_key() == "darwin":
        make_mac_app(report)
        return
    folder = config.app_dir()
    folder.mkdir(parents=True, exist_ok=True)
    icon = folder / ("clinic-scribe.ico" if os_key() == "win32" else "clinic-scribe.png")
    drawn = subprocess.run(
        [str(py), "-c", "import sys; from PySide6.QtGui import QGuiApplication; app = QGuiApplication([]); "
                        "from clinic_scribe.ui.widgets import brand_image; sys.exit(0 if brand_image(256).save(sys.argv[1]) else 1)",
         str(icon)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(REPO)}, capture_output=True, check=False,
    )
    if drawn.returncode != 0:
        report.info("아이콘 그림을 만들지 못해 기본 아이콘을 씁니다.")
    try:
        where = windows_shortcuts(py, icon) if os_key() == "win32" else linux_desktop_entry(py, icon)
    except (OSError, subprocess.SubprocessError) as exc:
        report.block(f"바로 가기를 만들지 못함 ({exc})", config.run_command() + "   # 터미널에서 열기")
        return
    report.ok(f"바로 가기: {where}")


def make_mac_app(report: Report) -> None:
    """macOS: 클릭해서 여는 앱 아이콘 ~/Applications/clinic-scribe.app (scripts/make_launcher.sh).

    이 repo 를 여는 앱이 이미 있으면 그대로 둔다: 다시 만들면 macOS 가 마이크 권한을 한 번 더 묻는다.
    """
    app = Path.home() / "Applications" / "clinic-scribe.app"
    binary = app / "Contents" / "MacOS" / "clinic-scribe"
    try:
        same_repo = binary.is_file() and str(REPO).encode() in binary.read_bytes()
    except OSError:
        same_repo = False
    if same_repo:
        report.ok(f"앱 아이콘 재사용: {app}")
        return
    made = subprocess.run(["bash", str(REPO / "scripts" / "make_launcher.sh")], capture_output=True, text=True, check=False)
    if made.returncode != 0:
        reason = ((made.stderr or made.stdout).strip().splitlines() or ["알 수 없는 오류"])[-1]
        report.block(f"앱 아이콘을 만들지 못함: {reason}", "./scripts/make_launcher.sh")
        return
    report.ok(f"앱 아이콘: {app} (Spotlight 에서 'clinic' 으로 찾거나 Dock 에 끌어다 두세요)")


def windows_shortcuts(py: Path, icon: Path) -> str:
    """시작 메뉴와 바탕 화면에 clinic-scribe 바로 가기 (pythonw: 콘솔 창 없이). 경로는 환경 변수로 넘겨 따옴표 문제가 없다."""
    script = (
        "$shell = New-Object -ComObject WScript.Shell; "
        "foreach ($dir in @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))) { "
        "$lnk = $shell.CreateShortcut((Join-Path $dir 'clinic-scribe.lnk')); "
        "$lnk.TargetPath = $env:CS_TARGET; $lnk.Arguments = '-m clinic_scribe'; "
        "$lnk.WorkingDirectory = $env:CS_REPO; $lnk.IconLocation = $env:CS_ICON; "
        "$lnk.Description = 'clinic-scribe'; $lnk.Save() }"
    )
    env = {**os.environ, "CS_TARGET": str(py.with_name("pythonw.exe")), "CS_REPO": str(REPO), "CS_ICON": str(icon)}
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                   env=env, check=True, capture_output=True)
    return "시작 메뉴 · 바탕 화면 'clinic-scribe'"


def _desktop_quote(text: str) -> str:
    """freedesktop Exec 칸의 따옴표 규칙."""
    for ch in ('\\', '"', "`", "$"):
        text = text.replace(ch, "\\" + ch)
    return f'"{text}"'


def linux_desktop_entry(py: Path, icon: Path) -> str:
    folder = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "applications"
    folder.mkdir(parents=True, exist_ok=True)
    entry = folder / "clinic-scribe.desktop"
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=clinic-scribe\nComment=진료 녹음 · 전사 · 요약\n"
        f"Exec={_desktop_quote(str(py))} -m clinic_scribe\nPath={REPO}\nIcon={icon}\nTerminal=false\n"
        "Categories=Office;AudioVideo;\nStartupWMClass=clinic-scribe\n",
        encoding="utf-8",
    )
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(folder)], capture_output=True, check=False)
    return f"앱 목록 'clinic-scribe' ({entry})"


# ---------------------------------------------------------------- 전체
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="clinic-scribe 설치 (macOS · Windows · Linux)")
    parser.add_argument("--no-launcher", action="store_true", help="바로 가기를 만들지 않는다 (자동 검증용)")
    parser.add_argument("--local-model", choices=[m.id for m in installer.LOCAL_MODELS],
                        help="로컬 요약 모델도 받는다 (llama.cpp 와 함께). 앱의 '설치' 창에서 골라도 된다")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows 에서 출력을 파일로 돌려도 한글·기호가 깨지지 않게
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if sys.version_info < MIN_PYTHON:
        print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 이상이 필요합니다 (지금 {platform.python_version()}).", file=sys.stderr)
        return 1

    report = Report()
    model_dir = config.default_model_dir()
    print("clinic-scribe 설치")
    print(f"  OS:     {platform.system()} {platform.release()} ({machine()})")
    print(f"  repo:   {REPO}")
    print(f"  모델:   {model_dir}")
    print(f"  데이터: {config.app_dir()}")

    report.section("1/6 whisper.cpp (whisper-cli · 음성 구간 탐지)")
    cli = install_whisper(report)

    report.section("2/6 Python 가상환경 (.venv)")
    py = make_venv(report)

    report.section(f"3/6 Whisper · VAD · 화자 인식 모델 ({model_dir})")
    model_dir.mkdir(parents=True, exist_ok=True)
    models = {item.name: fetch(item, model_dir, report) for item in MODELS}

    report.section("4/6 요약: Claude Code CLI 또는 로컬 모델")
    check_claude(report)
    specs = installer.system_specs()
    report.info(f"이 컴퓨터: {specs.describe()}")
    for model in installer.LOCAL_MODELS:
        report.info(f"로컬 모델 {model.label} ({installer.human_size(model.file.size)}): {installer.model_fit(model, specs).text}")
    if args.local_model:
        model = installer.local_model(args.local_model)
        try:
            report.info(f"llama.cpp 공식 빌드 {installer.LLAMA_BUILD} 설치")
            report.ok(f"llama-server: {installer.install_llama(_progress)}")
        except installer.InstallError as exc:
            report.block(str(exc), rerun_command())
        fetch(model.file, model_dir, report)

    report.section("5/6 whisper.cpp 자체 점검")
    whisper_model, vad_model = models[config.WHISPER_MODEL_FILE], models[config.VAD_MODEL_FILE]
    if cli and whisper_model and vad_model:
        self_test(cli, whisper_model, vad_model, report)
    else:
        report.info("whisper-cli 또는 모델이 준비되지 않아 건너뜁니다.")

    report.section("6/6 시스템 확인 · 바로 가기")
    if os_key() == "linux":
        check_linux(report)
    if py and not args.no_launcher:
        make_launcher(py, report)

    report.section("결과")
    print(f"  whisper-cli : {cli or '(없음)'}")
    for item in MODELS:
        state = "(검증됨)" if models[item.name] else "(없음)"
        print(f"  {item.name}: {state}")
    if not report.blocked:
        print(f"\n설치 완료. 앱 실행: {config.run_command()}" + ("" if args.no_launcher
                                                            else " 또는 앱 아이콘·바로 가기 'clinic-scribe'"))
        return 0
    print("\n막힌 단계가 있습니다. 아래 명령을 직접 실행한 뒤 이 설치를 다시 실행하세요.")
    for i, (step, command) in enumerate(report.blocked, 1):
        print(f"  {i}) {step}\n     $ {command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
