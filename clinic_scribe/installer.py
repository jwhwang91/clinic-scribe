"""필요한 프로그램·모델 받기 (앱의 '설치' 창과 scripts/setup.py 가 함께 쓴다). 표준 라이브러리만 쓴다.

- 받는 것은 모두 버전·SHA-256 을 고정해 두고, 크기와 SHA-256 이 맞을 때만 쓴다(그 전까지는 .part, 끊기면 이어받기).
- whisper.cpp·llama.cpp 는 GitHub 공식 빌드를 받아 사용자 캐시에 푼다 (macOS 의 whisper.cpp 는 Homebrew).
- 로컬 요약 모델(Qwen3, Apache-2.0)은 Hugging Face 공식 GGUF. 이 컴퓨터 메모리·CPU·GPU 로 돌아가는지 미리 판단한다.
- Claude Code 는 공식 설치 스크립트로 설치한다 (로그인은 사용자가 한다).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from clinic_scribe import config

Progress = Callable[[int, int], None]  # (받은 바이트, 전체 바이트)


class InstallError(Exception):
    pass


class InstallCancelled(InstallError):
    pass


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    sha256: str
    size: int


HF = "https://huggingface.co/"
MODELS = (
    Download(config.WHISPER_MODEL_FILE, HF + "ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
             "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2", 574041195),
    # whisper.cpp v1.9.x 의 models/download-vad-model.sh 가 받는 공식 Silero 변환본
    Download(config.VAD_MODEL_FILE, HF + "ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin",
             "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987", 885098),
    # 말한 사람 알아보기: sherpa-onnx 공식 배포의 3D-Speaker CAM++ (SHA-256 은 배포의 checksum.txt 와 같다)
    Download(config.SPEAKER_MODEL_FILE,
             "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
             "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
             "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2", 28281164),
)

# whisper.cpp 공식 빌드: v1.9.5 와 같은 날 나온 b5454 (macOS Homebrew whisper-cpp 1.9.5 와 같은 버전·출력 형식).
# CPU 기능(SSE4.2 ~ AVX-512)마다 다른 계산 모듈이 함께 들어 있어 어떤 x64·ARM64 CPU 에서도 돈다.
WHISPER_BUILD = "b5454"
_W = f"https://github.com/ggml-org/whisper.cpp/releases/download/{WHISPER_BUILD}/"
WHISPER_BUILDS = {
    ("win32", "x64"): Download("whisper-bin-x64.zip", _W + "whisper-bin-x64.zip",
                               "6ba69e3482d7826214f90a6a9c84ca07782aec1e1d0c6a7c30c994fd5d816ccb", 8928640),
    ("win32", "arm64"): Download("whisper-bin-win-cpu-arm64.zip", _W + "whisper-bin-win-cpu-arm64.zip",
                                 "28c37e7b598c3d9bbfef94f3bd67f2ee6f12b7c86f3da5edcf5e4308d615ff6d", 4371193),
    ("linux", "x64"): Download("whisper-bin-ubuntu-x64.tar.gz", _W + "whisper-bin-ubuntu-x64.tar.gz",
                               "a72becf15d7917f990f6313867a52638b82b7f9ef237fb0c980dac56a135781c", 10364195),
    ("linux", "arm64"): Download("whisper-bin-ubuntu-arm64.tar.gz", _W + "whisper-bin-ubuntu-arm64.tar.gz",
                                 "6b95ebfc60447df48e70ef00a73bdc3f41679ed2d01ef465827206a2ff325149", 4608377),
}
WHISPER_SOURCE_HELP = ("git clone --branch v1.9.5 https://github.com/ggml-org/whisper.cpp && cd whisper.cpp && "
                       "cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --config Release  "
                       "# 그 뒤 설정에서 whisper-cli 경로 지정")

# llama.cpp 공식 빌드 (로컬 요약 모델 실행: llama-server). macOS 는 Metal(GPU), Windows·Linux 는 CPU 빌드.
LLAMA_BUILD = "b12138"
_L = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-"
LLAMA_BUILDS = {
    ("darwin", "arm64"): Download(f"llama-{LLAMA_BUILD}-macos-arm64.tar.gz", _L + "macos-arm64.tar.gz",
                                  "ef898f71f6f5d773565b84670eb0e6b9acf24a64d4c045e2e99fa24a9c9f7e8c", 16670361),
    ("darwin", "x64"): Download(f"llama-{LLAMA_BUILD}-macos-x64.tar.gz", _L + "macos-x64.tar.gz",
                                "16cd528378489276861dffd896862a57bf7523c9151a71909f566c36f348f55c", 12408482),
    ("win32", "x64"): Download(f"llama-{LLAMA_BUILD}-win-cpu-x64.zip", _L + "win-cpu-x64.zip",
                               "aa345c7052ef2d03814a3d5a13bd55e70bc27057ee3021d8371199f6ae67e355", 24728204),
    ("win32", "arm64"): Download(f"llama-{LLAMA_BUILD}-win-cpu-arm64.zip", _L + "win-cpu-arm64.zip",
                                 "bb3ed4f64d39512948365b97c3163cddeeff6b35dac3fb858a4b6f12aeb12137", 13222742),
    ("linux", "x64"): Download(f"llama-{LLAMA_BUILD}-ubuntu-x64.tar.gz", _L + "ubuntu-x64.tar.gz",
                               "25509d3d9721926394361a78f8f242ac5213a5ba46b13c96a704bc77358f5760", 23077553),
    ("linux", "arm64"): Download(f"llama-{LLAMA_BUILD}-ubuntu-arm64.tar.gz", _L + "ubuntu-arm64.tar.gz",
                                 "18cc629e75e31509c1378645b391b34c8e1d2241274fd116c5fb01aca71ae8f0", 18754438),
}


@dataclass(frozen=True)
class LocalModel:
    """로컬 요약 모델 (Qwen3 GGUF Q4_K_M). ram_gb: 모델 + 문맥 16K 를 올리는 데 필요한 메모리."""

    id: str
    label: str
    file: Download
    ram_gb: float
    active_gb: float  # 낱말 하나를 만들 때 읽는 가중치 크기 (속도 계산용. 전문가 혼합 모델은 일부만 읽는다)
    note: str


def _qwen(size: str, sha: str, nbytes: int) -> Download:
    name = f"Qwen3-{size}-Q4_K_M.gguf"
    return Download(name, f"{HF}Qwen/Qwen3-{size}-GGUF/resolve/main/{name}", sha, nbytes)


LOCAL_MODELS = (
    LocalModel("qwen3-4b", "Qwen3 4B", _qwen("4B", "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
                                             2497280256), 5.0, 2.5, "가장 가벼움 · 빠뜨리는 내용이 조금 많음"),
    LocalModel("qwen3-8b", "Qwen3 8B", _qwen("8B", "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
                                             5027783488), 8.0, 5.0, "균형 · 메모리 16GB 이상 권장"),
    LocalModel("qwen3-14b", "Qwen3 14B", _qwen("14B", "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
                                               9001752960), 12.0, 9.0, "품질 좋음 · 메모리 24GB 이상 권장"),
    LocalModel("qwen3-30b-a3b", "Qwen3 30B-A3B",
               _qwen("30B-A3B", "0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5", 18556685824),
               21.0, 2.0, "로컬 최고 품질 · 크기에 비해 빠름 · 메모리 32GB 이상"),
)
DEFAULT_LOCAL_MODEL = "qwen3-8b"

CLAUDE_INSTALL = {  # 공식 설치 스크립트 (설치 위치: ~/.local/bin/claude, Windows %USERPROFILE%\.local\bin\claude.exe)
    "win32": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "irm https://claude.ai/install.ps1 | iex"],
    "other": ["bash", "-c", "curl -fsSL https://claude.ai/install.sh | bash"],
}


# ---------------------------------------------------------------- 이 컴퓨터
def os_key() -> str:
    if sys.platform == "win32":
        return "win32"
    return "darwin" if sys.platform == "darwin" else "linux"


def machine() -> str:
    m = platform.machine().lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    return "x64" if m in ("x86_64", "amd64", "x64") else m


@dataclass(frozen=True)
class Specs:
    os: str
    machine: str
    cpu: str
    threads: int
    ram_gb: float
    gpu: str  # "Apple Metal" / "NVIDIA …" / ""
    metal: bool  # Apple Silicon: 메모리를 GPU 와 함께 쓴다 (llama.cpp Metal 빌드)

    def describe(self) -> str:
        gpu = f" · {self.gpu}" if self.gpu else " · GPU 가속 없음(CPU)"
        return f"{self.cpu} · 메모리 {self.ram_gb:.0f}GB · 스레드 {self.threads}{gpu}"


def system_specs() -> Specs:
    key, arch = os_key(), machine()
    metal = key == "darwin" and arch == "arm64"
    nvidia = _nvidia_gpu()
    gpu = "Apple Metal" if metal else (f"{nvidia} (이 앱의 로컬 요약은 CPU 로 실행)" if nvidia else "")
    return Specs(key, arch, _cpu_name(), os.cpu_count() or 4, round(_ram_bytes() / 2**30, 1), gpu, metal)


def _ram_bytes() -> int:
    try:
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout)
        if sys.platform == "win32":
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_avail", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_avail", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.total)
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        pass
    return 0


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
            return name or platform.processor() or "CPU"
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "CPU"


def _nvidia_gpu() -> str:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return ""
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10, **_quiet()).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return ""
    return f"NVIDIA {out[0].strip()}" if out else ""


@dataclass(frozen=True)
class Fit:
    level: str  # "good" 원활 / "slow" 느릴 수 있음 / "no" 메모리 부족
    text: str
    minutes: float


RESERVED_GB = 4.0  # 운영체제·이 앱·Whisper 전사가 쓰는 메모리
SUMMARY_TOKENS = 700  # 요약 하나에 만드는 낱말 수 (대략)


def model_fit(model: LocalModel, specs: Specs) -> Fit:
    """이 컴퓨터에서 그 모델이 돌아가는지와 요약 하나에 걸리는 대략의 시간."""
    if specs.ram_gb and model.ram_gb > specs.ram_gb - RESERVED_GB:
        return Fit("no", f"메모리 부족 (필요 약 {model.ram_gb:.0f}GB · 이 컴퓨터 {specs.ram_gb:.0f}GB)", 0.0)
    # 낱말 하나에 가중치를 한 번 읽는다: Apple Silicon 통합 메모리 약 100GB/s, 일반 PC CPU 약 20GB/s 로 어림한다.
    bandwidth = 100.0 if specs.metal else 20.0
    minutes = round((model.active_gb / bandwidth * SUMMARY_TOKENS + (10 if specs.metal else 30)) / 60, 1)
    if specs.metal and model.ram_gb > specs.ram_gb * 0.7:
        return Fit("slow", f"느릴 수 있음 · GPU 메모리가 빠듯함 · {_duration(minutes * 2)}", minutes * 2)
    if minutes <= 2.0:
        return Fit("good", f"원활 · {_duration(minutes)}", minutes)
    return Fit("slow", f"느림 · {_duration(minutes)} (CPU)", minutes)


def _duration(minutes: float) -> str:
    return "요약 1분 이내" if minutes < 1 else f"요약 약 {minutes:.0f}분"


def recommended_model(specs: Specs) -> LocalModel:
    """'원활' 중 가장 큰 모델. 없으면 돌아가는 것 중 가장 가벼운 모델."""
    good = [m for m in LOCAL_MODELS if model_fit(m, specs).level == "good"]
    if good:
        return max(good, key=lambda m: m.ram_gb)
    runnable = [m for m in LOCAL_MODELS if model_fit(m, specs).level != "no"]
    return min(runnable or LOCAL_MODELS, key=lambda m: m.ram_gb)


def local_model(model_id: str) -> LocalModel | None:
    return next((m for m in LOCAL_MODELS if m.id == model_id), None)


# ---------------------------------------------------------------- 무엇이 없나
def llama_dir() -> Path:
    return config.cache_dir() / "llama.cpp" / "bin"


def llama_server_path() -> Path:
    return llama_dir() / config.exe_name("llama-server")


def local_model_path(model: LocalModel) -> Path:
    return config.default_model_dir() / model.file.name


def is_valid(item: Download, path: Path, check_hash: bool = False) -> bool:
    """받아 둔 파일이 맞는지. 앱이 켤 때는 크기만 보고(큰 파일), 설치할 때는 SHA-256 까지 본다."""
    try:
        if path.stat().st_size != item.size:
            return False
    except OSError:
        return False
    return not check_hash or sha256_of(path) == item.sha256


def whisper_ready(cli: str | None) -> bool:
    return bool(cli) and Path(cli).with_name(config.exe_name("whisper-vad-speech-segments")).is_file()


# ---------------------------------------------------------------- 받기
def sha256_of(path: Path, cancel: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            if cancel is not None and cancel.is_set():
                raise InstallCancelled("취소했습니다.")
            digest.update(chunk)
    return digest.hexdigest()


def fetch(item: Download, folder: Path, progress: Progress | None = None, cancel: threading.Event | None = None) -> Path:
    """item 을 folder 에 받는다. 크기와 SHA-256 이 맞을 때만 item.name 으로 두고, 그 전까지는 .part (이어받기)."""
    folder.mkdir(parents=True, exist_ok=True)
    dest, part = folder / item.name, folder / (item.name + ".part")
    if dest.exists():
        if is_valid(item, dest, check_hash=True):
            return dest
        dest.unlink()  # 손상되었거나 다른 파일: 다시 받는다
    if part.exists() and part.stat().st_size > item.size:
        part.unlink()
    for attempt in range(3):
        have = part.stat().st_size if part.exists() else 0
        if have < item.size:
            try:
                _download(item.url, part, have, item.size, progress, cancel)
            except InstallCancelled:
                raise
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                if attempt == 2:
                    raise InstallError(f"{item.name} 을(를) 받지 못했습니다 ({exc}). 받은 부분은 다음에 이어받습니다.") from exc
                continue
        if part.stat().st_size == item.size and sha256_of(part, cancel) == item.sha256:
            os.replace(part, dest)
            return dest
        part.unlink(missing_ok=True)  # 크기나 SHA-256 이 다르다: 처음부터 다시
    raise InstallError(f"{item.name} 검증에 실패했습니다 (크기 또는 SHA-256 불일치). 인터넷 연결을 확인하고 다시 시도하세요.")


def _download(url: str, part: Path, have: int, total: int, progress: Progress | None,
              cancel: threading.Event | None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "clinic-scribe"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = bool(have) and getattr(response, "status", 200) == 206
        done = have if resumed else 0
        with open(part, "ab" if resumed else "wb") as out:
            while True:
                if cancel is not None and cancel.is_set():
                    raise InstallCancelled("취소했습니다.")
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)


def unpack(archive: Path, target: Path, program: str, needed: tuple[str, ...] = ()) -> Path:
    """공식 빌드 압축 파일을 풀어 program 이 든 폴더의 내용(실행 파일·라이브러리)을 target 에 둔다."""
    exe = config.exe_name(program)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                for member in z.namelist():
                    if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                        raise InstallError(f"압축 파일 안의 경로가 이상합니다: {member}")
                z.extractall(tmp_path)
        else:
            with tarfile.open(archive) as t:
                if hasattr(tarfile, "data_filter"):
                    t.extractall(tmp_path, filter="data")  # 압축 파일 밖으로 나가는 경로·장치 파일을 막는다
                else:
                    for member in t.getmembers():
                        if member.name.startswith("/") or ".." in Path(member.name).parts:
                            raise InstallError(f"압축 파일 안의 경로가 이상합니다: {member.name}")
                    t.extractall(tmp_path)  # noqa: S202 — 위에서 경로를 확인했다
        found = next((p for p in tmp_path.rglob(exe) if p.is_file()), None)
        if found is None:
            raise InstallError(f"압축 파일 안에 {exe} 이(가) 없습니다.")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(found.parent, target, symlinks=True)
    for name in (program, *needed):
        path = target / config.exe_name(name)
        if not path.is_file():
            raise InstallError(f"{path.name} 이(가) 없습니다.")
        if os_key() != "win32":
            path.chmod(path.stat().st_mode | 0o111)
    return target / exe


def install_build(build: Download, target: Path, program: str, needed: tuple[str, ...] = (),
                  progress: Progress | None = None, cancel: threading.Event | None = None) -> Path:
    """공식 빌드를 받아(SHA-256 확인) target 에 푼다. 이미 같은 빌드가 있으면 그대로 쓴다."""
    marker = target / ".build"
    exe = target / config.exe_name(program)
    if exe.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == build.name:
        return exe
    archive = fetch(build, config.cache_dir() / "downloads", progress, cancel)
    unpack(archive, target, program, needed)
    marker.write_text(build.name, encoding="utf-8")
    archive.unlink(missing_ok=True)  # 풀었으면 압축 파일은 지운다
    return exe


def install_whisper(progress: Progress | None = None, cancel: threading.Event | None = None) -> Path:
    """Windows·Linux: whisper.cpp 공식 빌드. macOS: Homebrew (brew install whisper-cpp)."""
    if os_key() == "darwin":
        brew = shutil.which("brew") or next((b for b in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")
                                             if os.path.exists(b)), None)
        if not brew:
            raise InstallError("Homebrew 가 없어 whisper.cpp 를 설치하지 못했습니다. https://brew.sh 에서 Homebrew 를 "
                               "설치한 뒤 다시 시도하세요.")
        if subprocess.run([brew, "install", "whisper-cpp"], capture_output=True, check=False).returncode != 0:
            raise InstallError("brew install whisper-cpp 가 실패했습니다. 터미널에서 직접 실행해 보세요.")
        cli = config.find_executable("", "whisper-cli", config.whisper_cli_candidates())
        if not cli:
            raise InstallError("whisper-cli 를 설치했지만 찾지 못했습니다.")
        return Path(cli)
    build = WHISPER_BUILDS.get((os_key(), machine()))
    if build is None:
        raise InstallError(f"이 OS·CPU({os_key()} {machine()}) 용 whisper.cpp 공식 빌드가 없습니다. 직접 빌드하세요: "
                           + WHISPER_SOURCE_HELP)
    return install_build(build, config.whisper_dir(), "whisper-cli", ("whisper-vad-speech-segments",), progress, cancel)


def install_llama(progress: Progress | None = None, cancel: threading.Event | None = None) -> Path:
    build = LLAMA_BUILDS.get((os_key(), machine()))
    if build is None:
        raise InstallError(f"이 OS·CPU({os_key()} {machine()}) 용 llama.cpp 공식 빌드가 없습니다.")
    return install_build(build, llama_dir(), "llama-server", (), progress, cancel)


def install_claude(cancel: threading.Event | None = None) -> str:
    """Claude Code 공식 설치 스크립트를 실행하고 설치된 claude 경로를 돌려준다 (로그인은 사용자가 한다)."""
    command = CLAUDE_INSTALL["win32" if os_key() == "win32" else "other"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **_quiet())
    while True:
        try:
            out, _ = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if cancel is not None and cancel.is_set():
                proc.kill()
                raise InstallCancelled("취소했습니다.") from None
    cli = config.find_executable("", "claude", config.claude_cli_candidates())
    if proc.returncode != 0 or not cli:
        tail = (out or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
        raise InstallError("Claude Code 설치에 실패했습니다: " + (" / ".join(tail) or f"종료 코드 {proc.returncode}"))
    return cli


def claude_logged_in(cli: str) -> bool | None:
    """로그인 상태 (모르면 None). 이 확인으로는 어떤 전사문도 보내지 않는다."""
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    try:
        out = subprocess.run([cli, "auth", "status", "--json"], capture_output=True, text=True, timeout=30, env=env,
                             **_quiet()).stdout
        value = json.loads(out or "{}").get("loggedIn")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return value if isinstance(value, bool) else None


def open_claude_login(cli: str) -> bool:
    """터미널 창을 열어 'claude auth login' 을 실행한다 (브라우저로 로그인). 열지 못하면 False."""
    try:
        if os_key() == "darwin":
            script = Path(tempfile.gettempdir()) / "clinic-scribe-claude-login.command"
            script.write_text(f"#!/bin/bash\n{_sh_quote(cli)} auth login\n", encoding="utf-8")
            script.chmod(0o700)
            return subprocess.run(["open", str(script)], check=False).returncode == 0
        if os_key() == "win32":
            subprocess.Popen(["cmd", "/c", "start", "", "cmd", "/k", cli, "auth", "login"], **_quiet())
            return True
        for terminal in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
            path = shutil.which(terminal)
            if path:
                args = [path, "--", cli, "auth", "login"] if terminal == "gnome-terminal" else [path, "-e", cli, "auth", "login"]
                subprocess.Popen(args)
                return True
    except OSError:
        pass
    return False


def _sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def _quiet() -> dict:
    """Windows: 콘솔 창 없이 실행."""
    return {"creationflags": 0x08000000} if os_key() == "win32" else {}


def human_size(nbytes: int) -> str:
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.1f}GB"
    return f"{nbytes / 1e6:.0f}MB" if nbytes >= 1e6 else f"{max(nbytes, 1) / 1e3:.0f}KB"
