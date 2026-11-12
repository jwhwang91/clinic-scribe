"""설정 값과 경로 (macOS · Windows · Linux).

설정·의사 목소리·딕션 사전·진료 기록은 repo 밖 사용자 폴더(app_dir)에 둔다.
  macOS    ~/Library/Application Support/clinic-scribe
  Windows  %LOCALAPPDATA%\\clinic-scribe
  Linux    ~/.local/share/clinic-scribe   ($XDG_DATA_HOME)
모델과 Windows·Linux 용 whisper.cpp 는 설치 스크립트(scripts/setup.py)와 같은 규칙으로 사용자 캐시(cache_dir)에 둔다.
  macOS·Linux  ~/.cache/clinic-scribe   ($XDG_CACHE_HOME)      Windows  %LOCALAPPDATA%\\clinic-scribe
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "clinic-scribe"
WHISPER_MODEL_FILE = "ggml-large-v3-turbo-q5_0.bin"
VAD_MODEL_FILE = "ggml-silero-v6.2.0.bin"
# 말한 사람 알아보기(화자 인식): 3D-Speaker CAM++ (중국어·영어 대규모 화자로 학습), sherpa-onnx 로 실행
SPEAKER_MODEL_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT_CHOICES = ("", "low", "medium", "high", "xhigh", "max")
SUMMARY_MODES = {"general": "주요 내용 요약", "chart": "진료 차트 초안 (검토용)"}
VIEW_MODES = ("simple", "developer")

# Whisper 초기 프롬프트. 영어 약명·약어를 한글로 음역하지 않고 원문 표기로 남기도록 유도한다.
# 프롬프트에 들어 있는 단어일수록 원문 표기로 나온다. 자주 쓰는 약명을 설정에서 추가하면 된다.
# (프롬프트 없이: "에서테미너펀 500mg", 이 프롬프트로: "acetaminophen 500mg", "CBC", "MRI")
# 단어만 나열한다. "…진료 대화." 같은 문장형 프롬프트는 일상 대화를 문장 구분 없이 한 줄로 뭉쳐 버렸다.
DEFAULT_WHISPER_PROMPT = "CBC, CT, MRI, PPI, NSAIDs, omeprazole, acetaminophen, amoxicillin."
# 이전 기본값. 저장된 설정에 그대로 남아 있으면 새 기본값으로 바꾼다(사용자가 고친 프롬프트는 유지).
LEGACY_WHISPER_PROMPTS = (
    "CBC, CT, MRI 검사와 PPI, NSAIDs, omeprazole, acetaminophen, amoxicillin 처방이 나오는 한국어 진료 대화.",
)


def _local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")


def app_dir() -> Path:
    """설정·의사 목소리·딕션 사전·진료 기록을 두는 사용자 폴더 (OS 관례 위치, repo 밖, 이 컴퓨터에만)."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if sys.platform == "win32":
        return _local_appdata() / APP_NAME  # 로밍 프로필로 다른 PC 에 복사되지 않는 곳 (목소리 특징값·진료 녹음)
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / APP_NAME


def cache_dir() -> Path:
    """모델과 받은 whisper.cpp 를 두는 사용자 캐시 (지워도 설치 스크립트로 다시 받을 수 있는 것)."""
    if sys.platform == "win32":
        return _local_appdata() / APP_NAME
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / APP_NAME


def default_model_dir() -> Path:
    env = os.environ.get("CLINIC_SCRIBE_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return cache_dir() / "models"


def whisper_dir() -> Path:
    """설치 스크립트가 받은 whisper.cpp 공식 빌드 (Windows·Linux. macOS 는 Homebrew)."""
    return cache_dir() / "whisper.cpp" / "bin"


def exe_name(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def setup_command() -> str:
    """이 OS 의 설치 명령 (안내 문구용)."""
    if sys.platform == "win32":
        return r"scripts\setup_windows.bat"
    return "./scripts/setup_macos.sh" if sys.platform == "darwin" else "./scripts/setup_linux.sh"


def run_command() -> str:
    """이 OS 에서 터미널로 앱을 여는 명령 (안내 문구용)."""
    return r"scripts\run.bat" if sys.platform == "win32" else "./scripts/run.sh"


def native_path(path: Path | str) -> str:
    """외부 프로그램(whisper-cli·화자 인식 모델)에 넘길 경로.

    Windows 의 whisper.cpp 등은 한글처럼 ASCII 밖 글자가 든 경로(예: C:\\Users\\홍길동\\…)를 열지 못할 수 있어
    8.3 짧은 경로(C:\\Users\\B1FD~1\\…)로 바꿔 넘긴다. 아직 없는 파일(출력 파일)은 폴더만 바꾼다. 다른 OS 는 그대로.
    """
    text = str(path)
    if sys.platform != "win32" or text.isascii():
        return text
    p = Path(text)
    if p.exists():
        short = _short_path(text)
        return short if short and short.isascii() else text
    parent = _short_path(str(p.parent)) if p.parent.exists() else None
    if parent and parent.isascii() and p.name.isascii():
        return str(Path(parent) / p.name)
    return text


def _short_path(path: str) -> str | None:
    """Windows GetShortPathNameW (8.3 이름을 만들지 않는 드라이브면 None)."""
    try:
        import ctypes

        get = ctypes.windll.kernel32.GetShortPathNameW
        size = get(path, None, 0)
        if not size:
            return None
        buffer = ctypes.create_unicode_buffer(size)
        return buffer.value if get(path, buffer, size) else None
    except (AttributeError, OSError):
        return None


def settings_path() -> Path:
    env = os.environ.get("CLINIC_SCRIBE_SETTINGS")
    if env:
        return Path(env).expanduser()
    return app_dir() / "settings.json"


def voices_path() -> Path:
    """등록한 의사 목소리 특징값(생체인식정보) 파일. 설정 파일과 같은 폴더, 이 컴퓨터에만 둔다."""
    return settings_path().parent / "voices.json"


def diction_path() -> Path:
    """의사별 딕션 사전(용어 목록·교정). 설정 파일과 같은 폴더."""
    return settings_path().parent / "diction.json"


def records_dir() -> Path:
    """진료 기록(무음을 줄인 녹음 · 전사문 · 요약). 설정 파일과 같은 폴더, 이 컴퓨터에만 둔다."""
    return settings_path().parent / "records"


def whisper_cli_candidates() -> list[str]:
    """PATH 에 없을 때 찾아볼 whisper-cli 위치 (설치 스크립트가 받은 공식 빌드, Homebrew 등)."""
    built = str(whisper_dir() / exe_name("whisper-cli"))
    if sys.platform == "darwin":
        return ["/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli", built]
    if sys.platform == "win32":
        return [built]
    return [built, "/usr/local/bin/whisper-cli", "/usr/bin/whisper-cli"]


def claude_cli_candidates() -> list[str]:
    """PATH 에 없을 때 찾아볼 Claude Code CLI 위치 (공식 설치 프로그램, npm 전역 설치)."""
    home = Path.home()
    if sys.platform == "win32":
        npm = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming") / "npm"
        return [str(home / ".local" / "bin" / "claude.exe"), str(npm / "claude.cmd")]
    return [
        str(home / ".local/bin/claude"),
        str(home / ".claude/local/claude"),
        "/opt/homebrew/bin/claude",
        "/usr/local/bin/claude",
        str(home / ".npm-global/bin/claude"),
    ]


@dataclass
class VadParams:
    """whisper.cpp Silero VAD 파라미터 (whisper-cli --vad-* 옵션과 1:1)."""

    threshold: float = 0.5
    # 짧은 대답("네", "아니요")이 버려지지 않도록 whisper.cpp 기본값(250ms)보다 낮게 둔다.
    min_speech_ms: int = 150
    # 이보다 긴 무음에서만 구간을 나눈다. 문장 사이의 짧은 쉼은 한 구간으로 합쳐진다.
    # (whisper.cpp 기본값 100ms 는 쉼표 쉼에서도 문장을 끊는다.)
    min_silence_ms: int = 600
    # 구간 앞뒤 여유. 문장 첫소리·끝소리가 잘리지 않게 한다. (기본값 30ms 는 문장 끝을 자른다.)
    speech_pad_ms: int = 200
    # 한 구간의 최대 길이(초). 0 이면 제한하지 않는다.
    max_speech_s: float = 0.0
    samples_overlap_s: float = 0.1

    def describe(self) -> str:
        return f"무음 {self.min_silence_ms}ms · 여유 {self.speech_pad_ms}ms · 최소 발화 {self.min_speech_ms}ms"


@dataclass
class Settings:
    # 비워 두면 자동으로 찾는다.
    whisper_cli: str = ""
    whisper_model: str = ""
    vad_model: str = ""
    language: str = "ko"
    whisper_prompt: str = DEFAULT_WHISPER_PROMPT
    whisper_threads: int = 0
    vad_enabled: bool = True
    vad: VadParams = field(default_factory=VadParams)
    input_device: str = ""
    claude_cli: str = ""
    claude_model: str = DEFAULT_CLAUDE_MODEL
    claude_effort: str = ""
    claude_timeout_s: int = 300
    # 요약 방식: "claude" (Claude Code CLI · 전사문 전송) / "local" (이 컴퓨터의 로컬 모델 · 전송 없음) / "" 아직 안 고름
    summary_engine: str = ""
    local_model: str = "qwen3-8b"  # installer.LOCAL_MODELS 의 id
    auto_summarize: bool = True
    summary_mode: str = "general"  # "general" 주요 내용 요약 / "chart" 진료 차트 초안
    view_mode: str = "simple"  # "simple" 간단 화면 / "developer" 개발자 대시보드
    show_evidence: bool = True
    live_preview: bool = True  # 녹음 중 실시간 전사 미리보기 (whisper.cpp, 이 컴퓨터)
    # 자동 진료 나누기: 스페이스바로 끝내는 것을 깜빡했을 때의 대비책. 녹음은 이어지고 앞 진료만 따로 요약한다.
    auto_split: bool = True
    split_silence_s: int = 90  # 이만큼 말이 없으면 진료가 끝난 것으로 본다
    split_cues_next: str = "들어오세요, 들어오실게요, 다음 환자, 다음 분 오세요"  # 다음 환자 신호: 이 말부터 새 진료
    split_cues_end: str = "안녕히 가세요, 안녕히 가십시오, 조심히 가세요, 조심해서 가세요, 다음에 뵙겠습니다"  # 끝인사: 이 말까지
    min_session_s: int = 20  # 이보다 짧은 진료는 신호 단어로 나누지 않는다 (오인식으로 잘게 쪼개지는 것 방지)
    auto_stop_idle_s: int = 600  # 이만큼 말이 없으면 녹음 자체를 멈춰 마이크를 놓는다 (0 = 끄기)
    # 말한 사람 알아보기: 등록한 의사 목소리와 비교한다 (등록한 의사가 있을 때만 동작)
    speaker_model: str = ""
    speaker_threshold: float = 0.55  # 등록 목소리와의 유사도가 이 이상이면 그 의사로 본다
    split_doctor_only: bool = True  # 신호 단어는 등록한 의사 목소리일 때만 진료를 나눈다

    # ---- 경로 해석 -------------------------------------------------------
    def whisper_model_path(self) -> Path:
        return Path(self.whisper_model).expanduser() if self.whisper_model else default_model_dir() / WHISPER_MODEL_FILE

    def vad_model_path(self) -> Path:
        return Path(self.vad_model).expanduser() if self.vad_model else default_model_dir() / VAD_MODEL_FILE

    def speaker_model_path(self) -> Path:
        return Path(self.speaker_model).expanduser() if self.speaker_model else default_model_dir() / SPEAKER_MODEL_FILE

    def whisper_cli_path(self) -> str | None:
        return find_executable(
            self.whisper_cli or os.environ.get("CLINIC_SCRIBE_WHISPER_CLI", ""), "whisper-cli", whisper_cli_candidates()
        )

    def claude_cli_path(self) -> str | None:
        return find_executable(
            self.claude_cli or os.environ.get("CLINIC_SCRIBE_CLAUDE_CLI", ""), "claude", claude_cli_candidates()
        )

    # ---- 저장/불러오기 ----------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or settings_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in raw.items() if k in known and k != "vad"}
        settings = cls(**values)
        if isinstance(raw.get("vad"), dict):
            vad_known = {f.name for f in fields(VadParams)}
            settings.vad = VadParams(**{k: v for k, v in raw["vad"].items() if k in vad_known})
        if settings.whisper_prompt in LEGACY_WHISPER_PROMPTS:
            settings.whisper_prompt = DEFAULT_WHISPER_PROMPT
        if settings.summary_mode not in SUMMARY_MODES:
            settings.summary_mode = "general"
        if settings.view_mode not in VIEW_MODES:
            settings.view_mode = "simple"
        return settings

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def find_executable(configured: str, name: str, fallbacks: list[str]) -> str | None:
    """설정값 → PATH → 흔한 설치 위치 순으로 실행 파일을 찾는다.

    Finder·바로 가기·IDE 에서 실행하면 PATH 에 ~/.local/bin, /opt/homebrew/bin 등이 없을 수 있어 fallback 을 둔다.
    """
    if configured:
        p = Path(configured).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        found = shutil.which(configured)
        return found
    found = shutil.which(name)
    if found:
        return found
    for candidate in fallbacks:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None
