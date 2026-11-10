"""설정 값과 경로.

설정 파일은 repo 밖(~/Library/Application Support/clinic-scribe/settings.json)에 둔다.
모델은 setup_macos.sh 와 같은 규칙으로 사용자 캐시(~/.cache/clinic-scribe/models)에서 찾는다.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "clinic-scribe"
WHISPER_MODEL_FILE = "ggml-large-v3-turbo-q5_0.bin"
VAD_MODEL_FILE = "ggml-silero-v6.2.0.bin"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT_CHOICES = ("", "low", "medium", "high", "xhigh", "max")

# Whisper 초기 프롬프트. 영어 약명·약어를 한글로 음역하지 않고 원문 표기로 남기도록 유도한다.
# 프롬프트에 들어 있는 단어일수록 원문 표기로 나온다. 자주 쓰는 약명을 설정에서 추가하면 된다.
# (프롬프트 없이: "에서테미너펀 500mg", 이 프롬프트로: "acetaminophen 500mg", "CBC", "MRI")
DEFAULT_WHISPER_PROMPT = (
    "CBC, CT, MRI 검사와 PPI, NSAIDs, omeprazole, acetaminophen, amoxicillin 처방이 나오는 한국어 진료 대화."
)


def default_model_dir() -> Path:
    env = os.environ.get("CLINIC_SCRIBE_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / APP_NAME / "models"


def settings_path() -> Path:
    env = os.environ.get("CLINIC_SCRIBE_SETTINGS")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Library" / "Application Support" / APP_NAME / "settings.json"


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
    auto_summarize: bool = True
    show_evidence: bool = True
    live_preview: bool = True  # 녹음 중 실시간 전사 미리보기 (whisper.cpp, 이 Mac)

    # ---- 경로 해석 -------------------------------------------------------
    def whisper_model_path(self) -> Path:
        return Path(self.whisper_model).expanduser() if self.whisper_model else default_model_dir() / WHISPER_MODEL_FILE

    def vad_model_path(self) -> Path:
        return Path(self.vad_model).expanduser() if self.vad_model else default_model_dir() / VAD_MODEL_FILE

    def whisper_cli_path(self) -> str | None:
        return find_executable(
            self.whisper_cli or os.environ.get("CLINIC_SCRIBE_WHISPER_CLI", ""),
            "whisper-cli",
            ["/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli"],
        )

    def claude_cli_path(self) -> str | None:
        home = Path.home()
        return find_executable(
            self.claude_cli or os.environ.get("CLINIC_SCRIBE_CLAUDE_CLI", ""),
            "claude",
            [
                str(home / ".local/bin/claude"),
                str(home / ".claude/local/claude"),
                "/opt/homebrew/bin/claude",
                "/usr/local/bin/claude",
            ],
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
        return settings

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def find_executable(configured: str, name: str, fallbacks: list[str]) -> str | None:
    """설정값 → PATH → 흔한 설치 위치 순으로 실행 파일을 찾는다.

    Finder/IDE 에서 실행하면 PATH 에 ~/.local/bin, /opt/homebrew/bin 이 없을 수 있어 fallback 을 둔다.
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
