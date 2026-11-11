import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "samples"
# 테스트가 쓰는 가상 음성. 하나라도 없으면 다시 만든다 (생성기에 새 음성을 추가했을 때).
REQUIRED_SAMPLES = (
    "manifest.json", "mock_consult_short.wav", "mock_consult.wav", "long_pause.wav", "injection.wav", "daily_chat.wav",
    "two_patients.wav", "voice_enroll_doctor.wav", "voice_enroll_other.wav", "silence.wav", "quiet_room.wav",
)


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """테스트가 사용자 설정 파일(~/Library/Application Support/clinic-scribe)을 건드리지 않게 한다."""
    monkeypatch.setenv("CLINIC_SCRIBE_SETTINGS", str(tmp_path / "settings.json"))


@pytest.fixture(scope="session")
def samples() -> Path:
    """가상 진료 대화 음성. 없으면 scripts/make_test_audio.py 로 만든다 (macOS `say` 필요)."""
    if any(not (SAMPLES / name).is_file() for name in REQUIRED_SAMPLES):
        subprocess.run([sys.executable, str(REPO / "scripts" / "make_test_audio.py"), str(SAMPLES)], check=True)
    return SAMPLES


@pytest.fixture(scope="session")
def manifest(samples) -> dict:
    return json.loads((samples / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def settings():
    from clinic_scribe.config import Settings

    return Settings()  # 기본값 (사용자 설정과 무관)


@pytest.fixture(scope="session")
def speaker_encoder(settings):
    from clinic_scribe.speakers import SpeakerEncoder, SpeakerError

    try:
        return SpeakerEncoder(settings.speaker_model_path())
    except SpeakerError as exc:
        pytest.skip(f"화자 인식 모델이 없습니다 ({exc}). ./scripts/setup_macos.sh 를 먼저 실행하세요.")


@pytest.fixture(scope="session")
def whisper_runner(settings):
    from clinic_scribe.whisper import WhisperConfig, WhisperRunner

    runner = WhisperRunner(
        WhisperConfig(
            cli=settings.whisper_cli_path(),
            model=settings.whisper_model_path(),
            vad_model=settings.vad_model_path(),
            prompt=settings.whisper_prompt,
        )
    )
    if runner.problems(need_vad=True) or not runner.vad_tool:
        pytest.skip("whisper.cpp 또는 모델이 없습니다. ./scripts/setup_macos.sh 를 먼저 실행하세요.")
    return runner


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CLINIC_SCRIBE_TEST_CLAUDE") == "1":
        return
    skip = pytest.mark.skip(
        reason="실제 Claude 요약 테스트는 CLINIC_SCRIBE_TEST_CLAUDE=1 일 때만 실행합니다 (가상 전사문이 Claude 서비스로 전송됨)."
    )
    for item in items:
        if "claude" in item.keywords:
            item.add_marker(skip)
