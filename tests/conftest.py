import faulthandler
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# 테스트 중에 만든 앱 창이 GitHub 에 새 버전을 묻거나 repo 를 바꾸지 않게 (자동 업데이트 테스트는 이 값을 지운다)
os.environ.setdefault("CLINIC_SCRIBE_NO_UPDATE", "1")
# 테스트 중에 만든 앱 창이 실제 녹음 버튼(전역 단축키)을 등록하지 않게 (단축키 테스트는 직접 등록한다)
os.environ.setdefault("CLINIC_SCRIBE_NO_HOTKEY", "1")

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "samples"
# 테스트가 쓰는 가상 음성. 하나라도 없으면 다시 만든다 (생성기에 새 음성을 추가했을 때).
REQUIRED_SAMPLES = (
    "manifest.json", "mock_consult_short.wav", "mock_consult.wav", "long_pause.wav", "injection.wav", "daily_chat.wav",
    "two_patients.wav", "voice_enroll_doctor.wav", "voice_enroll_other.wav", "doctor_drugs.wav", "training_session.wav",
    "interruption.wav", "silence.wav", "quiet_room.wav",
)


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """테스트가 사용자 설정 파일(~/Library/Application Support/clinic-scribe)을 건드리지 않게 한다."""
    monkeypatch.setenv("CLINIC_SCRIBE_SETTINGS", str(tmp_path / "settings.json"))


@pytest.fixture(scope="session")
def samples() -> Path:
    """가상 진료 대화 음성. 없으면 scripts/make_test_audio.py 로 만든다 (macOS `say` 필요)."""
    if any(not (SAMPLES / name).is_file() for name in REQUIRED_SAMPLES):
        if shutil.which("say") is None:
            pytest.skip("가상 음성(samples/)은 macOS 의 say 로 만듭니다. Mac 에서 만든 samples 폴더를 복사하면 이 OS 에서도 실행됩니다.")
        made = subprocess.run([sys.executable, str(REPO / "scripts" / "make_test_audio.py"), str(SAMPLES)],
                              capture_output=True, text=True)
        if made.returncode != 0:
            pytest.skip("가상 음성을 만들지 못했습니다 (macOS 한국어 음성 Yuna 필요): "
                        + (made.stderr.strip().splitlines() or ["?"])[-1])
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
        pytest.skip(f"화자 인식 모델이 없습니다 ({exc}). 설치 스크립트를 먼저 실행하세요.")


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
        pytest.skip("whisper.cpp 또는 모델이 없습니다. 설치 스크립트(setup_macos.sh · setup_linux.sh · setup_windows.bat)를 먼저 실행하세요.")
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


def pytest_sessionfinish(session, exitstatus):
    """Qt 화면 테스트를 했으면 남은 창과 QApplication 을 지금 지운다 (앱을 닫을 때와 같게, clinic_scribe.app.close_qt).

    Python 이 닫히는 도중에 Qt 가 지우게 두면 Windows·Linux 에서 가끔 비정상 종료했다.
    """
    if "PySide6.QtWidgets" in sys.modules:
        from clinic_scribe.app import close_qt

        close_qt()


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """CI: 테스트가 다 끝났는데 Python 이 닫히지 않으면(Linux 에서 1시간 멈춘 일) 60초 뒤 모든 스레드가 멈춘 곳을 남기고 끝낸다."""
    if os.environ.get("CI"):
        faulthandler.dump_traceback_later(60, exit=True)
