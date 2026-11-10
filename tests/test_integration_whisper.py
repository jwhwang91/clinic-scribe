"""whisper.cpp + Silero VAD 실제 실행 (가상 진료 대화 음성)."""

import glob
import os
import tempfile

import pytest

from clinic_scribe.audio import write_wav
from clinic_scribe.config import VadParams
from tests.helpers import load_pcm

pytestmark = pytest.mark.models

SPEECH_FILES = ["mock_consult_short.wav", "mock_consult.wav", "long_pause.wav", "injection.wav"]


def regions_for(runner, pcm, params):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "in.wav")
        write_wav(wav, pcm)
        return runner.detect_regions(wav, params)


@pytest.mark.parametrize("name", SPEECH_FILES)
def test_default_vad_regions_cover_every_utterance(whisper_runner, settings, samples, manifest, name):
    """짧은 대답('네.' 0.23초)과 문장 끝이 기본 VAD 설정에서 잘리지 않는다 (실제 발화 경계와 비교)."""
    regions = regions_for(whisper_runner, load_pcm(samples / name), settings.vad)
    for utt in manifest[name]["utterances"]:
        assert any(r.start <= utt["start"] + 0.02 and r.end >= utt["end"] - 0.02 for r in regions), (
            f"'{utt['text']}' ({utt['start']:.2f}–{utt['end']:.2f}) 가 구간 {[(r.start, r.end) for r in regions]} 에 다 들어가지 않음"
        )


def test_whisper_cpp_defaults_would_clip(whisper_runner, samples, manifest):
    """비교용: whisper.cpp 기본값(무음 100ms, 여유 30ms)은 문장을 쪼개고 끝을 자른다. 기본값을 바꾼 이유."""
    params = VadParams(min_silence_ms=100, speech_pad_ms=30, min_speech_ms=250)
    regions = regions_for(whisper_runner, load_pcm(samples / "mock_consult.wav"), params)
    utts = manifest["mock_consult.wav"]["utterances"]
    clipped = [u for u in utts if not any(r.start <= u["start"] + 0.02 and r.end >= u["end"] - 0.02 for r in regions)]
    assert clipped


def test_long_silence_splits_short_pause_merges(whisper_runner, settings, samples, manifest):
    regions = regions_for(whisper_runner, load_pcm(samples / "long_pause.wav"), settings.vad)
    utts = manifest["long_pause.wav"]["utterances"]
    assert len(regions) == 2
    assert regions[0].start <= utts[0]["start"] and regions[0].end >= utts[1]["end"]  # 0.5초 쉼은 합쳐짐
    assert regions[1].start > utts[1]["end"] + 5 and regions[1].start <= utts[2]["start"]  # 9초 무음에서 분리


def test_silence_threshold_is_adjustable(whisper_runner, samples):
    pcm = load_pcm(samples / "mock_consult.wav")
    fine = regions_for(whisper_runner, pcm, VadParams(min_silence_ms=400))
    coarse = regions_for(whisper_runner, pcm, VadParams(min_silence_ms=1500))
    assert len(fine) > len(coarse) >= 1


def test_transcript_keeps_original_timeline(whisper_runner, settings, samples, manifest):
    result = whisper_runner.transcribe(load_pcm(samples / "long_pause.wav"), use_vad=True, params=settings.vad)
    doctor_start = manifest["long_pause.wav"]["utterances"][2]["start"]
    text = " ".join(s.text for s in result.segments)
    assert "사흘" in text and "식후" in text and "열은 없어요" in text and "네" in text
    doctor = next(s for s in result.segments if "경과" in s.text)
    assert abs(doctor.start - doctor_start) < 0.5  # 13.5초 근처 (무음 제거 후 시각이 아님)
    for s in result.segments:
        region = result.regions[s.region]
        assert region.start - 0.01 <= s.start and s.end <= result.regions[s.last_region].end + 0.01


def test_mock_consult_short_transcription(whisper_runner, settings, samples):
    result = whisper_runner.transcribe(load_pcm(samples / "mock_consult_short.wav"), use_vad=True, params=settings.vad)
    text = " ".join(s.text for s in result.segments)
    for phrase in ("사흘", "식후", "속이 쓰려요", "열은 없어요", "계속되면", "검사를 논의"):
        assert phrase in text


@pytest.mark.parametrize("name", ["silence.wav", "quiet_room.wav"])
def test_silence_only_file(whisper_runner, settings, samples, name):
    pcm = load_pcm(samples / name)
    with_vad = whisper_runner.transcribe(pcm, use_vad=True, params=settings.vad)
    assert with_vad.regions == [] and with_vad.segments == [] and with_vad.no_speech_detected
    without_vad = whisper_runner.transcribe(pcm, use_vad=False, params=settings.vad)
    # VAD 없이 돌리면 Whisper 가 무음에서 문장을 지어낼 수 있다. 그런 문장은 모두 음성 미감지로 표시된다.
    assert all(not s.speech_detected for s in without_vad.segments)


def test_temporary_audio_is_removed(whisper_runner, settings, samples):
    pattern = os.path.join(tempfile.gettempdir(), "clinic-scribe-*")
    before = set(glob.glob(pattern))
    whisper_runner.transcribe(load_pcm(samples / "mock_consult_short.wav"), use_vad=True, params=settings.vad)
    assert set(glob.glob(pattern)) <= before
