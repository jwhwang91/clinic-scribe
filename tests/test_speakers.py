import os
import stat
import threading

import numpy as np
import pytest

from clinic_scribe.speakers import (
    MIN_ENROLL_S,
    PATIENT,
    SAMPLE_RATE,
    SpeakerError,
    VoiceBook,
    clean_name,
    enrollment_embeddings,
    label_segments,
    main_doctor,
    speech_only,
)
from clinic_scribe.whisper import TranscriptSegment


class FakeEncoder:
    """소리 값으로 목소리를 흉내 낸다: 구간을 채운 정수 값 k → k 번째 축 방향 특징값."""

    def embed(self, samples):
        vector = np.zeros(8, np.float32)
        vector[int(round(float(np.median(np.abs(samples))))) % 8] = 1.0
        return vector


def axis(k):
    vector = np.zeros(8, np.float32)
    vector[k] = 1.0
    return vector


def test_voice_book_enroll_identify_remove_and_file(tmp_path):
    book = VoiceBook(tmp_path / "voices.json")
    book.enroll("김원장", [axis(1), axis(1) * 2], seconds=25.0)
    book.enroll("이원장", [axis(2)], seconds=18.0)
    assert book.identify(axis(1), 0.5) == ("김원장", pytest.approx(1.0))
    name, score = book.identify(axis(3), 0.5)
    assert name is None and score < 0.5  # 등록되지 않은 목소리
    book.save()
    assert stat.S_IMODE(os.stat(book.path).st_mode) == 0o600  # 생체인식정보: 이 사용자만 읽는다
    loaded = VoiceBook.load(book.path)
    assert loaded.names() == ["김원장", "이원장"]
    loaded.remove("김원장")
    assert loaded.names() == ["이원장"]
    loaded.enroll("이원장", [axis(4)], seconds=12.0)  # 같은 이름은 새 녹음으로 바꾼다
    assert loaded.names() == ["이원장"] and loaded.identify(axis(4), 0.5)[0] == "이원장"
    assert VoiceBook.load(tmp_path / "없음.json").voices == []


def test_voice_book_rejects_bad_file_and_names(tmp_path):
    bad = tmp_path / "voices.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SpeakerError):
        VoiceBook.load(bad)
    for name in ("", "   ", PATIENT):
        with pytest.raises(SpeakerError):
            clean_name(name)
    assert clean_name("  <김>  원장  ") == "김 원장"
    assert len(clean_name("가" * 50)) == 20
    with pytest.raises(SpeakerError):
        VoiceBook(tmp_path / "v.json").enroll("김원장", [], seconds=0)


def test_label_segments_marks_doctor_patient_and_skips_short(tmp_path):
    pcm = np.zeros(10 * SAMPLE_RATE, np.int16)
    pcm[0 : 2 * SAMPLE_RATE] = 1  # 등록한 의사 목소리 흉내
    pcm[2 * SAMPLE_RATE : 5 * SAMPLE_RATE] = 2  # 등록하지 않은 목소리
    pcm[5 * SAMPLE_RATE : 6 * SAMPLE_RATE] = 1
    book = VoiceBook(tmp_path / "v.json")
    book.enroll("김원장", [axis(1)], seconds=20)
    segments = [
        TranscriptSegment(0.0, 2.0, "어디가 불편해서 오셨어요?"),
        TranscriptSegment(2.0, 5.0, "사흘 전부터 속이 쓰려요."),
        TranscriptSegment(5.0, 5.4, "네."),  # 너무 짧아 판단하지 않는다
        TranscriptSegment(6.0, 8.0, "감사합니다.", speech_detected=False),  # 환각 의심 줄은 판단하지 않는다
    ]
    labeled = label_segments(segments, pcm, FakeEncoder(), book, 0.5)
    assert [s.speaker for s in labeled] == ["김원장", PATIENT, None, None]
    assert [s.text for s in labeled] == [s.text for s in segments]
    assert main_doctor(labeled) == "김원장" and main_doctor(segments) is None
    cancel = threading.Event()
    cancel.set()
    assert label_segments(segments, pcm, FakeEncoder(), book, 0.5, cancel) is segments  # 취소되면 그대로 둔다


def test_speech_only_drops_silence():
    rng = np.random.default_rng(0)
    room = (rng.standard_normal(3 * SAMPLE_RATE) * 30).astype(np.int16)  # 조용한 방
    t = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    voice = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)  # 말소리 대신 4초 소리
    assert 3.6 <= len(speech_only(np.concatenate([room, voice, room]))) / SAMPLE_RATE <= 4.4
    assert len(speech_only(np.zeros(5 * SAMPLE_RATE, np.int16))) == 0  # 디지털 무음


@pytest.mark.models
def test_real_model_tells_mock_doctor_from_patient(samples, manifest, speaker_encoder, tmp_path):
    """가상 진료 음성: 실제 화자 인식 모델이 등록한 의사 역할 목소리와 환자 역할 목소리를 구분한다."""
    from clinic_scribe.audio import load_audio_file, to_whisper_pcm
    from clinic_scribe.config import Settings

    def pcm(name):
        data, rate = load_audio_file(samples / name)
        return to_whisper_pcm(data, rate)

    threshold = Settings().speaker_threshold
    utterances = manifest["mock_consult.wav"]["utterances"]
    segments = [TranscriptSegment(u["start"], u["end"], u["text"]) for u in utterances]
    consult = pcm("mock_consult.wav")

    book = VoiceBook(tmp_path / "voices.json")
    speech = speech_only(pcm("voice_enroll_doctor.wav"))
    assert len(speech) / SAMPLE_RATE >= MIN_ENROLL_S
    book.enroll("김원장", enrollment_embeddings(speaker_encoder, speech), len(speech) / SAMPLE_RATE)
    for utt, seg in zip(utterances, label_segments(segments, consult, speaker_encoder, book, threshold)):
        if utt["end"] - utt["start"] < 1.0:
            assert seg.speaker is None, utt["text"]  # "네." 같은 짧은 대답은 판단하지 않는다
        else:
            assert seg.speaker == ("김원장" if utt["role"] == "doctor" else PATIENT), utt["text"]

    # 다른 사람만 등록했다면 이 대화의 의사도 '환자'(등록되지 않은 목소리)로 나와야 한다
    other = VoiceBook(tmp_path / "other.json")
    speech = speech_only(pcm("voice_enroll_other.wav"))
    other.enroll("박원장", enrollment_embeddings(speaker_encoder, speech), len(speech) / SAMPLE_RATE)
    assert all(s.speaker in (PATIENT, None) for s in label_segments(segments, consult, speaker_encoder, other, threshold))
