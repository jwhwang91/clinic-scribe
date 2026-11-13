import os
import stat
import sys
import threading

import numpy as np
import pytest

from clinic_scribe.speakers import (
    MIN_ENROLL_S,
    MIN_SEGMENT_S,
    PATIENT,
    SAMPLE_RATE,
    SpeakerError,
    VoiceBook,
    clean_name,
    enrollment_embeddings,
    label_segments,
    label_turns,
    main_doctor,
    speech_only,
    split_turns,
)
from clinic_scribe.whisper import TranscriptSegment, merge_phrases, phrase_from_words


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


class MixEncoder:
    """구간에 섞인 목소리 비율대로 특징값을 만든다: 값 k 인 표본의 비율 → k 번째 축 (0 은 무음)."""

    def embed(self, samples):
        counts = np.bincount(np.abs(samples.astype(np.int64)) % 8, minlength=8).astype(np.float32)
        counts[0] = 0.0
        return counts if counts.any() else axis(0)


class PureEncoder:
    """한 사람 소리가 80% 이상인 구간만 그 사람으로 들리고, 섞인 구간은 등록되지 않은 목소리(3)로 들린다."""

    def embed(self, samples):
        values = np.abs(samples.astype(np.int64))
        values = values[values > 0]
        counts = np.bincount(values % 8, minlength=8)
        k = int(np.argmax(counts))
        return axis(k if counts[k] >= 0.8 * len(values) else 3)


def conversation(parts, total_s=8.0):
    """[(목소리 값, [(낱말, 길이초), ...]), ...] → 소리(목소리 값으로 채움)와 낱말이 든 말 덩어리 하나."""
    pcm = np.zeros(int(total_s * SAMPLE_RATE), np.int16)
    words, t = [], 0.5
    for value, said in parts:
        for text, length in said:
            pcm[int(t * SAMPLE_RATE) : int((t + length) * SAMPLE_RATE)] = value
            words.append(TranscriptSegment(t, t + length, text, confidence=0.9))
            t += length
    return pcm, phrase_from_words(words, 0.5, t)


def test_voice_book_enroll_identify_remove_and_file(tmp_path):
    book = VoiceBook(tmp_path / "voices.json")
    book.enroll("김원장", [axis(1), axis(1) * 2], seconds=25.0)
    book.enroll("이원장", [axis(2)], seconds=18.0)
    assert book.identify(axis(1), 0.5) == ("김원장", pytest.approx(1.0))
    name, score = book.identify(axis(3), 0.5)
    assert name is None and score < 0.5  # 등록되지 않은 목소리
    book.save()
    if sys.platform != "win32":  # Windows 는 사용자 폴더(%LOCALAPPDATA%) 권한으로 보호된다
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


def test_split_turns_cuts_at_the_sentence_end_when_word_timing_is_off(tmp_path):
    """Whisper 낱말 시각이 0.2초쯤 어긋난 경우(Windows CPU): 목소리로는 짧은 첫 낱말 '등'이 앞사람 쪽에 들어가도,
    문장이 끝난 자리 0.4초 안이면 문장 끝에서 나눈다 ("아파서요. 등 / 어느 쪽이요?" 가 아니라 "/ 등 어느 쪽이요?")."""
    book = VoiceBook(tmp_path / "v.json")
    book.enroll("김원장", [axis(1)], seconds=20)
    said = [("사흘", 0.5), ("전부터", 0.5), ("밥만", 0.5), ("먹으면", 0.5), ("속이", 0.5), ("쓰려요.", 0.5),  # 0.5~3.5
            ("등", 0.3), ("어느", 0.5), ("쪽이요?", 0.5),  # 3.5~4.8: 낱말 시각은 3.5 부터인데 의사 목소리는 3.95 부터
            ("오른쪽", 0.5), ("아래쪽", 0.5), ("이요.", 0.5)]
    words, t = [], 0.5
    for text, length in said:
        words.append(TranscriptSegment(t, t + length, text, confidence=0.9))
        t += length
    pcm = np.zeros(int(8 * SAMPLE_RATE), np.int16)
    for value, start, end in ((2, 0.5, 3.95), (1, 3.95, 4.8), (2, 4.8, t)):
        pcm[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = value
    parts = split_turns(phrase_from_words(words, 0.5, t), pcm, MixEncoder(), book, 0.75)
    assert [(p.speaker, p.text) for p in parts] == [
        (PATIENT, "사흘 전부터 밥만 먹으면 속이 쓰려요."), ("김원장", "등 어느 쪽이요?"), (PATIENT, "오른쪽 아래쪽 이요."),
    ]


def test_split_turns_cuts_where_doctor_cuts_in_without_pause(tmp_path):
    """쉼 없이 끼어든 의사 질문: 말 덩어리 하나를 말한 사람이 바뀌는 낱말 경계에서 셋으로 나눈다."""
    book = VoiceBook(tmp_path / "v.json")
    book.enroll("김원장", [axis(1)], seconds=20)
    patient = [("사흘", 0.5), ("전부터", 0.5), ("밥만", 0.5), ("먹으면", 0.5), ("속이", 0.5), ("쓰려요.", 0.5)]
    pcm, phrase = conversation([(2, patient), (1, [("등", 0.5), ("어느", 0.5), ("쪽이요?", 0.5)]),
                                (2, [("오른쪽", 0.5), ("아래쪽", 0.5), ("이요.", 0.5)])])
    parts = split_turns(phrase, pcm, MixEncoder(), book, 0.75)
    assert [(p.start, p.end, p.speaker, p.text) for p in parts] == [
        (0.5, 3.5, PATIENT, "사흘 전부터 밥만 먹으면 속이 쓰려요."),
        (3.5, 5.0, "김원장", "등 어느 쪽이요?"),
        (5.0, 6.5, PATIENT, "오른쪽 아래쪽 이요."),
    ]
    assert [len(p.words) for p in parts] == [6, 3, 3] and parts[1].confidence == pytest.approx(0.9)

    # label_turns: 덩어리마다 말한 사람을 붙이고 섞인 덩어리는 나눈다 → 줄로 이으면 사람마다 한 줄
    lines = merge_phrases(label_turns([phrase], pcm, MixEncoder(), book, 0.75))
    assert [(line.speaker, line.text) for line in lines] == [(p.speaker, p.text) for p in parts]
    assert all(line.words == [] for line in lines)
    cancel = threading.Event()
    cancel.set()
    assert label_turns([phrase], pcm, MixEncoder(), book, 0.75, cancel) == [phrase]


def test_split_turns_keeps_one_speaker_and_too_short_interjections(tmp_path):
    book = VoiceBook(tmp_path / "v.json")
    book.enroll("김원장", [axis(1)], seconds=20)
    pcm, alone = conversation([(2, [("사흘", 0.8), ("전부터", 0.8), ("속이", 0.8), ("쓰려요.", 0.8)])])
    assert split_turns(alone, pcm, MixEncoder(), book, 0.75) == [alone]
    # 0.4초 "네?" 는 목소리로 가리기 어려워 나누지 않는다
    pcm, short = conversation([(2, [("화장실을", 1.0), ("세", 0.5), ("번", 1.0)]), (1, [("네?", 0.4)]),
                               (2, [("갔어요.", 1.0), ("물처럼", 1.0)])])
    assert split_turns(short, pcm, MixEncoder(), book, 0.75) == [short]
    assert split_turns(short, pcm, MixEncoder(), VoiceBook(tmp_path / "none.json"), 0.75) == [short]  # 등록 목소리 없음
    no_words = TranscriptSegment(short.start, short.end, short.text)
    assert split_turns(no_words, pcm, MixEncoder(), book, 0.75) == [no_words]


def test_split_turns_finds_short_question_that_windows_miss(tmp_path):
    """0.7초짜리 "설사요?": 1초 창마다 환자 말이 섞여 놓치지만, '?' 로 끝나는 짧은 구간을 따로 비교해 찾는다."""
    book = VoiceBook(tmp_path / "v.json")
    book.enroll("김원장", [axis(1)], seconds=20)
    pcm, phrase = conversation([(2, [("화장실을", 1.0), ("세", 0.5), ("번이나", 1.0), ("갔는데", 1.0)]),
                                (1, [("설사요?", 0.7)]), (2, [("네", 0.4), ("물처럼", 0.8), ("나왔어요.", 1.0)])])
    parts = split_turns(phrase, pcm, PureEncoder(), book, 0.75)
    assert [(p.speaker, p.text) for p in parts] == [
        (PATIENT, "화장실을 세 번이나 갔는데"), ("김원장", "설사요?"), (PATIENT, "네 물처럼 나왔어요."),
    ]
    assert (parts[1].start, parts[1].end) == pytest.approx((4.0, 4.7))


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


@pytest.mark.models
def test_lines_split_where_speaker_changes_even_without_punctuation(samples, manifest, whisper_runner, speaker_encoder,
                                                                     settings, tmp_path):
    """Whisper 가 문장부호 없이 받아 적어도, 의사 목소리를 등록했으면 말한 사람이 바뀌는 곳에서 줄을 나눈다."""
    from clinic_scribe.audio import load_audio_file, to_whisper_pcm
    from clinic_scribe.whisper import merge_phrases

    def pcm(name):
        data, rate = load_audio_file(samples / name)
        return to_whisper_pcm(data, rate)

    book = VoiceBook(tmp_path / "voices.json")
    speech = speech_only(pcm("voice_enroll_doctor.wav"))
    book.enroll("김원장", enrollment_embeddings(speaker_encoder, speech), len(speech) / SAMPLE_RATE)
    audio = pcm("doctor_drugs.wav")
    # 이 힌트로는 문장부호가 빠져 말 덩어리만으로는 의사·환자 말이 한 줄로 이어진다
    result = whisper_runner.transcribe(audio, use_vad=True, params=settings.vad,
                                       prompt=f"트리메부틴, 알마게이트, 세티리진, 트리메부틴은, 현탁액, {settings.whisper_prompt}")
    utterances = manifest["doctor_drugs.wav"]["utterances"]

    def roles(seg):
        return {u["role"] for u in utterances if min(seg.end, u["end"]) - max(seg.start, u["start"]) > 0.15}

    # 이 힌트로 Metal(macOS) 빌드는 문장부호를 빼 의사·환자 말이 한 줄로 섞인다. CPU 빌드(Windows·Linux)는 처음부터 나뉘기도 한다.
    # 어느 쪽이든 아래처럼 말한 사람을 붙여 다시 이은 줄은 한 사람 말이어야 한다.
    lines = merge_phrases(label_turns(result.phrases, audio, speaker_encoder, book, settings.speaker_threshold))
    assert all(len(roles(line)) == 1 for line in lines), [(line.speaker, line.text) for line in lines]
    for line in lines:
        expected = "김원장" if roles(line) == {"doctor"} else PATIENT
        # CPU 빌드는 "다른 약은요?"(0.97초)를 따로 나눠 받아 적는다: 1초보다 짧은 말은 판단하지 않는다
        assert line.speaker == expected or (line.speaker is None and line.end - line.start < MIN_SEGMENT_S), line.text


@pytest.mark.models
def test_doctor_cutting_in_without_pause_gets_own_line(samples, manifest, whisper_runner, speaker_encoder, settings,
                                                        tmp_path):
    """쉼 없이 끼어든 말(겹쳐 시작한 말, 0.8초짜리 "설사요?" 포함)도 말한 사람이 바뀌는 곳에서 줄이 나뉘고 맞는 사람이 붙는다."""
    from clinic_scribe.audio import load_audio_file, to_whisper_pcm

    def pcm(name):
        data, rate = load_audio_file(samples / name)
        return to_whisper_pcm(data, rate)

    book = VoiceBook(tmp_path / "voices.json")
    speech = speech_only(pcm("voice_enroll_doctor.wav"))
    book.enroll("김원장", enrollment_embeddings(speaker_encoder, speech), len(speech) / SAMPLE_RATE)
    audio = pcm("interruption.wav")
    result = whisper_runner.transcribe(audio, use_vad=True, params=settings.vad)
    lines = merge_phrases(label_turns(result.phrases, audio, speaker_encoder, book, settings.speaker_threshold))
    utterances = manifest["interruption.wav"]["utterances"]

    def overlap(line, u):
        return max(0.0, min(line.end, u["end"]) - max(line.start, u["start"]))

    shown = [(line.speaker, line.text) for line in lines]
    for line in lines:
        by_role: dict[str, float] = {}
        for u in utterances:
            by_role[u["role"]] = by_role.get(u["role"], 0.0) + overlap(line, u)
        role = max(by_role, key=by_role.__getitem__)
        assert sum(v for r, v in by_role.items() if r != role) <= 0.35, shown  # 겹쳐 말한 0.25초 정도만 섞일 수 있다
        expected = "김원장" if role == "doctor" else PATIENT
        assert line.speaker == expected or (line.speaker is None and line.end - line.start < MIN_SEGMENT_S), shown
    for u in utterances:  # 의사 말은 짧은 질문까지 모두 의사 줄로 나온다
        if u["role"] == "doctor":
            mine = sum(overlap(line, u) for line in lines if line.speaker == "김원장")
            assert mine >= 0.7 * (u["end"] - u["start"]), (u["text"], shown)
    # 낱말 하나도 앞사람 줄로 새지 않는다: 각 발화의 줄은 그 발화가 시작한 곳에서 시작한다
    # ("아파서요. 등 / 어느 쪽이요?"처럼 끼어든 질문의 첫 낱말 "등"(0.2초)이 앞 줄에 붙으면 0.18초 늦게 시작한다).
    # 두 사람이 겹쳐 말한 부분은 마이크 하나로 가를 수 없어 그만큼은 봐준다
    # (Windows CI: "노랬어요. 열은 안 / 났어요?" — 0.25초 겹친 "열은 안"이 앞 줄로. Mac 은 GPU·CPU 모두 낱말까지 맞음)
    for previous, u in zip([None] + utterances, utterances):
        name = "김원장" if u["role"] == "doctor" else PATIENT
        line = max((line for line in lines if line.speaker == name), key=lambda line: overlap(line, u))
        overlapped = max(0.0, previous["end"] - u["start"]) if previous else 0.0
        assert line.start <= u["start"] + 0.15 + overlapped, (u["text"], shown)


def test_rename_keeps_voiceprint_and_refuses_taken_names(tmp_path):
    book = VoiceBook(tmp_path / "voices.json")
    book.enroll("황재", [np.ones(192, np.float32)], 30.0)
    book.enroll("김원장", [np.arange(192, dtype=np.float32) + 1.0], 20.0)
    working_copy = book.copy()  # 전사 중인 작업 스레드가 쓰는 사본
    assert book.rename("황재", "  황재웅 ") == "황재웅"
    assert book.names() == ["황재웅", "김원장"] and working_copy.names() == ["황재", "김원장"]
    assert book.voices[0].embedding == working_copy.voices[0].embedding  # 목소리 특징값은 그대로
    with pytest.raises(SpeakerError):
        book.rename("황재웅", "김원장")  # 이미 있는 이름
    with pytest.raises(SpeakerError):
        book.rename("황재웅", "환자")
