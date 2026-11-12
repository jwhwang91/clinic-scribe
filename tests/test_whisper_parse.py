import json

import pytest

from clinic_scribe.config import VadParams
from clinic_scribe.whisper import (
    SpeechRegion,
    TranscriptionResult,
    TranscriptSegment,
    WhisperError,
    assign_regions,
    format_time,
    group_words,
    merge_phrases,
    parse_vad_output,
    parse_whisper_json,
    phrase_params,
    vad_args,
)

# whisper-vad-speech-segments 1.9.5 실제 출력 형식 (센티초)
VAD_STDOUT = "\nDetected 2 speech segments:\nSpeech segment 0: start = 66.00, end = 474.00\nSpeech segment 1: start = 1330.00, end = 1879.00\n\n"


def test_parse_vad_output_centiseconds_to_seconds():
    regions = parse_vad_output(VAD_STDOUT)
    assert [(r.start, r.end) for r in regions] == [(0.66, 4.74), (13.30, 18.79)]


def test_parse_vad_output_no_speech():
    assert parse_vad_output("\nDetected 0 speech segments:\n\n") == []


def _token(text, p, start=0, end=0):
    return {"text": text, "p": p, "offsets": {"from": start, "to": end}, "id": 1}


def test_parse_whisper_json_full_output():
    data = {
        "transcription": [
            {
                "offsets": {"from": 660, "to": 4740},
                "text": " omeprazole 20mg을 드세요.",
                "tokens": [_token("[_BEG_]", 0.8), _token(" omeprazole", 0.9), _token(" 20", 0.31), _token("mg", 0.9),
                           _token("을", 0.95), _token(" 드세요.", 0.97), _token("[_TT_210]", 0.5)],
            },
            {"offsets": {"from": 5000, "to": 5200}, "text": "   ", "tokens": []},
        ]
    }
    segments = parse_whisper_json(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    assert len(segments) == 1  # 빈 문장은 버린다
    seg = segments[0]
    assert (seg.start, seg.end, seg.text) == (0.66, 4.74, "omeprazole 20mg을 드세요.")
    assert seg.uncertain_terms == ["20mg을"]
    assert 0.7 < seg.confidence < 0.9  # 특수 토큰 제외 평균


def test_parse_whisper_json_invalid():
    with pytest.raises(WhisperError):
        parse_whisper_json(b"{not json")


def seg(start, end, text="x"):
    return TranscriptSegment(start=start, end=end, text=text)


def test_vad_junction_times_snap_to_speech_regions():
    # whisper-cli --vad 실제 출력(long_pause.wav): 이음매 시각 6.45 초가 무음 구간에 찍힌다.
    regions = [SpeechRegion(0.66, 4.74), SpeechRegion(13.30, 18.79)]
    segments = [seg(0.66, 6.45), seg(6.45, 18.04), seg(18.04, 18.79)]
    assign_regions(segments, regions, vad_enabled=True)
    assert [(s.region, s.start, s.end) for s in segments] == [(0, 0.66, 4.74), (1, 13.30, 18.04), (1, 18.04, 18.79)]


def test_vad_segment_goes_to_largest_overlap():
    regions = [SpeechRegion(0.6, 3.24), SpeechRegion(3.48, 7.34)]
    segments = [seg(3.2, 7.3)]
    assign_regions(segments, regions, vad_enabled=True)
    assert (segments[0].region, segments[0].start, segments[0].end) == (1, 3.48, 7.3)


def test_segment_spanning_two_regions_keeps_both():
    regions = [SpeechRegion(0.6, 4.0), SpeechRegion(4.4, 9.2), SpeechRegion(9.5, 13.8)]
    segments = [seg(0.6, 9.2, "두 발화가 한 문장"), seg(9.2, 13.8, "의사")]
    assign_regions(segments, regions, vad_enabled=True)
    assert (segments[0].region, segments[0].last_region, segments[0].end) == (0, 1, 9.2)
    result = TranscriptionResult(segments, regions, True, VadParams(), 14.0, 1.0)
    assert result.region_text(0) == "두 발화가 한 문장"
    assert "이어진" in result.region_text(1)
    assert result.region_text(2) == "의사"


def test_without_vad_text_outside_speech_is_flagged():
    regions = [SpeechRegion(5.0, 6.0)]
    segments = [seg(0.0, 2.0, "시청해 주셔서 감사합니다."), seg(4.8, 6.2, "네.")]
    assign_regions(segments, regions, vad_enabled=False)
    assert segments[0].speech_detected is False and segments[0].region is None
    assert segments[1].speech_detected is True and segments[1].region == 0


def test_without_vad_and_no_speech_everything_is_flagged():
    segments = [seg(0.0, 29.98, "자막 제공 및 광고를 포함하고 있습니다.")]
    assign_regions(segments, [], vad_enabled=False)
    assert segments[0].speech_detected is False


def test_vad_args():
    args = vad_args(VadParams(min_silence_ms=600, speech_pad_ms=200, min_speech_ms=150))
    assert args[args.index("-vsd") + 1] == "600"
    assert args[args.index("-vp") + 1] == "200"
    assert args[args.index("-vspd") + 1] == "150"
    assert "-vmsd" not in args


def test_format_time():
    assert format_time(0) == "00:00.0"
    assert format_time(13.3) == "00:13.3"
    assert format_time(59.96) == "01:00.0"
    assert format_time(3725.4) == "1:02:05.4"


def word(start, end, text, confidence=0.9, uncertain=None):
    return TranscriptSegment(start, end, text, confidence=confidence, uncertain_terms=uncertain or [])


def test_group_words_into_phrases_by_short_pauses():
    """단어 시각은 이음매에서 쉼 쪽으로 밀린다(실제 whisper.cpp 출력 형태). 겹침이 큰 말 덩어리에 붙이고 시각은 덩어리 경계로."""
    pauses = [SpeechRegion(0.75, 3.12), SpeechRegion(3.63, 7.19), SpeechRegion(7.95, 11.19)]
    words = [
        word(0.60, 0.80, ""),  # 빈 단어는 버린다
        word(0.80, 1.18, "어디가"), word(1.18, 1.98, "불편해서"), word(1.98, 3.38, "오셨어요?"),
        word(3.38, 3.89, "사흘", 0.5, ["3"]), word(3.89, 4.47, "전부터"), word(6.61, 7.34, "없어요."),
        word(7.34, 8.45, "토하거나"), word(12.0, 12.4, "네."),  # 어느 덩어리와도 겹치지 않으면 가장 가까운 덩어리로
    ]
    phrases = group_words(words, pauses)
    assert [p.text for p in phrases] == ["어디가 불편해서 오셨어요?", "사흘 전부터 없어요.", "토하거나 네."]
    assert [(p.start, p.end) for p in phrases] == [(0.75, 3.12), (3.63, 7.19), (7.95, 11.19)]
    assert phrases[1].confidence == pytest.approx((0.5 + 0.9 + 0.9) / 3) and phrases[1].uncertain_terms == ["3"]
    assert [w.text for w in phrases[2].words] == ["토하거나", "네."]  # 덩어리 안 낱말과 시각도 남긴다 (말한 사람이 바뀌면 나눌 때)
    assert phrases[2].words[0].start == 7.34
    assert group_words(words, []) == [w for w in words if w.text]


def test_merge_phrases_breaks_at_sentence_end_long_pause_and_speaker_change():
    def phrase(start, end, text, speaker=None):
        return TranscriptSegment(start, end, text, confidence=0.8, speaker=speaker)

    phrases = [
        phrase(0.0, 2.0, "어디가 불편해서 오셨어요?"),
        phrase(2.5, 4.0, "사흘 전부터 속이"),
        phrase(4.3, 5.0, "쓰려요"),  # 문장이 안 끝났고 쉼이 짧으면 앞 줄에 잇는다
        phrase(6.5, 7.0, "열은 없어요"),  # 1초 이상 쉬면 새 줄
        phrase(7.2, 9.0, "약은 하루 세 번", "김원장"),  # 말한 사람이 바뀌면 새 줄 (문장부호가 없어도)
        phrase(9.2, 9.6, "식후에"),  # 누구인지 모르는 짧은 덩어리는 앞 줄에
        phrase(9.8, 11.0, "드세요.", "김원장"),
    ]
    lines = merge_phrases(phrases)
    assert [(line.text, line.speaker) for line in lines] == [
        ("어디가 불편해서 오셨어요?", None),
        ("사흘 전부터 속이 쓰려요", None),
        ("열은 없어요", None),
        ("약은 하루 세 번 식후에 드세요.", "김원장"),
    ]
    assert (lines[1].start, lines[1].end) == (2.5, 5.0)
    assert all(line.words == [] for line in merge_phrases(group_words([word(0.8, 1.2, "네.")], [SpeechRegion(0.7, 1.3)])))
    assert phrases[1].text == "사흘 전부터 속이"  # 원래 덩어리는 바꾸지 않는다
    two = merge_phrases([phrase(0, 2, "네 알겠습니다", "환자"), phrase(2.3, 4, "약은 식후에 드세요", "김원장")])
    assert [line.speaker for line in two] == ["환자", "김원장"]


def test_phrase_params_split_at_shorter_pauses_only():
    user = VadParams(threshold=0.6, min_silence_ms=600, speech_pad_ms=200, min_speech_ms=150)
    fine = phrase_params(user)
    assert (fine.threshold, fine.min_silence_ms, fine.speech_pad_ms, fine.min_speech_ms) == (0.6, 300, 50, 100)
    tight = VadParams(min_silence_ms=200, speech_pad_ms=30, min_speech_ms=80)
    assert phrase_params(tight) == tight  # 이미 더 촘촘하면 그대로
