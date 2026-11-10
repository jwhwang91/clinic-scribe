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
    parse_vad_output,
    parse_whisper_json,
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
