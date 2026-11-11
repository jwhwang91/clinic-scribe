import numpy as np
import pytest

from clinic_scribe.config import VadParams
from clinic_scribe.speakers import SAMPLE_RATE
from clinic_scribe.training import (
    DEFAULT_PROMPTS,
    Finding,
    analyze,
    build_prompts,
    exact,
    find_misheard,
    join_clips,
    split_by_prompts,
)
from clinic_scribe.whisper import TranscriptionResult, TranscriptSegment


def test_find_misheard_ignores_spacing_and_trims_shared_parts():
    prompt = "알마게이트 현탁액은 속 쓰릴 때 드세요."
    assert find_misheard(prompt, "알마게이트 현타객은 속쓰릴 때 드세요.") == [Finding("현타객", "현탁액", prompt)]
    prompt = "트리메부틴은 배가 아플 때만 드세요."
    assert find_misheard(prompt, "트림에 부티는 배가 아플 때만 드세요.") == [Finding("트림에 부티는", "트리메부틴은", prompt)]
    assert find_misheard("배를 한번 눌러 볼게요.", "배를 한번 눌러볼게요") == []  # 띄어쓰기·문장부호만 다름
    assert find_misheard("혈액검사랑 소변검사 결과 보고", "혈액 검사랑 소변 검사 결가 보고") == [
        Finding("결가", "결과", "혈액검사랑 소변검사 결과 보고")
    ]
    assert find_misheard("약은 하루 3번 드세요.", "약은 드세요.") == []  # 빠진 말은 고쳐 쓸 대상이 없다
    assert exact("에소메프라졸 40mg은 아침에", "에소메프라졸 40MG 은 아침에.")


def test_split_by_prompts_uses_text_not_timestamps():
    prompts = ["어디가 불편해서 오셨어요?", "트리메부틴은 배가 아플 때만 드세요.", "약은 일주일 치 드릴게요."]
    heard = "어디가 불편해서 오셨어요? 트림에 부티는 배가 아플 때만 드세요. 음 약은 일주일치 드릴게요."
    assert split_by_prompts(prompts, heard) == [
        "어디가 불편해서 오셨어요?",
        "트림에 부티는 배가 아플 때만 드세요. 음",  # 더 들린 말은 바로 앞 문장에
        "약은 일주일치 드릴게요.",
    ]
    assert split_by_prompts(prompts, "") == ["", "", ""]


def test_build_prompts_puts_doctor_terms_first():
    assert build_prompts([]) == list(DEFAULT_PROMPTS)
    prompts = build_prompts(["레바미피드", "모사프리드", " "], count=5)
    assert prompts[:2] == ["레바미피드 처방해 드릴게요.", "오늘은 모사프리드 이야기를 해 볼게요."]
    assert len(prompts) == 5 and prompts[2:] == list(DEFAULT_PROMPTS[:3])
    assert len(build_prompts([f"약{i}" for i in range(30)])) == len(DEFAULT_PROMPTS)  # 용어 문장은 절반까지


def test_join_clips_keeps_each_prompt_span():
    clips = [np.ones(SAMPLE_RATE, np.int16), np.ones(SAMPLE_RATE // 2, np.int16)]
    audio, spans = join_clips(clips, gap_s=1.0)
    assert spans == [(1.0, 2.0), (3.0, 3.5)] and len(audio) == int(4.5 * SAMPLE_RATE)


class FakeEncoder:
    def embed(self, samples):
        return np.ones(4, np.float32) / 2


def test_analyze_learns_corrections_and_checks_them(monkeypatch):
    """학습 전에 잘못 들린 말을 찾고, 그 교정·용어를 넣어 다시 받아 적으면(학습 후) 맞는다. 힌트에 용어가 들어간다."""
    from clinic_scribe import training

    monkeypatch.setattr(training, "speech_only", lambda pcm: pcm.astype(np.float32))
    prompts = ["트리메부틴은 배가 아플 때만 드세요.", "세티리진은 자기 전에 드세요."]
    hints = []

    def transcribe(pcm16, prompt=None):
        hints.append(prompt)
        text = "트림에 부티는 배가 아플 때만 드세요. 세트리지는 자기 전에 드세요."
        return TranscriptionResult([TranscriptSegment(0.5, 6.0, text)], None, True, VadParams(), 7.0, 0.1)

    clips = [np.full(3 * SAMPLE_RATE, 100, np.int16), np.full(2 * SAMPLE_RATE, 100, np.int16)]
    result = analyze(prompts, clips, transcribe, lambda: FakeEncoder(), "CBC.")
    assert [(f.heard, f.expected) for f in result.findings] == [("트림에 부티는", "트리메부틴은"), ("세트리지는", "세티리진은")]
    assert (result.correct_before, result.correct_after) == (0, 2)
    assert hints[0] is None and hints[1] == "트리메부틴은, 세티리진은. CBC."
    assert result.embeddings and result.speech_seconds == pytest.approx(5 + 3 * 1.2)


@pytest.mark.models
def test_real_reading_session_improves_transcription(samples, manifest, whisper_runner, speaker_encoder, settings):
    """가상 의사가 기본 문장 12개를 따라 읽은 녹음: 학습 뒤 정확히 받아 적는 문장이 늘고 약 이름 교정을 찾는다."""
    from clinic_scribe.audio import load_audio_file, to_whisper_pcm

    data, rate = load_audio_file(samples / "training_session.wav")
    pcm = to_whisper_pcm(data, rate)
    utterances = manifest["training_session.wav"]["utterances"]
    assert [u["text"] for u in utterances] == list(DEFAULT_PROMPTS)
    clips = [pcm[int((u["start"] - 0.3) * SAMPLE_RATE) : int((u["end"] + 0.5) * SAMPLE_RATE)] for u in utterances]

    def transcribe(pcm16, prompt=None):
        return whisper_runner.transcribe(pcm16, use_vad=True, params=settings.vad, prompt=prompt)

    result = analyze(list(DEFAULT_PROMPTS), clips, transcribe, lambda: speaker_encoder, settings.whisper_prompt)
    assert result.correct_after > result.correct_before and result.correct_after >= len(DEFAULT_PROMPTS) - 1
    assert any(f.expected.startswith("트리메부틴") for f in result.findings)
    assert result.embeddings and result.speech_seconds > 30
