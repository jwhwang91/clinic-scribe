from pathlib import Path

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
    prompt = "트리메부틴은 배가 아플 때만 드세요."
    assert find_misheard(prompt, "트림에 부티는 배가 아플 때만 드세요.") == [Finding("트림에 부티는", "트리메부틴은", prompt)]
    prompt = "축농증이 있어서 항생제를 일주일 정도 드셔야 해요."
    assert find_misheard(prompt, "충농증이 있어서 항생재를 일주일정도 드셔야 해요.") == [
        Finding("충농증이 있어서", "축농증이 있어서", prompt, "축농증이"),  # 짧은 말은 바로 뒤 낱말과 묶는다
        Finding("항생재를 일주일", "항생제를 일주일", prompt, "항생제를"),
    ]
    assert find_misheard("배를 한번 눌러 볼게요.", "배를 한번 눌러볼게요") == []  # 띄어쓰기·문장부호만 다름
    prompt = "고막이 부어 있어서 중이염이 맞네요."
    assert find_misheard(prompt, "고막이 부어있어서 중이염이 많네요.") == [  # 문장 끝이면 바로 앞 낱말과
        Finding("중이염이 많네요", "중이염이 맞네요", prompt, "맞네요")
    ]
    assert find_misheard("약은 하루 3번 드세요.", "약은 드세요.") == []  # 빠진 말은 고쳐 쓸 대상이 없다
    [finding] = find_misheard("역류성 식도염 같아요.", "역류성 10도염 같아요.")
    assert (finding.heard, finding.expected, finding.hint) == ("10도염 같아요", "식도염 같아요", "식도염")
    assert exact("에소메프라졸 40mg은 아침에", "에소메프라졸 40MG 은 아침에.")


def test_short_corrections_do_not_change_other_sentences():
    """흔한 낱말로 잘못 들린 말("가래"→"발에")은 그 자리에서만 고친다: "손발에 물집"은 그대로 둔다."""
    from clinic_scribe.diction import COMMON, DictionBook, apply_corrections
    from clinic_scribe.training import corrections_from

    prompt = "기침은 밤에 더 심해요? 가래 끓는 소리도 나요?"
    findings = find_misheard(prompt, "기침은 밤에 더 심해요? 발에 끓는 소리도 나요?")
    assert [(f.heard, f.expected, f.hint) for f in findings] == [("발에 끓는", "가래 끓는", "가래")]
    book = DictionBook(Path(), {})
    book.set_entry(COMMON, [f.hint for f in findings], corrections_from(findings))
    fixed = apply_corrections([TranscriptSegment(0, 2, "발에 끓는 소리가 나요"), TranscriptSegment(2, 4, "손발에 물집이 생겨요")], book)
    assert [seg.text for seg in fixed] == ["가래 끓는 소리가 나요", "손발에 물집이 생겨요"]


def test_split_by_prompts_uses_text_not_timestamps():
    prompts = ["어디가 불편해서 오셨어요?", "트리메부틴은 배가 아플 때만 드세요.", "약은 일주일 치 드릴게요."]
    heard = "어디가 불편해서 오셨어요? 트림에 부티는 배가 아플 때만 드세요. 음 약은 일주일치 드릴게요."
    assert split_by_prompts(prompts, heard) == [
        "어디가 불편해서 오셨어요?",
        "트림에 부티는 배가 아플 때만 드세요. 음",  # 더 들린 말은 바로 앞 문장에
        "약은 일주일치 드릴게요.",
    ]
    assert split_by_prompts(prompts, "") == ["", "", ""]
    # 띄어쓰기가 다른 말("확인해볼게요") 바로 뒤에 잘못 들린 말이 와도 제 문장으로 간다
    prompts = ["손발 저림은 신경전도 검사로 확인해 볼게요.", "아스피린은 매일 빠짐없이 드세요."]
    heard = "손발처림은 신경전도 검사로 확인해볼게요. 아스피리는 매일 빠짐없이 드세요."
    assert split_by_prompts(prompts, heard) == ["손발처림은 신경전도 검사로 확인해볼게요.", "아스피리는 매일 빠짐없이 드세요."]


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
    """가상 의사가 기본 문장 12개를 따라 읽은 녹음: 학습 뒤 정확히 받아 적는 문장이 줄지 않고, 목소리 특징값을 얻는다."""
    from clinic_scribe.audio import load_audio_file, to_whisper_pcm

    data, rate = load_audio_file(samples / "training_session.wav")
    pcm = to_whisper_pcm(data, rate)
    utterances = manifest["training_session.wav"]["utterances"]
    assert [u["text"] for u in utterances] == list(DEFAULT_PROMPTS)
    clips = [pcm[int((u["start"] - 0.3) * SAMPLE_RATE) : int((u["end"] + 0.5) * SAMPLE_RATE)] for u in utterances]

    def transcribe(pcm16, prompt=None):
        return whisper_runner.transcribe(pcm16, use_vad=True, params=settings.vad, prompt=prompt)

    result = analyze(list(DEFAULT_PROMPTS), clips, transcribe, lambda: speaker_encoder, settings.whisper_prompt)
    # 진료실에서 하는 흔한 말이라 처음부터 거의 다 맞게 받아 적는다. 잘못 들린 말이 있으면 학습 뒤 고쳐진다.
    assert result.correct_after >= result.correct_before and result.correct_after >= len(DEFAULT_PROMPTS) - 1
    assert result.correct_after > result.correct_before or not result.findings
    assert result.embeddings and result.speech_seconds > 30


def test_each_specialty_has_its_own_reading_sentences():
    from clinic_scribe.training import GENERAL, RENAMED_SPECIALTIES, SPECIALTIES, SPECIALTY_PROMPTS

    assert SPECIALTIES[0] == GENERAL and SPECIALTY_PROMPTS[GENERAL] == DEFAULT_PROMPTS
    assert SPECIALTIES[:4] == ("내과", "가정의학과", "소아청소년과", "정신건강의학과")  # 말을 많이 하는 과부터
    assert len(SPECIALTIES) >= 10 and RENAMED_SPECIALTIES["내과·가정의학과"] in SPECIALTIES
    seen = set()
    for specialty, prompts in SPECIALTY_PROMPTS.items():
        assert len(prompts) == 12 and len(set(prompts)) == 12, specialty
        assert all(p.strip() and len(p) <= 40 for p in prompts), specialty  # 한 문장을 몇 초 안에 따라 읽을 수 있게
        assert not (seen & set(prompts)), specialty  # 과마다 다른 문장
        seen |= set(prompts)
    # 의사가 진료실에서 실제로 하는 말: 약사들이 쓰는 성분명은 넣지 않는다 (의사 의견)
    ingredients = ("트리메부틴", "에소메프라졸", "알마게이트", "셀레콕시브", "오셀타미비르", "에스시탈로프람", "탐스로신")
    assert not any(name in p for prompts in SPECIALTY_PROMPTS.values() for p in prompts for name in ingredients)
    assert any("역류성 식도염" in p for p in DEFAULT_PROMPTS)
    assert build_prompts([], specialty="정형외과") == list(SPECIALTY_PROMPTS["정형외과"])
    assert build_prompts([], specialty="없는 과") == list(DEFAULT_PROMPTS)
    own = build_prompts(["트리메부틴"], specialty="안과")
    assert own[0] == "트리메부틴 처방해 드릴게요." and own[1:] == list(SPECIALTY_PROMPTS["안과"][:11])
