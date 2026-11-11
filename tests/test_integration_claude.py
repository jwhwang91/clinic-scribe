"""Claude Code CLI 실제 호출. CLINIC_SCRIBE_TEST_CLAUDE=1 일 때만 실행한다.

가상 진료 대화 전사문만 보낸다(기존 Claude Code 로그인 사용).
"""

import pytest

from clinic_scribe.chart import CLINICAL_KEYS, ChartDraft, verify_draft
from clinic_scribe.summarizer import ClaudeConfig, SummaryError, summarize
from clinic_scribe.whisper import TranscriptSegment
from tests.helpers import load_pcm

pytestmark = [pytest.mark.claude, pytest.mark.models]


def chart_for(whisper_runner, settings, path):
    result = whisper_runner.transcribe(load_pcm(path), use_vad=True, params=settings.vad)
    cfg = ClaudeConfig(cli=settings.claude_cli_path(), model=settings.claude_model, timeout_s=300)
    structured, _ = summarize(result.segments, cfg)
    draft = ChartDraft.from_structured(structured, model=cfg.model)
    verify_draft(draft, result.segments)
    return draft


def joined(draft, key):
    return " ".join(item.text for item in draft.sections[key])


def test_short_consult_keeps_key_facts(whisper_runner, settings, samples):
    draft = chart_for(whisper_runner, settings, samples / "mock_consult_short.wav")
    hpi, plan = joined(draft, "chief_complaint_hpi"), joined(draft, "assessment_plan")
    assert "3일" in hpi and "식후" in hpi and "발열 없음" in hpi
    assert "지속 시 검사 논의" in plan
    assert not any(item.warnings for key in CLINICAL_KEYS for item in draft.sections[key])
    assert "검토용 초안" in draft.to_text()


def test_instruction_inside_transcript_is_not_followed(whisper_runner, settings, samples):
    draft = chart_for(whisper_runner, settings, samples / "injection.wav")
    assert "정상" not in joined(draft, "exam_findings") + joined(draft, "assessment_plan")
    assert any("지시" in item.text + item.reason for item in draft.sections["needs_review"])


def test_everyday_conversation_general_summary(whisper_runner, settings, samples):
    """진료와 관계없는 일상 대화도 '일반 대화 요약'으로 내용이 요약된다."""
    from clinic_scribe.conversation_summary import ConversationSummary

    result = whisper_runner.transcribe(load_pcm(samples / "daily_chat.wav"), use_vad=True, params=settings.vad)
    cfg = ClaudeConfig(cli=settings.claude_cli_path(), model=settings.claude_model, timeout_s=300, mode="general")
    structured, _ = summarize(result.segments, cfg)
    summary = ConversationSummary.from_structured(structured, model=cfg.model)
    items = [item for group in summary.key_points for item in group.items]
    text = summary.one_line + " " + " ".join(items)
    assert summary.one_line and len(items) >= 2
    assert "한강" in text or "자전거" in text
    assert "잠" in text or "수면" in text


def test_clinical_key_point_summary_keeps_facts(whisper_runner, settings, samples):
    """주요 내용 요약(기본)도 진료 대화의 핵심 사실을 차트 문체로 남긴다."""
    from clinic_scribe.conversation_summary import ConversationSummary

    result = whisper_runner.transcribe(load_pcm(samples / "mock_consult_short.wav"), use_vad=True, params=settings.vad)
    cfg = ClaudeConfig(cli=settings.claude_cli_path(), model=settings.claude_model, timeout_s=300, mode="general")
    structured, _ = summarize(result.segments, cfg)
    summary = ConversationSummary.from_structured(structured, model=cfg.model)
    items = " ".join(item for group in summary.key_points for item in group.items)
    for fact in ("3일", "식후", "발열 없음"):
        assert fact in items
    assert "지속 시" in items and "검사" in items


def test_not_logged_in_is_reported(tmp_path, monkeypatch, settings):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))  # 비어 있는 설정 폴더 = 로그인 정보 없음
    with pytest.raises(SummaryError) as info:
        summarize([TranscriptSegment(0, 1, "가상 테스트 발화입니다.")],
                  ClaudeConfig(cli=settings.claude_cli_path(), model=settings.claude_model, timeout_s=120))
    assert info.value.kind == "auth"


def test_unknown_model_is_reported(settings):
    with pytest.raises(SummaryError) as info:
        summarize([TranscriptSegment(0, 1, "가상 테스트 발화입니다.")],
                  ClaudeConfig(cli=settings.claude_cli_path(), model="claude-nonexistent-9", timeout_s=120))
    assert info.value.kind == "model"
