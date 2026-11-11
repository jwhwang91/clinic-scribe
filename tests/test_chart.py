import pytest

from clinic_scribe.chart import REVIEW_BANNER, ChartDraft, transcript_numbers, verify_draft
from clinic_scribe.whisper import TranscriptSegment
from tests.helpers import CHART

SEGMENTS = [
    TranscriptSegment(0.7, 4.5, "사흘 전부터 식후에 속이 쓰려요. 열은 없어요."),
    TranscriptSegment(5.0, 9.6, "오늘은 경과를 보고 계속되면 검사를 논의하죠."),
]


def test_from_structured_and_text_has_review_banner_and_sections():
    draft = ChartDraft.from_structured(CHART, model="m", based_on="전사 #1")
    text = draft.to_text()
    assert text.startswith(REVIEW_BANNER)
    assert "검토용 초안" in text.splitlines()[-1]
    for title in ("주호소·현병력", "대화에서 확인된 진찰·검사", "의사가 언급한 평가·계획", "확인 필요"):
        assert f"■ {title}" in text
    assert "대화에서 확인된 내용 없음" in text  # 빈 칸은 비워 두되 표시한다
    assert "(환자)" in text and "(의사)" in text
    assert '근거: "사흘 전부터' in text
    assert "근거:" not in draft.to_text(include_evidence=False)


def test_empty_chart_explains_non_clinical_conversation():
    empty = {"chief_complaint_hpi": [], "exam_findings": [], "assessment_plan": [], "needs_review": []}
    assert "진료 관련 내용을 찾지 못했습니다" in ChartDraft.from_structured(empty).to_html()
    assert "진료 관련 내용을 찾지 못했습니다" not in ChartDraft.from_structured(CHART).to_html()


@pytest.mark.parametrize(
    "bad",
    [None, [], {"chief_complaint_hpi": []}, {**CHART, "exam_findings": "없음"}, {**CHART, "needs_review": [{"text": ""}]}],
)
def test_from_structured_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        ChartDraft.from_structured(bad)


def test_verify_accepts_supported_items():
    draft = ChartDraft.from_structured(CHART)
    verify_draft(draft, SEGMENTS)
    assert all(not item.warnings for items in draft.sections.values() for item in items)
    assert draft.sections["needs_review"] == []


def test_verify_flags_unsupported_evidence_and_numbers():
    chart = {
        **CHART,
        "assessment_plan": [
            {"text": "omeprazole 20mg 처방", "source": "의사", "evidence": "오메프라졸 드릴게요"},
            {"text": "지속 시 검사 논의", "source": "의사", "evidence": ""},
        ],
    }
    draft = ChartDraft.from_structured(chart)
    verify_draft(draft, SEGMENTS)
    first, second = draft.sections["assessment_plan"]
    assert "근거 인용을 전사문에서 찾지 못함" in first.warnings
    assert "전사문에 없는 숫자 '20'" in first.warnings
    assert second.warnings == ["근거 인용 없음"]
    auto = [i for i in draft.sections["needs_review"] if i.auto]
    assert len(auto) == 2 and all(i.text.startswith("[자동 점검]") for i in auto)


def test_verify_reports_text_without_speech_and_low_confidence_terms():
    segments = SEGMENTS + [
        TranscriptSegment(20.0, 30.0, "시청해 주셔서 감사합니다.", speech_detected=False),
        TranscriptSegment(31.0, 33.0, "omeprazole 20mg", uncertain_terms=["20mg"]),
    ]
    draft = ChartDraft.from_structured(CHART)
    verify_draft(draft, segments)
    texts = [i.text for i in draft.sections["needs_review"]]
    assert any("음성이 감지되지 않은 곳의 전사 1건" in t for t in texts)
    assert any("20mg" in t for t in texts)


def test_transcript_numbers_understand_korean_number_words():
    nums = transcript_numbers("사흘 전부터, 하루 두 번, 오백 밀리, 혈압은 130에 85, 일주일 뒤, 열두 시")
    for n in (3, 1, 2, 500, 130, 85, 7, 12):
        assert float(n) in nums
