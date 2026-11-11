import pytest

from clinic_scribe.conversation_summary import GENERAL_BANNER, ConversationSummary
from clinic_scribe.export import build_markdown
from clinic_scribe.session import Session
from tests.helpers import GENERAL


def test_text_is_key_point_focused_and_grouped():
    summary = ConversationSummary.from_structured(GENERAL, model="m", based_on="전사 #1")
    text = summary.to_text()
    assert text.startswith(GENERAL_BANNER)
    assert text.index("■ 한 줄 요약") < text.index("■ 주요 내용") < text.index("■ 할 일·결정 사항") < text.index("■ 확인 필요")
    assert "[근황]\n- 주말에 한강에서 자전거\n- 저녁에 치킨\n[건강·생활]\n- 업무가 많아 수면 부족" in text
    assert "■ 확인 필요\n- 없음" in text
    html = summary.to_html()
    assert "근황" in html and "업무가 많아 수면 부족" in html


def test_empty_groups_are_dropped():
    data = {**GENERAL, "key_points": [{"category": "빈 묶음", "items": ["  "]}, {"category": "", "items": ["항목"]}]}
    summary = ConversationSummary.from_structured(data)
    assert [(g.category, g.items) for g in summary.key_points] == [("기타", ["항목"])]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {**GENERAL, "one_line": None},
        {**GENERAL, "key_points": ["문자열"]},
        {**GENERAL, "key_points": [{"category": "x", "items": "문자열"}]},
        {**GENERAL, "action_items": [1]},
    ],
)
def test_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        ConversationSummary.from_structured(bad)


def test_export_uses_summary_kind():
    session = Session(session_id=1, source="파일 · daily.wav")
    md = build_markdown(session, None, ConversationSummary.from_structured(GENERAL))
    assert "## 주요 내용 요약" in md and GENERAL_BANNER in md
