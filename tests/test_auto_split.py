from clinic_scribe.auto_split import find_cue, parse_cues
from clinic_scribe.config import Settings

NEXT = parse_cues(Settings().split_cues_next)
END = parse_cues(Settings().split_cues_end)


def test_parse_cues_from_settings_text():
    assert parse_cues(" 들어오세요, 다음 환자 ,\n안녕히 가세요,, ") == ["들어오세요", "다음 환자", "안녕히 가세요"]
    assert "들어오세요" in NEXT and "안녕히 가세요" in END


def test_next_patient_cue_ignores_spacing_and_punctuation():
    cue = find_cue("네, 다음 분 들어 오세요!", NEXT, END)
    assert cue is not None and cue.kind == "next" and cue.phrase == "들어오세요"


def test_farewell_cue():
    cue = find_cue("네 그럼 일주일 뒤에 봬요. 안녕히가세요.", NEXT, END)
    assert cue is not None and cue.kind == "end"


def test_farewell_wins_when_both_in_one_sentence():
    # "안녕히 가세요. 다음 분 들어오세요." → 그 문장까지 앞 진료로 본다
    assert find_cue("안녕히 가세요. 다음 분 들어오세요.", NEXT, END).kind == "end"


def test_ordinary_consult_sentences_are_not_cues():
    for text in ("사흘 전부터 식후에 속이 쓰려요.", "오늘은 경과를 보고 계속되면 검사를 논의하죠.", "검사실로 가세요.", ""):
        assert find_cue(text, NEXT, END) is None
