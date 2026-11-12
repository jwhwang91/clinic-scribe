import numpy as np
import pytest

from clinic_scribe import diction as diction_module
from clinic_scribe.diction import (
    COMMON,
    MAX_PROMPT_CHARS,
    Correction,
    DictionBook,
    DictionError,
    apply_corrections,
    build_prompt,
    check_correction,
    guess_doctor,
)
from clinic_scribe.speakers import PATIENT, SAMPLE_RATE, VoiceBook
from clinic_scribe.whisper import TranscriptSegment


def seg(text, speaker=None):
    return TranscriptSegment(0.0, 2.0, text, speaker=speaker)


def test_corrections_ignore_spacing_and_case_and_keep_original(tmp_path):
    book = DictionBook(tmp_path / "diction.json")
    book.add_correction(COMMON, "오메 프라졸", "omeprazole")
    book.add_correction(COMMON, "tylenol er", "타이레놀 ER")
    book.add_correction("김원장", "트림에 부티는", "트리메부틴은")
    segments = [
        seg("오메프라졸은 아침에 드세요.", "김원장"),
        seg("Tylenol ER 은 먹어 봤어요.", PATIENT),
        seg("트림에부티는 배가 아플 때만 드세요.", "김원장"),
        seg("트림에 부티는 뭐예요?", PATIENT),  # 의사 사전은 그 의사 발화에만
    ]
    fixed = apply_corrections(segments, book)
    assert [s.text for s in fixed] == [
        "omeprazole은 아침에 드세요.",
        "타이레놀 ER 은 먹어 봤어요.",
        "트리메부틴은 배가 아플 때만 드세요.",
        "트림에 부티는 뭐예요?",
    ]
    assert fixed[2].original_text == "트림에부티는 배가 아플 때만 드세요."
    assert fixed[2].corrections == ["트림에 부티는 → 트리메부틴은"]
    assert fixed[3].original_text is None and fixed[3] is segments[3]


def test_corrections_are_single_pass_and_repeatable(tmp_path):
    book = DictionBook(tmp_path / "diction.json")
    book.add_correction(COMMON, "프라졸", "오메프라졸")  # 바른 말 안에 잘못 들린 말이 들어 있다
    book.add_correction(COMMON, "가나", "다라")
    book.add_correction(COMMON, "다라", "마바")  # 바꾼 말이 다시 바뀌면 안 된다
    once = apply_corrections([seg("프라졸 가나 다라")], book)
    twice = apply_corrections(once, book)
    assert once[0].text == twice[0].text == "오메프라졸 다라 마바"
    book.set_entry(COMMON, [], [])  # 교정을 지우면 원래 들린 말로 돌아간다
    assert apply_corrections(twice, book)[0].text == "프라졸 가나 다라"
    assert apply_corrections(twice, book)[0].original_text is None


def test_doctor_rule_wins_over_common_for_that_doctor(tmp_path):
    book = DictionBook(tmp_path / "diction.json")
    book.add_correction(COMMON, "현타객", "현탁액")
    book.add_correction("김원장", "현타객", "현탁액제")
    fixed = apply_corrections([seg("현타객", "김원장"), seg("현타객", PATIENT), seg("현타객")], book)
    assert [s.text for s in fixed] == ["현탁액제", "현탁액", "현탁액"]


def test_diction_book_validation_save_and_load(tmp_path):
    for wrong, right in (("가", "나"), ("오메", " "), ("오메 프라졸", "오메프라졸")):
        with pytest.raises(DictionError):
            check_correction(wrong, right)
    book = DictionBook(tmp_path / "diction.json")
    dropped = book.set_entry("김원장", ["세티리진", " 세티 리진 ", "레바미피드", ""],
                             [Correction("현타객", "현탁액"), Correction("가", "나"), Correction("현 타객", "현탁액!")])
    assert dropped == 1
    entry = book.entry("김원장")
    assert entry.vocabulary == ["세티리진", "레바미피드"]  # 띄어쓰기·중복 정리
    assert entry.corrections == [Correction("현 타객", "현탁액!")]  # 같은 말은 나중 것
    book.add_correction("김원장", "트림에 부티는", "트리메부틴은")
    assert "트리메부틴은" in book.entry("김원장").vocabulary  # 바른 말은 용어 목록에도 들어간다
    book.set_entry("이원장", [], [])
    book.save()
    loaded = DictionBook.load(book.path)
    assert loaded.doctors() == ["김원장"] and loaded.entry("김원장") == book.entry("김원장")
    assert DictionBook.load(tmp_path / "없음.json").empty
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(DictionError):
        DictionBook.load(tmp_path / "bad.json")


def test_build_prompt_puts_doctor_terms_first_and_stays_short(tmp_path):
    base = "CBC, CT, MRI, PPI, NSAIDs, omeprazole, acetaminophen, amoxicillin."
    book = DictionBook(tmp_path / "diction.json")
    assert build_prompt(base, book, None) == base
    book.set_entry(COMMON, ["알마게이트", "Omeprazole"], [Correction("새티리지", "세티리진")])
    book.set_entry("김원장", ["트리메부틴", "알마게이트"], [Correction("트림에 부티는", "트리메부틴은")])
    # 용어 목록은 마침표로 끝낸다. 기본 힌트에 있는 말과 조사만 붙은 말("트리메부틴은")은 뺀다.
    assert build_prompt(base, book, "김원장") == f"트리메부틴, 알마게이트, 세티리진. {base}"
    assert build_prompt(base, book, None) == f"알마게이트, 세티리진. {base}"
    assert build_prompt("", book, None) == "알마게이트, Omeprazole, 세티리진."  # 기본 힌트가 비면 그 말도 넣는다
    book.set_entry("이원장", [f"아주긴약이름{i:02d}번" for i in range(80)], [])
    long = build_prompt(base, book, "이원장")
    assert len(long) <= MAX_PROMPT_CHARS and long.endswith(base) and long.startswith("아주긴약이름00번")


class FakeEncoder:
    def embed(self, samples):
        vector = np.zeros(8, np.float32)
        vector[int(round(float(np.median(np.abs(samples))))) % 8] = 1.0
        return vector


def test_guess_doctor_by_majority(tmp_path, monkeypatch):
    monkeypatch.setattr(diction_module, "speech_only", lambda pcm: pcm.astype(np.float32))  # 말소리만 남기기는 건너뜀
    voices = VoiceBook(tmp_path / "voices.json")
    axis = np.eye(8, dtype=np.float32)
    voices.enroll("김원장", [axis[1]], 20)
    voices.enroll("이원장", [axis[3]], 20)
    three = int(3 * SAMPLE_RATE)
    pcm = np.concatenate([np.full(2 * three, 1, np.int16), np.full(three, 2, np.int16), np.full(three, 3, np.int16)])
    assert guess_doctor(pcm, FakeEncoder(), voices, 0.5) == "김원장"  # 3초 조각 4개: 김원장 2, 환자 1, 이원장 1
    assert guess_doctor(np.full(4 * three, 2, np.int16), FakeEncoder(), voices, 0.5) is None  # 등록한 의사 목소리가 없다
    assert guess_doctor(pcm, FakeEncoder(), VoiceBook(tmp_path / "none.json"), 0.5) is None


@pytest.mark.models
def test_doctor_terms_and_corrections_fix_drug_names(samples, whisper_runner, speaker_encoder, settings, tmp_path):
    """가상 진료: 의사를 알아내 그 의사의 용어로 전사하고, 그 의사의 교정으로 잘못 들린 약 이름을 고친다."""
    from clinic_scribe.audio import load_audio_file, to_whisper_pcm
    from clinic_scribe.speakers import enrollment_embeddings, label_segments, speech_only

    def pcm(name):
        data, rate = load_audio_file(samples / name)
        return to_whisper_pcm(data, rate)

    voices = VoiceBook(tmp_path / "voices.json")
    speech = speech_only(pcm("voice_enroll_doctor.wav"))
    voices.enroll("김원장", enrollment_embeddings(speaker_encoder, speech), len(speech) / SAMPLE_RATE)
    book = DictionBook(tmp_path / "diction.json")
    # 이 녹음을 기본 힌트로 전사하면 "트림에 부티는", "현타객은", "새티리지는"으로 잘못 받아 적는다
    book.set_entry("김원장", ["트리메부틴", "알마게이트", "세티리진"],
                   [Correction("트림에 부티는", "트리메부틴은"), Correction("현타객", "현탁액")])

    consult = pcm("doctor_drugs.wav")
    threshold = settings.speaker_threshold

    def transcribe(prompt=None):
        result = whisper_runner.transcribe(consult, use_vad=True, params=settings.vad, prompt=prompt)
        return apply_corrections(label_segments(result.segments, consult, speaker_encoder, voices, threshold), book)

    # 1) 기본 힌트로 받아 적어도 교정 사전이 그 의사 발화의 잘못 들린 말을 고친다 (원래 들린 말은 남는다)
    plain = transcribe()
    plain_text = " ".join(s.text for s in plain)
    assert "트리메부틴은" in plain_text and "현탁액" in plain_text, plain_text
    assert any(s.original_text and "트림에" in s.original_text for s in plain)

    # 2) 이 진료 의사를 알아내 그 의사의 용어를 힌트로 넣으면 사전에 교정이 없는 '세티리진'도 처음부터 바르게 받아 적는다
    doctor = guess_doctor(consult, speaker_encoder, voices, threshold)
    assert doctor == "김원장"
    final = transcribe(build_prompt(settings.whisper_prompt, book, doctor))
    text = " ".join(s.text for s in final)
    for word in ("트리메부틴은", "현탁액", "세티리진"):
        assert word in text, text
    assert len(final) >= 3  # 힌트 때문에 의사·환자 말이 한 줄로 합쳐지지 않는다
    patient = [s for s in final if s.speaker == PATIENT]
    assert patient and all(not s.corrections for s in patient)  # 환자 말에는 의사 사전을 쓰지 않는다


def test_rename_scope_moves_and_merges_dictionary(tmp_path):
    book = DictionBook(tmp_path / "diction.json")
    book.add_correction("황재", "트림에 부티는", "트리메부틴은")
    book.rename_scope("황재", "황재웅")
    assert book.doctors() == ["황재웅"] and book.entry("황재웅").corrections[0].right == "트리메부틴은"
    book.add_correction("황재", "현타객", "현탁액")  # 옛 이름으로 다시 생긴 사전은 합친다
    book.rename_scope("황재", "황재웅")
    assert book.doctors() == ["황재웅"] and {c.right for c in book.entry("황재웅").corrections} == {"트리메부틴은", "현탁액"}
    book.rename_scope("없는 이름", "아무개")  # 사전이 없으면 아무 일도 없다
    assert book.doctors() == ["황재웅"]
