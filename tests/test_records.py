"""진료 기록 저장소: 무음 줄이기, 저장 형식, SQLite·녹음 파일 저장·삭제 (가상 데이터만 쓴다)."""

import sqlite3
import sys
from datetime import datetime

import numpy as np
import pytest

from clinic_scribe.chart import ChartDraft, ChartItem
from clinic_scribe.config import VadParams
from clinic_scribe.conversation_summary import ConversationSummary, KeyPointGroup
from clinic_scribe.records import (
    INTERRUPTED,
    MAX_PAUSE_S,
    RecordError,
    RecordSaver,
    RecordStore,
    apply_data,
    record_row,
    remap_result,
    remap_time,
    session_data,
    session_from_row,
    trim_silence,
)
from clinic_scribe.session import Session, TranscriptRun
from clinic_scribe.whisper import SpeechRegion, TranscriptionResult, TranscriptSegment

RATE = 16000


def make_result() -> TranscriptionResult:
    segments = [
        TranscriptSegment(1.0, 3.0, "어디가 불편하세요?", confidence=0.9, region=0, speaker="김원장"),
        TranscriptSegment(10.0, 12.0, "트리메부틴은 하루 세 번", region=1, speaker="김원장", original_text="트림에 부티는 하루 세 번",
                          corrections=["트림에 부티는 → 트리메부틴은"], uncertain_terms=["세"]),
        TranscriptSegment(12.5, 13.0, "네.", region=2),
        TranscriptSegment(5.0, 6.0, "자막 제공", speech_detected=False),
    ]
    regions = [SpeechRegion(1.0, 3.0), SpeechRegion(10.0, 12.0), SpeechRegion(12.5, 13.0)]
    return TranscriptionResult(segments, regions, True, VadParams(min_silence_ms=700), 20.0, 3.5, doctor="김원장",
                               phrases=list(segments))


def test_trim_silence_shortens_long_pauses_and_keeps_speech():
    pcm = np.zeros(RATE * 20, np.int16)
    pcm[RATE * 1:RATE * 3] = 1000
    pcm[RATE * 10:RATE * 13] = -800
    trimmed = trim_silence(pcm, make_result().regions)
    half = MAX_PAUSE_S / 2
    # 10~12 와 12.5~13 사이 0.5초 쉼은 그대로 (한 덩어리), 3~10 의 7초 무음은 1.2초로
    assert trimmed.spans == [(1.0 - half, 3.0 + half), (10.0 - half, 13.0 + half)]
    assert len(trimmed.pcm16) == round((3.0 + 2 * half - 1.0 + 13.0 + 2 * half - 10.0) * RATE)
    first = slice(round(half * RATE), round((half + 2.0) * RATE))
    assert np.array_equal(trimmed.pcm16[first], pcm[RATE * 1:RATE * 3])  # 말소리는 한 샘플도 바뀌지 않는다
    assert remap_time(3.0, trimmed.spans) == pytest.approx(2.0 + half)
    assert remap_time(10.0, trimmed.spans) - remap_time(3.0, trimmed.spans) == pytest.approx(MAX_PAUSE_S)
    assert remap_time(6.0, trimmed.spans) == pytest.approx(2.0 + 2 * half)  # 잘라 낸 무음 안 → 이어 붙인 자리
    assert remap_time(19.0, trimmed.spans) == pytest.approx(len(trimmed.pcm16) / RATE)


def test_trim_silence_without_regions_or_audio():
    assert len(trim_silence(np.zeros(RATE, np.int16), []).pcm16) == 0
    short = trim_silence(np.ones(RATE // 2, np.int16), [SpeechRegion(0.0, 0.5)])  # 처음·끝이 녹음 밖이면 녹음 끝까지만
    assert short.spans == [(0.0, 0.5)] and len(short.pcm16) == RATE // 2


def test_remap_result_moves_times_and_keeps_text():
    result = make_result()
    spans = trim_silence(np.zeros(RATE * 20, np.int16), result.regions).spans
    moved = remap_result(result, spans)
    assert [s.text for s in moved.segments] == [s.text for s in result.segments]
    assert moved.segments[1].speaker == "김원장" and moved.segments[1].corrections == ["트림에 부티는 → 트리메부틴은"]
    assert moved.segments[1].start == pytest.approx(remap_time(10.0, spans), abs=0.001)
    assert moved.regions[1].start - moved.regions[0].end == pytest.approx(MAX_PAUSE_S, abs=0.001)
    assert moved.audio_duration == pytest.approx(sum(b - a for a, b in spans), abs=0.001)
    assert moved.phrases is None and result.segments[1].start == 10.0  # 원래 결과는 그대로


def make_session(session_id: int = 7) -> Session:
    s = Session(session_id=session_id, source="녹음", created_at=datetime(2026, 11, 12, 10, 2, 5), duration_s=431.5,
                end_reason="다음 환자 신호 '들어오세요'")
    s.runs.append(TranscriptRun(s.new_run_id(), make_result()))
    s.draft = ChartDraft(
        sections={"chief_complaint_hpi": [ChartItem("3일 전부터 식후 속쓰림", "환자", "사흘 전부터", warnings=["숫자 확인"])],
                  "exam_findings": [], "assessment_plan": [], "needs_review": [ChartItem("용량", reason="불명확", auto=True)]},
        model="claude-opus-5-5", based_on="진료 #7 · 전사 #1", created_at="2026-11-12 10:10",
    )
    s.draft_run_id, s.summary_seconds, s.audio_trimmed = 1, 12.5, True
    return s


def test_store_round_trip_list_details_and_audio(tmp_path):
    store = RecordStore(tmp_path / "records")
    session = make_session()
    pcm = (np.sin(np.arange(RATE * 3) / 9) * 9000).astype(np.int16)
    store.save(record_row(session, "요약 준비됨"), session_data(session), pcm)

    [row] = store.load_index()
    assert (row.id, row.status, row.doctor, row.audio, row.audio_s) == (7, "요약 준비됨", "김원장", "7.flac", 3.0)
    loaded = session_from_row(row)
    assert not loaded.loaded and loaded.stored_audio and loaded.audio_trimmed and loaded.created_at == session.created_at
    assert loaded.stored_status == "요약 준비됨" and loaded.end_reason == session.end_reason
    apply_data(loaded, store.load_data(7))
    assert loaded.loaded and loaded.runs[0].result == session.runs[0].result.__class__(**{
        **session.runs[0].result.__dict__, "phrases": None})  # 말 덩어리만 빼고 그대로
    assert loaded.draft == session.draft and loaded.draft_run_id == 1 and loaded.summary_seconds == 12.5
    assert loaded.new_run_id() == 2  # 다시 전사하면 다음 번호
    assert np.array_equal(store.read_audio(7), pcm)  # 무손실 FLAC

    # 내용만 고쳐 저장하면 녹음 파일은 그대로 둔다
    session.summary_stale = True
    store.save(record_row(session, "요약 준비됨"), session_data(session))
    assert store.load_index()[0].audio == "7.flac" and store.load_data(7)["summary_stale"] is True
    assert np.array_equal(store.read_audio(7), pcm)


def test_general_summary_round_trip(tmp_path):
    store = RecordStore(tmp_path)
    session = make_session(3)
    session.draft = ConversationSummary("주말 근황", [KeyPointGroup("근황", ["한강 자전거"])], ["쉬기"], [], model="m")
    store.save(record_row(session, "요약 준비됨"), session_data(session))
    loaded = session_from_row(store.load_index()[0])
    apply_data(loaded, store.load_data(3))
    assert loaded.draft == session.draft and loaded.draft.kind_label == "주요 내용 요약"


def test_private_permissions_and_delete(tmp_path):
    store = RecordStore(tmp_path / "records")
    store.save(record_row(make_session(), "요약 준비됨"), session_data(make_session()), np.ones(RATE, np.int16))
    if sys.platform != "win32":  # Windows 는 사용자 폴더(%LOCALAPPDATA%) 권한으로 보호된다
        assert store.root.stat().st_mode & 0o777 == 0o700 and store.audio_dir.stat().st_mode & 0o777 == 0o700
        assert store.db_path.stat().st_mode & 0o777 == 0o600 and store.audio_path(7).stat().st_mode & 0o777 == 0o600
    assert "트리메부틴" in store.db_path.read_bytes().decode("utf-8", "ignore")
    store.delete(7)
    assert store.load_index() == [] and not store.audio_path(7).exists()
    assert "트리메부틴" not in store.db_path.read_bytes().decode("utf-8", "ignore")  # 지운 전사문이 파일에 남지 않는다


def test_ids_continue_and_are_not_reused(tmp_path):
    store = RecordStore(tmp_path)
    first = store.allocate_id()
    store.save(record_row(make_session(first), "대기"), session_data(make_session(first)))
    second = store.allocate_id()
    store.delete(second)
    assert first == 1 and second == 2 and RecordStore(tmp_path).allocate_id() == 3


def test_interrupted_processing_becomes_retryable(tmp_path):
    store = RecordStore(tmp_path)
    fresh = Session(session_id=1, source="녹음")
    store.save(record_row(fresh, "전사 중…", "transcribe"), session_data(fresh), np.ones(RATE, np.int16))
    summarizing = make_session(2)
    summarizing.draft = None
    store.save(record_row(summarizing, "요약 중…", "summary"), session_data(summarizing))
    rows = {row.id: session_from_row(row) for row in store.load_index()}
    assert rows[1].stored_status == "전사 실패" and rows[2].stored_status == "요약 실패"
    for session in rows.values():
        apply_data(session, store.load_data(session.session_id))
    assert rows[1].transcribe_error == INTERRUPTED["transcribe"] and not rows[1].runs
    assert rows[2].summary_error == INTERRUPTED["summary"] and rows[2].runs


def test_newer_schema_and_broken_rows_are_refused(tmp_path):
    store = RecordStore(tmp_path)
    store.save(record_row(make_session(), "요약 준비됨"), {"runs": [{"result": {}}]})  # run_id 가 없는 깨진 내용
    with pytest.raises(RecordError):
        apply_data(session_from_row(store.load_index()[0]), store.load_data(7))
    with sqlite3.connect(store.db_path) as db:
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(RecordError, match="새로운 버전"):
        RecordStore(tmp_path)


def test_saver_keeps_order_and_ignores_saves_after_delete(tmp_path):
    store = RecordStore(tmp_path)
    errors = []
    saver = RecordSaver(store, errors.append)
    session = make_session()
    saver.save(record_row(session, "전사 중…", "transcribe"), session_data(session), np.ones(RATE, np.int16))
    saver.delete(session.session_id)
    saver.save(record_row(session, "요약 준비됨"), session_data(session))  # 지운 뒤에 끝난 처리의 저장은 버린다
    assert saver.flush()
    assert store.load_index() == [] and not store.audio_path(7).exists() and not errors
    saver.close()
    saver.save(record_row(make_session(8), "대기"), session_data(make_session(8)))  # 닫은 뒤에는 아무것도 하지 않는다
    assert store.load_index() == []
    assert RecordSaver(None).flush()  # 저장소가 없으면 저장하지 않는다


def test_rename_speaker_updates_saved_records(tmp_path):
    store = RecordStore(tmp_path)
    session = make_session()
    store.save(record_row(session, "요약 준비됨"), session_data(session))
    assert store.rename_speaker("김원장", "김민수") == 1
    row = store.load_index()[0]
    loaded = session_from_row(row)
    apply_data(loaded, store.load_data(7))
    assert row.doctor == "김민수" and loaded.runs[0].result.doctor == "김민수"
    assert [seg.speaker for seg in loaded.runs[0].result.segments] == ["김민수", "김민수", None, None]
    assert store.rename_speaker("없는 이름", "아무개") == 0
