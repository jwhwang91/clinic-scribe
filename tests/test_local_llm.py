"""로컬 요약 모델 (llama-server): 가짜 서버로 요청 형식·열쇠·스트리밍·취소를 확인한다 (실제 모델은 받지 않는다)."""

import json
import threading
import time

import pytest

from clinic_scribe.local_llm import LocalLLM, LocalLLMError
from clinic_scribe.summarizer import MODES, LocalConfig, SummaryError, summarize_local
from clinic_scribe.whisper import TranscriptSegment
from tests.helpers import CHART, GENERAL, make_executable, make_fake_llama_server

SEGMENTS = [
    TranscriptSegment(0.0, 2.0, "어디가 불편하세요?", speaker="김원장"),
    TranscriptSegment(2.5, 6.0, "사흘 전부터 식후에 속이 쓰려요. 열은 없어요.", speaker="환자"),
]


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "Qwen3-8B-Q4_K_M.gguf"
    path.write_bytes(b"GGUF")
    return path


def test_summary_with_schema_key_and_no_thinking(tmp_path, model_file):
    record = tmp_path / "request.json"
    llm = LocalLLM(make_fake_llama_server(tmp_path, record), model_file, threads=6)
    try:
        structured, meta = summarize_local(SEGMENTS, LocalConfig("Qwen3 8B (이 컴퓨터)", 60, "chart"), llm)
        assert structured == CHART and meta == {"engine": "local", "model": "Qwen3 8B (이 컴퓨터)"}
        sent = json.loads(record.read_text(encoding="utf-8"))
        args, body = sent["args"], sent["body"]
        assert args[args.index("--host") + 1] == "127.0.0.1" and "--api-key" in args and "--jinja" in args
        assert args[args.index("-m") + 1] == str(model_file) and args[args.index("-t") + 1] == "6"
        assert body["response_format"]["json_schema"]["schema"] == MODES["chart"][2]  # Claude 와 같은 스키마
        assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["stream"] is True
        assert body["messages"][0]["content"] == MODES["chart"][0]
        assert "<transcript>" in body["messages"][1]["content"] and "김원장(의사): 어디가 불편하세요?" in body["messages"][1]["content"]
        assert llm.running  # 서버는 켜 둔다: 다음 요약은 모델을 다시 올리지 않는다
        structured, _ = summarize_local(SEGMENTS, LocalConfig("Qwen3 8B", 60, "general"), llm)
        assert structured == GENERAL
    finally:
        llm.close()
    assert not llm.running


def test_cancel_stops_quickly(tmp_path, model_file):
    llm = LocalLLM(make_fake_llama_server(tmp_path, delay=0.5), model_file)
    cancel = threading.Event()
    try:
        llm.chat_json("s", "u", MODES["general"][2], 60)  # 서버를 먼저 띄운다
        threading.Timer(0.6, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(LocalLLMError) as info:
            llm.chat_json("s", "u", MODES["general"][2], 60, cancel)
        assert info.value.kind == "cancelled" and time.monotonic() - started < 3
    finally:
        llm.close()


def test_missing_files_and_errors_become_summary_errors(tmp_path, model_file):
    with pytest.raises(SummaryError) as info:
        summarize_local(SEGMENTS, LocalConfig("x"), LocalLLM(tmp_path / "없음", model_file))
    assert info.value.kind == "not_found" and "설치" in str(info.value)
    with pytest.raises(SummaryError) as info:
        summarize_local(SEGMENTS, LocalConfig("x"), LocalLLM(make_fake_llama_server(tmp_path), tmp_path / "없는.gguf"))
    assert info.value.kind == "not_found"
    broken = make_executable(tmp_path / "broken-server", "raise SystemExit(3)\n")
    with pytest.raises(SummaryError) as info:
        summarize_local(SEGMENTS, LocalConfig("x"), LocalLLM(broken, model_file))
    assert info.value.kind == "failed" and "메모리" in str(info.value)  # 바로 끝나면 메모리·모델 파일을 의심하라고 안내
    with pytest.raises(SummaryError):
        summarize_local([TranscriptSegment(0, 1, " ")], LocalConfig("x"), LocalLLM(broken, model_file))
