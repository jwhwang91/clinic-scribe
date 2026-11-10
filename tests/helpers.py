"""테스트 도우미: 가상 음성 읽기, 가짜 `claude` 실행 파일(실제 Claude 서비스에 아무것도 보내지 않음)."""

import json
import sys
import textwrap
from pathlib import Path


def load_pcm(path: Path):
    """오디오 파일 → 앱과 같은 방식으로 16 kHz mono int16."""
    from clinic_scribe.audio import load_audio_file, to_whisper_pcm

    data, rate = load_audio_file(path)
    return to_whisper_pcm(data, rate)


CHART = {
    "chief_complaint_hpi": [
        {"text": "3일 전부터 식후 속쓰림", "source": "환자", "evidence": "사흘 전부터 식후에 속이 쓰려요"},
        {"text": "발열 없음", "source": "환자", "evidence": "열은 없어요"},
    ],
    "exam_findings": [],
    "assessment_plan": [
        {"text": "지속 시 검사 논의", "source": "의사", "evidence": "계속되면 검사를 논의하죠"},
    ],
    "needs_review": [],
}


def _envelope(**fields) -> str:
    base = {"type": "result", "subtype": "success", "is_error": False, "api_error_status": None, "num_turns": 1}
    base.update(fields)
    return json.dumps(base, ensure_ascii=False)


SCENARIOS = {
    "success": ("print({out!r})", 0),
    "usage_limit": ("print({out!r})", 1),
    "auth": ("print({out!r})", 1),
    "model": ("print({out!r})", 1),
    "overloaded": ("print({out!r})", 1),
    "garbage": ("print('this is not json')", 0),
    "bad_schema": ("print({out!r})", 0),
    "sleep": ("import time; time.sleep(60)", 0),
}

OUTPUTS = {
    "success": _envelope(result=json.dumps(CHART, ensure_ascii=False), structured_output=CHART, total_cost_usd=0.0),
    "usage_limit": _envelope(is_error=True, api_error_status=429, result="Claude AI usage limit reached|1767225600"),
    "auth": _envelope(is_error=True, result="Not logged in · Please run /login", terminal_reason="api_error"),
    "model": _envelope(
        is_error=True, api_error_status=404,
        result="There's an issue with the selected model (claude-x). It may not exist or you may not have access to it.",
    ),
    "overloaded": _envelope(is_error=True, api_error_status=529, result='API Error: 529 {"type":"overloaded_error"}'),
    "bad_schema": _envelope(structured_output={"chief_complaint_hpi": "문자열이면 안 됨"}),
}


def make_fake_cli(directory: Path, scenario: str, record: Path | None = None) -> str:
    """scenario 대로 응답하는 가짜 claude. record 를 주면 받은 인자·stdin·환경을 JSON 으로 남긴다."""
    action, code = SCENARIOS[scenario]
    action = action.format(out=OUTPUTS.get(scenario, ""))
    record_code = ""
    if record is not None:
        record_code = textwrap.dedent(
            f"""
            import json, os, sys
            data = {{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd(),
                     "CLAUDECODE": os.environ.get("CLAUDECODE"),
                     "CLAUDE_CODE_SESSION_ID": os.environ.get("CLAUDE_CODE_SESSION_ID"),
                     "NONESSENTIAL": os.environ.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")}}
            open({str(record)!r}, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False))
            """
        )
    path = directory / f"fake-claude-{scenario}"
    path.write_text(f"#!{sys.executable}\n{record_code}\n{action}\nraise SystemExit({code})\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)
