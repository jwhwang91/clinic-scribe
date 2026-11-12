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


GENERAL = {
    "one_line": "주말 근황과 업무 부담으로 잠을 못 잔다는 일상 대화",
    "key_points": [
        {"category": "근황", "items": ["주말에 한강에서 자전거", "저녁에 치킨"]},
        {"category": "건강·생활", "items": ["업무가 많아 수면 부족"]},
    ],
    "action_items": ["주말에 쉬기로 함"],
    "needs_review": [],
}


def _envelope(**fields) -> str:
    base = {"type": "result", "subtype": "success", "is_error": False, "api_error_status": None, "num_turns": 1}
    base.update(fields)
    return json.dumps(base, ensure_ascii=False)


SCENARIOS = {
    "success": ("print({out!r})", 0),
    "general": ("print({out!r})", 0),
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
    "general": _envelope(result=json.dumps(GENERAL, ensure_ascii=False), structured_output=GENERAL, total_cost_usd=0.0),
    "usage_limit": _envelope(is_error=True, api_error_status=429, result="Claude AI usage limit reached|1767225600"),
    "auth": _envelope(is_error=True, result="Not logged in · Please run /login", terminal_reason="api_error"),
    "model": _envelope(
        is_error=True, api_error_status=404,
        result="There's an issue with the selected model (claude-x). It may not exist or you may not have access to it.",
    ),
    "overloaded": _envelope(is_error=True, api_error_status=529, result='API Error: 529 {"type":"overloaded_error"}'),
    "bad_schema": _envelope(structured_output={"chief_complaint_hpi": "문자열이면 안 됨"}),
}


def make_executable(path: Path, code: str) -> Path:
    """python 코드로 된 가짜 실행 파일.

    Windows 는 shebang 이 없고 .bat 는 cmd.exe 가 인자(따옴표·꺾쇠·한글)를 다시 해석해 깨뜨리므로, pip 이 콘솔 스크립트를
    만드는 것과 같이 실행 파일(launcher .exe + 실행할 python + __main__.py 를 담은 zip)로 만든다.
    """
    if sys.platform == "win32":
        import io
        import platform
        import zipfile

        import pip

        stub = "t64-arm.exe" if platform.machine().lower() in ("arm64", "aarch64") else "t64.exe"
        launcher = (Path(pip.__file__).parent / "_vendor" / "distlib" / stub).read_bytes()
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("__main__.py", code)
        exe = path.with_suffix(".exe")
        exe.write_bytes(launcher + f'#!"{sys.executable}"\n'.encode() + archive.getvalue())
        return exe
    path.write_text(f"#!{sys.executable}\n{code}", encoding="utf-8")
    path.chmod(0o755)
    return path


def make_fake_cli(directory: Path, scenario: str, record: Path | None = None) -> str:
    """scenario 대로 응답하는 가짜 claude. record 를 주면 받은 인자·stdin·환경을 JSON 으로 남긴다.

    "success" 는 요청한 JSON 스키마를 보고 진료 차트 초안 또는 주요 내용 요약을 돌려준다.
    """
    action, code = SCENARIOS[scenario]
    action = action.format(out=OUTPUTS.get(scenario, ""))
    if scenario == "success":
        action = (
            "import sys\n"
            "schema = sys.argv[sys.argv.index('--json-schema') + 1] if '--json-schema' in sys.argv else ''\n"
            f"print({OUTPUTS['general']!r} if 'one_line' in schema else {OUTPUTS['success']!r})"
        )
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
    return str(make_executable(directory / f"fake-claude-{scenario}", f"{record_code}\n{action}\nraise SystemExit({code})\n"))


FAKE_LLAMA_SERVER = r'''
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
port, key = int(args[args.index("--port") + 1]), args[args.index("--api-key") + 1]
RECORD, DELAY, ANSWERS = __RECORD__, __DELAY__, __ANSWERS__


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != "Bearer " + key:
            self.send_response(401)
            self.end_headers()
            return
        if RECORD:
            with open(RECORD, "w", encoding="utf-8") as f:
                json.dump({"args": args, "body": body}, f, ensure_ascii=False)
        schema = body["response_format"]["json_schema"]["schema"]
        text = json.dumps(ANSWERS["general" if "one_line" in schema["properties"] else "chart"], ensure_ascii=False)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(text), 25):
            chunk = {"choices": [{"delta": {"content": text[i:i + 25]}}]}
            self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(DELAY)
        self.wfile.write(b"data: [DONE]\n\n")


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def make_fake_llama_server(directory: Path, record: Path | None = None, delay: float = 0.0) -> Path:
    """llama-server 와 같은 인자·HTTP API 로 답하는 가짜 (요청한 스키마에 따라 차트 초안 또는 주요 내용 요약)."""
    code = (FAKE_LLAMA_SERVER.replace("__RECORD__", repr(str(record) if record else ""))
            .replace("__DELAY__", repr(delay)).replace("__ANSWERS__", repr({"chart": CHART, "general": GENERAL})))
    return make_executable(directory / "fake-llama-server", code)
