"""로컬 요약 모델 실행 (llama.cpp llama-server). 전사문이 이 컴퓨터 밖으로 나가지 않는다.

- 앱이 켜져 있는 동안 llama-server 를 한 번 띄워 두고(처음 요약할 때 모델을 메모리에 올린다) 앱을 닫을 때 끈다.
- 127.0.0.1 의 빈 포트에서만 듣고, 실행할 때마다 만든 열쇠(--api-key)를 아는 요청만 받는다.
- 서버의 표준 출력·오류는 버린다(전사문이 로그에 남지 않게).
- 응답은 흘려받아(stream) 취소하면 바로 끊고, JSON 스키마로 형식을 강제해 Claude 요약과 같은 구조로 받는다.
"""

from __future__ import annotations

import http.client
import json
import re
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path

from clinic_scribe import procutil

CONTEXT = 16384  # 긴 진료(전사문 수천 자)와 작성 규칙·응답을 함께 담는 문맥 길이
LOAD_TIMEOUT_S = 300  # 큰 모델을 처음 메모리에 올리는 시간


class LocalLLMError(Exception):
    """kind: not_found, start, timeout, cancelled, failed"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalLLM:
    def __init__(self, server: Path, model: Path, threads: int = 0) -> None:
        self.server, self.model, self.threads = server, model, threads
        self._proc: subprocess.Popen | None = None
        self._port = 0
        self._key = ""
        self._lock = threading.Lock()  # 한 번에 하나씩 요약한다 (메모리를 두 배로 쓰지 않게)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self, cancel: threading.Event | None) -> None:
        if self.running:
            return
        if not self.server.is_file():
            raise LocalLLMError("not_found", f"llama-server 가 없습니다: {self.server}. '설치' 창에서 로컬 모델을 다시 받으세요.")
        if not self.model.is_file():
            raise LocalLLMError("not_found", f"로컬 모델 파일이 없습니다: {self.model}. '설치' 창에서 다시 받으세요.")
        self._port, self._key = _free_port(), secrets.token_urlsafe(24)
        cmd = [str(self.server), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self._port),
               "--api-key", self._key, "-c", str(CONTEXT), "--jinja", "-ngl", "99"]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, **procutil.popen_group_options())
        except OSError as exc:
            raise LocalLLMError("start", f"llama-server 를 실행하지 못했습니다: {exc}") from exc
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                self.close()
                raise LocalLLMError("cancelled", "요약을 취소했습니다.")
            if self._proc.poll() is not None:
                self._proc = None
                raise LocalLLMError("start", "llama-server 가 바로 끝났습니다. 메모리가 부족하거나 모델 파일이 손상되었을 수 "
                                             "있습니다. '설치' 창에서 더 작은 모델을 고르거나 다시 받으세요.")
            if self._get("/health") == 200:
                return
            time.sleep(0.3)
        self.close()
        raise LocalLLMError("timeout", f"로컬 모델을 {LOAD_TIMEOUT_S}초 안에 올리지 못했습니다.")

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout)

    def _get(self, path: str) -> int:
        try:
            conn = self._connection(2)
            conn.request("GET", path, headers={"Authorization": f"Bearer {self._key}"})
            status = conn.getresponse().status
            conn.close()
            return status
        except OSError:
            return 0

    def chat_json(self, system: str, user: str, schema: dict, timeout_s: float = 600,
                  cancel: threading.Event | None = None) -> dict:
        """JSON 스키마에 맞는 답을 받는다 (Qwen3 의 생각 모드는 끈다)."""
        with self._lock:
            self._start(cancel)
            body = {
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.2,
                "max_tokens": 4096,
                "stream": True,
                "response_format": {"type": "json_schema", "json_schema": {"name": "summary", "schema": schema}},
                "chat_template_kwargs": {"enable_thinking": False},
            }
            text = self._stream(body, timeout_s, cancel)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise LocalLLMError("failed", "로컬 모델의 답이 JSON 형식이 아닙니다. 다시 시도하거나 더 큰 모델을 고르세요.") from exc
        if not isinstance(data, dict):
            raise LocalLLMError("failed", "로컬 모델의 답이 JSON 객체가 아닙니다.")
        return data

    def _stream(self, body: dict, timeout_s: float, cancel: threading.Event | None) -> str:
        """요청은 작업 스레드에서 보내고, 여기서는 취소·제한 시간만 본다 (멈추면 연결을 끊어 서버도 답 만들기를 멈춘다)."""
        conn = self._connection(timeout_s)
        outcome: dict = {}

        def run() -> None:
            try:
                conn.request("POST", "/v1/chat/completions", body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
                # 응답이 연결을 닫는 방식이면 http.client 가 소켓을 응답 쪽으로 넘기므로, 끊을 수 있게 미리 잡아 둔다
                outcome["sock"] = conn.sock
                response = conn.getresponse()
                if response.status != 200:
                    detail = response.read(400).decode("utf-8", "replace")
                    raise LocalLLMError("failed", f"로컬 모델 요청 실패 (HTTP {response.status}): {detail}")
                parts = []
                for raw in response:  # 줄마다 'data: {...}' (server-sent events)
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0].get("delta", {})
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
                    parts.append(delta.get("content") or "")
                outcome["text"] = "".join(parts)
            except BaseException as exc:  # noqa: BLE001 — 아래에서 알맞은 오류로 다시 올린다
                outcome["error"] = exc

        worker = threading.Thread(target=run, name="clinic-scribe-local-llm", daemon=True)
        worker.start()
        deadline = time.monotonic() + timeout_s
        while worker.is_alive():
            worker.join(0.2)
            cancelled = cancel is not None and cancel.is_set()
            if worker.is_alive() and (cancelled or time.monotonic() > deadline):
                self._abort(outcome.get("sock"), conn)
                worker.join(5)
                if cancelled:
                    raise LocalLLMError("cancelled", "요약을 취소했습니다.")
                raise LocalLLMError("timeout", "로컬 모델이 제한 시간 안에 답하지 않았습니다. 설정에서 시간을 늘리거나 "
                                               "더 작은 모델을 고르세요.")
        conn.close()
        error = outcome.get("error")
        if isinstance(error, LocalLLMError):
            raise error
        if error is not None:
            raise LocalLLMError("failed", f"로컬 모델과 연결이 끊겼습니다: {error}") from error
        return outcome.get("text", "")

    @staticmethod
    def _abort(sock: socket.socket | None, conn: http.client.HTTPConnection) -> None:
        sock = sock or conn.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)  # 다른 스레드에서 막혀 있는 읽기를 깨운다 (close 만으로는 안 깨는 OS 가 있다)
            except OSError:
                pass
        conn.close()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            procutil.terminate_tree(proc)
