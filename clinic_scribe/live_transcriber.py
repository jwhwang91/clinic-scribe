"""녹음 중 실시간 전사 미리보기.

발화 클립이 닫힐 때마다(무음 또는 최대 길이) 그 클립만 whisper.cpp 로 전사해 바로 화면에 보낸다.
클립 하나만 보므로 앞뒤 문맥이 없어 최종 전사보다 부정확할 수 있다. 녹음이 끝나면 전체 녹음을 다시 전사한다.
모든 처리는 이 Mac 에서 하며 결과는 화면에만 쓴다(로그 없음). 작업은 한 줄로 순서대로 처리한다.
"""

from __future__ import annotations

import dataclasses
import queue
import threading

import numpy as np
from PySide6.QtCore import QObject, Signal

from clinic_scribe import procutil
from clinic_scribe.audio import to_whisper_pcm
from clinic_scribe.config import VadParams
from clinic_scribe.whisper import WhisperRunner


class LiveTranscriber(QObject):
    # (세션 번호, 클립 번호, 전사 구간 목록 또는 None, 오류 메시지 또는 None)
    finished = Signal(int, int, object, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="clinic-scribe-live", daemon=True)
        self._thread.start()

    def submit(self, session_id: int, clip_id: int, offset_s: float, audio: np.ndarray, samplerate: int,
               runner: WhisperRunner, params: VadParams) -> None:
        """audio 는 녹음 원본의 일부 (offset_s 부터). 결과 시각은 원본 녹음 기준으로 돌려준다."""
        self._queue.put((session_id, clip_id, offset_s, audio, samplerate, runner, params, self._cancel))

    def cancel_all(self) -> None:
        """대기 중인 작업을 버리고 진행 중인 whisper-cli 를 멈춘다."""
        self._cancel.set()
        self._cancel = threading.Event()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self.cancel_all()
        self._queue.put(None)

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            session_id, clip_id, offset, audio, rate, runner, params, cancel = job
            if cancel.is_set():
                continue
            try:
                pcm = to_whisper_pcm(audio, rate)
                result = runner.transcribe(pcm, use_vad=True, params=params, cancel=cancel)
                segments = [dataclasses.replace(s, start=s.start + offset, end=s.end + offset) for s in result.segments]
            except procutil.Cancelled:
                continue
            except Exception as exc:  # 미리보기 실패는 화면에 한 줄로만 알린다
                self._emit(session_id, clip_id, None, str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
                continue
            if not cancel.is_set():
                self._emit(session_id, clip_id, segments, None)

    def _emit(self, *args) -> None:
        try:
            self.finished.emit(*args)
        except RuntimeError:
            pass  # 앱 종료 중
