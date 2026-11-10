"""백그라운드 작업: 전사·요약을 스레드에서 돌리고 결과를 UI 스레드로 돌려준다."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class Job:
    kind: str  # "transcribe" | "summary"
    session_id: int
    run_id: int
    auto_summary: bool = False


class Task:
    def __init__(self, job: Job) -> None:
        self.job = job
        self.cancel = threading.Event()
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


class TaskRunner(QObject):
    # (Task, 결과, 예외 또는 None). 다른 스레드에서 emit 하므로 UI 쪽 슬롯은 큐 연결로 실행된다.
    finished = Signal(object, object, object)

    def start(self, job: Job, fn: Callable[[threading.Event], Any]) -> Task:
        task = Task(job)

        def target() -> None:
            try:
                result = fn(task.cancel)
            except BaseException as exc:  # 결과는 UI 에서 메시지로만 보여 준다
                self._emit(task, None, exc)
            else:
                self._emit(task, result, None)

        threading.Thread(target=target, name=f"clinic-scribe-{job.kind}", daemon=True).start()
        return task

    def _emit(self, task: Task, result: Any, error: BaseException | None) -> None:
        try:
            self.finished.emit(task, result, error)
        except RuntimeError:
            pass  # 앱 종료 중 QObject 가 이미 사라짐
