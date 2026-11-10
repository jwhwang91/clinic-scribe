"""진료 세션: 한 번의 녹음(또는 불러온 파일)과 그 전사·요약 결과.

새 세션을 시작하면 이전 세션 객체는 버려진다. 백그라운드 작업 결과는 세션 번호가 맞을 때만 반영한다.
녹음은 메모리(16 kHz mono)에만 두고, 사용자가 내보내기를 고를 때만 파일로 저장한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from clinic_scribe.chart import ChartDraft
from clinic_scribe.live_monitor import LiveClip
from clinic_scribe.whisper import TranscriptionResult


@dataclass
class TranscriptRun:
    run_id: int
    result: TranscriptionResult

    @property
    def label(self) -> str:
        return f"전사 #{self.run_id} · {self.result.describe()}"


@dataclass
class Session:
    session_id: int
    source: str
    created_at: datetime = field(default_factory=datetime.now)
    pcm16: np.ndarray | None = None  # 16 kHz mono, 전사 직전 변환본
    original_rate: int = 0
    original_channels: int = 0
    runs: list[TranscriptRun] = field(default_factory=list)
    draft: ChartDraft | None = None
    draft_run_id: int | None = None
    summary_error: str | None = None
    summary_note: str | None = None  # 요약을 요청하지 않은 이유 등 안내
    live_clips: list[LiveClip] = field(default_factory=list)
    exported: bool = False
    _next_run: int = 1

    def new_run_id(self) -> int:
        run_id = self._next_run
        self._next_run += 1
        return run_id

    def run(self, run_id: int | None) -> TranscriptRun | None:
        return next((r for r in self.runs if r.run_id == run_id), None)

    @property
    def has_results(self) -> bool:
        return bool(self.runs) or self.draft is not None
