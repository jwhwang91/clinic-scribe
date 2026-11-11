"""진료 세션: 한 진료(녹음 구간 또는 불러온 파일)와 그 전사·요약 결과.

앱이 켜져 있는 동안 진료들은 목록으로 쌓인다. 자동 진료 나누기로 다음 진료를 녹음하는 동안에도 앞 진료의
전사·요약은 그 진료 객체에만 반영되므로 진료끼리 결과가 섞이지 않는다.
녹음은 메모리(16 kHz mono)에만 두고, 사용자가 내보내기를 고를 때만 파일로 저장한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from clinic_scribe.chart import ChartDraft
from clinic_scribe.conversation_summary import ConversationSummary
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
    draft: ChartDraft | ConversationSummary | None = None  # 진료 차트 초안 또는 일반 대화 요약
    draft_run_id: int | None = None
    summary_error: str | None = None
    summary_note: str | None = None  # 요약을 요청하지 않은 이유 등 안내
    summary_short: str | None = None  # 진행 단계에 쓰는 짧은 안내
    live_clips: list[LiveClip] = field(default_factory=list)
    exported: bool = False
    # 녹음 진료 정보
    offset_s: float = 0.0  # 연속 녹음 안에서 이 진료가 시작한 위치(초)
    duration_s: float = 0.0
    end_reason: str = ""  # 스페이스바 / 무음 90초 / 다음 환자 신호 '들어오세요' / 끝인사 '안녕히 가세요'
    auto_continued: bool = False  # 자동 나누기로 이어서 시작된 진료
    transcribe_error: str | None = None
    transcribe_seconds: float = 0.0
    summary_seconds: float = 0.0
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
