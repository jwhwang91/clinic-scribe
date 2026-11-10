"""실험 모드: 녹음 중 발화 시작·무음 종료를 실시간으로 표시한다.

에너지(입력 레벨) 기반이며, 최근 몇 초의 잡음 수준보다 충분히 커진 소리를 발화로 본다.
조용해져서 클립을 닫아도 진료 세션(녹음)은 사용자가 종료 버튼을 누를 때까지 계속된다.
최종 음성 구간은 녹음이 끝난 뒤 Silero VAD 로 다시 계산한다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from clinic_scribe.audio import rms_dbfs


@dataclass
class LiveClip:
    start: float
    end: float | None = None


class LiveActivityMonitor:
    def __init__(self, min_speech_ms: int = 150, min_silence_ms: int = 600, margin_db: float = 10.0,
                 min_threshold_db: float = -50.0) -> None:
        self.min_speech_s = min_speech_ms / 1000.0
        self.min_silence_s = min_silence_ms / 1000.0
        self.margin_db = margin_db
        self.min_threshold_db = min_threshold_db
        self.reset()

    def reset(self) -> None:
        self.time = 0.0
        self.clips: list[LiveClip] = []
        self.in_speech = False
        self._levels: deque[float] = deque(maxlen=400)  # 약 4~8초
        self._above_since: float | None = None
        self._below_since: float | None = None

    @property
    def threshold_db(self) -> float:
        if not self._levels:
            return self.min_threshold_db
        noise = float(np.percentile(np.fromiter(self._levels, float), 20))
        return max(noise + self.margin_db, self.min_threshold_db)

    def feed(self, block: np.ndarray, samplerate: int) -> list[str]:
        """mono float 블록을 넣는다. 'start'(발화 시작) / 'end'(무음으로 클립 종료) 이벤트를 돌려준다."""
        if block.size == 0 or samplerate <= 0:
            return []
        level = rms_dbfs(block)
        threshold = self.threshold_db
        t0 = self.time
        self.time += len(block) / samplerate
        self._levels.append(level)
        events: list[str] = []
        if level >= threshold:
            self._below_since = None
            if self._above_since is None:
                self._above_since = t0
            if not self.in_speech and self.time - self._above_since >= self.min_speech_s:
                self.in_speech = True
                self.clips.append(LiveClip(start=self._above_since))
                events.append("start")
        else:
            self._above_since = None
            if self.in_speech:
                if self._below_since is None:
                    self._below_since = t0
                if self.time - self._below_since >= self.min_silence_s:
                    self.in_speech = False
                    self.clips[-1].end = self._below_since
                    events.append("end")
        return events
