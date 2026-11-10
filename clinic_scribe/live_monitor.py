"""녹음 중 발화 시작·무음 종료를 실시간으로 찾아 클립으로 나눈다 (실시간 전사 미리보기용).

에너지(입력 레벨) 기반이며, 최근 몇 초의 잡음 수준보다 충분히 커진 소리를 발화로 본다.
조용해져서 클립을 닫아도 진료 세션(녹음)은 사용자가 종료 버튼을 누를 때까지 계속된다.
쉬지 않고 길게 말하면 max_clip_s 마다 클립을 나눠 미리보기가 너무 늦어지지 않게 한다.
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
                 min_threshold_db: float = -50.0, max_threshold_db: float = -30.0, max_clip_s: float = 10.0) -> None:
        self.min_speech_s = min_speech_ms / 1000.0
        self.min_silence_s = min_silence_ms / 1000.0
        self.margin_db = margin_db
        self.min_threshold_db = min_threshold_db
        # 녹음 시작부터 쉬지 않고 말하면 잡음 수준이 말소리 크기로 잡힌다. 그래도 보통 말소리(-30 dBFS 이상)는 발화로 본다.
        self.max_threshold_db = max_threshold_db
        self.max_clip_s = max_clip_s
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
        return min(max(noise + self.margin_db, self.min_threshold_db), self.max_threshold_db)

    def feed(self, block: np.ndarray, samplerate: int) -> list[str]:
        """mono float 블록을 넣는다. 이벤트를 돌려준다:
        'start' 발화 시작(새 클립), 'end' 무음으로 클립 종료, 'split' 긴 발화를 나눔(앞 클립 종료 + 새 클립 시작).
        """
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
            elif self.in_speech and self.max_clip_s > 0 and self.time - self.clips[-1].start >= self.max_clip_s:
                self.clips[-1].end = self.time
                self.clips.append(LiveClip(start=self.time))
                events.append("split")
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
