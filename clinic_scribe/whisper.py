"""whisper.cpp 호출과 결과 해석 (Homebrew whisper.cpp 1.9.5 의 whisper-cli / whisper-vad-speech-segments 기준).

전사:   whisper-cli -m MODEL -f in.wav -l ko -oj -ojf -of OUT [--prompt P] [--vad -vm VAD -vt -vspd -vsd -vp -vo]
        - 전사문이 표준출력에 찍히므로 표준출력은 버린다(콘솔에 진료 내용을 남기지 않음).
        - --vad 로 전사하면 whisper.cpp 가 음성 구간만 이어 붙여 전사하고 보간표로 원본 시각을 되돌린다.
          이음매 근처 시각은 무음 쪽으로 밀린다(예: 13.3초에 시작한 발화가 6.5초 시작으로 나옴).
          그래서 각 전사 구간을 겹침이 가장 큰 음성 구간에 붙이고 그 구간 경계로 시각을 맞춘다.
        - 구간마다 따로 잘라 전사하는 방식도 시험했으나 앞뒤 문맥이 끊겨 오인식이 늘었다
          ("변이 검게" → "편이 검게"). 그래서 whisper.cpp 의 VAD 전사를 그대로 쓰고 시각만 맞춘다.
음성 구간: whisper-vad-speech-segments 를 같은 VAD 설정으로 실행한다.
        출력 예: "Speech segment 0: start = 66.00, end = 474.00" (센티초, 원본 녹음 기준)
VAD 를 끄고 전사할 때도 음성 구간을 구해, 음성이 없는 곳에서 나온 전사(무음에서의 Whisper 환각,
예: "자막 제공 및 광고를 포함하고 있습니다")를 표시한다.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from clinic_scribe import procutil
from clinic_scribe.audio import WHISPER_SR, duration_s, write_wav
from clinic_scribe.config import VadParams

SEGMENT_LINE = re.compile(r"Speech segment\s+\d+:\s*start\s*=\s*([\d.]+),\s*end\s*=\s*([\d.]+)")
LOW_TOKEN_PROB = 0.5


class WhisperError(Exception):
    pass


@dataclass
class SpeechRegion:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float | None = None  # 토큰 확률 평균
    region: int | None = None  # 소속 음성 구간 번호
    last_region: int | None = None  # 여러 구간에 걸쳐 한 문장으로 전사된 경우 마지막 구간 번호
    speech_detected: bool = True  # False: VAD 가 음성을 찾지 못한 곳의 전사(환각 의심)
    uncertain_terms: list[str] = field(default_factory=list)  # 확률이 낮은 숫자·영문 단어
    speaker: str | None = None  # 말한 사람: 등록한 의사 이름 / "환자"(등록되지 않은 목소리) / None(판단 안 함)
    original_text: str | None = None  # 딕션 사전으로 고치기 전 Whisper 가 받아 적은 그대로 (고친 줄만)
    corrections: list[str] = field(default_factory=list)  # 적용한 교정 ("잘못 들린 말 → 바른 말")


@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment]
    regions: list[SpeechRegion] | None  # None: VAD 모델/도구가 없어 구간을 구하지 못함
    vad_enabled: bool
    vad_params: VadParams
    audio_duration: float
    elapsed: float
    doctor: str | None = None  # 이 진료 의사로 보고 그 의사의 딕션 사전(용어 목록)을 쓴 경우

    @property
    def is_empty(self) -> bool:
        return not any(s.text.strip() for s in self.segments)

    @property
    def no_speech_detected(self) -> bool:
        return self.regions is not None and len(self.regions) == 0

    @property
    def suspicious_segments(self) -> list[TranscriptSegment]:
        return [s for s in self.segments if not s.speech_detected]

    def describe(self) -> str:
        if self.vad_enabled:
            return f"VAD 켬 · {self.vad_params.describe()}"
        return "VAD 끔 · 녹음 전체 전사"

    def region_text(self, index: int) -> str:
        own = [s.text for s in self.segments if s.region == index]
        if own:
            return " ".join(own)
        if any(s.region is not None and s.last_region is not None and s.region < index <= s.last_region
               for s in self.segments):
            return "(앞 구간과 이어진 한 문장으로 전사됨)"
        return ""


@dataclass
class WhisperConfig:
    cli: str | None
    model: Path
    vad_model: Path | None
    language: str = "ko"
    prompt: str = ""
    threads: int = 0


def find_vad_tool(cli: str | None) -> str | None:
    """whisper-cli 와 같은 곳에 설치된 whisper-vad-speech-segments 를 찾는다."""
    if cli:
        p = Path(cli)
        for candidate in (p.with_name("whisper-vad-speech-segments"), p.resolve().with_name("whisper-vad-speech-segments")):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return shutil.which("whisper-vad-speech-segments")


def vad_args(params: VadParams) -> list[str]:
    args = [
        "-vt", f"{params.threshold:.2f}",
        "-vspd", str(int(params.min_speech_ms)),
        "-vsd", str(int(params.min_silence_ms)),
        "-vp", str(int(params.speech_pad_ms)),
        "-vo", f"{params.samples_overlap_s:.2f}",
    ]
    if params.max_speech_s > 0:
        args += ["-vmsd", f"{params.max_speech_s:.1f}"]
    return args


def default_threads() -> int:
    return max(4, min(8, (os.cpu_count() or 8) // 2))


def parse_vad_output(text: str) -> list[SpeechRegion]:
    """whisper-vad-speech-segments 표준출력(센티초) → 초 단위 구간."""
    return [SpeechRegion(float(a) / 100.0, float(b) / 100.0) for a, b in SEGMENT_LINE.findall(text)]


def parse_whisper_json(raw: bytes) -> list[TranscriptSegment]:
    """whisper-cli -oj -ojf 출력 해석. 오프셋은 ms, 토큰 p 는 확률."""
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise WhisperError("whisper-cli 의 JSON 출력을 해석하지 못했습니다.") from exc
    segments: list[TranscriptSegment] = []
    for item in data.get("transcription") or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        offsets = item.get("offsets") or {}
        start = float(offsets.get("from", 0)) / 1000.0
        end = float(offsets.get("to", 0)) / 1000.0
        probs: list[float] = []
        low_tokens: list[str] = []
        for token in item.get("tokens") or []:
            token_text = token.get("text") or ""
            p = token.get("p")
            if token_text.startswith("[_") or not isinstance(p, (int, float)):
                continue  # [_BEG_], [_TT_123] 같은 특수 토큰
            probs.append(float(p))
            if p < LOW_TOKEN_PROB and re.search(r"[0-9A-Za-z]", token_text):
                low_tokens.append(token_text.strip())
        segments.append(
            TranscriptSegment(
                start=start,
                end=max(start, end),
                text=text,
                confidence=sum(probs) / len(probs) if probs else None,
                uncertain_terms=_words_containing(text, low_tokens),
            )
        )
    return segments


def _words_containing(text: str, tokens: list[str]) -> list[str]:
    words = text.split()
    found: list[str] = []
    for token in tokens:
        word = next((w for w in words if token and token in w), token)
        word = word.strip(".,?!")
        if word and word not in found:
            found.append(word)
    return found


def assign_regions(segments: list[TranscriptSegment], regions: list[SpeechRegion] | None, vad_enabled: bool) -> None:
    """전사 구간을 음성 구간에 연결한다.

    VAD 켬: 겹침이 큰 음성 구간에 붙이고 그 경계로 시각을 맞춘다. 두 구간에 걸친 전사는 두 구간을 모두 덮는다.
    VAD 끔: 음성 구간과 겹치는 길이가 짧으면 음성 미감지(환각 의심)로 표시한다.
    """
    if regions is None:
        return
    if not regions:
        for seg in segments:
            seg.region = None
            seg.speech_detected = False
        return
    for seg in segments:
        overlaps = [max(0.0, min(seg.end, r.end) - max(seg.start, r.start)) for r in regions]
        touched = [i for i, ov in enumerate(overlaps) if ov > 0 and ov >= min(0.3, 0.3 * regions[i].duration)]
        if vad_enabled:
            if not touched:
                centre = (seg.start + seg.end) / 2
                touched = [min(range(len(regions)), key=lambda i: abs(centre - (regions[i].start + regions[i].end) / 2))]
            first, last = regions[touched[0]], regions[touched[-1]]
            seg.region, seg.last_region = touched[0], touched[-1]
            seg.start = min(max(seg.start, first.start), first.end)
            seg.end = max(min(seg.end, last.end), seg.start)
            seg.speech_detected = True
        else:
            if not touched:
                best = max(range(len(regions)), key=lambda i: overlaps[i])
                touched = [best] if overlaps[best] > 0 else []
            seg.region = touched[0] if touched else None
            seg.last_region = touched[-1] if touched else None
            seg.speech_detected = sum(overlaps) >= min(0.3, 0.3 * max(0.01, seg.end - seg.start))


class WhisperRunner:
    def __init__(self, config: WhisperConfig):
        self.config = config
        self.vad_tool = find_vad_tool(config.cli)

    def problems(self, need_vad: bool) -> list[str]:
        issues = []
        if not self.config.cli:
            issues.append("whisper-cli 를 찾지 못했습니다. ./scripts/setup_macos.sh 를 실행하거나 설정에서 경로를 지정하세요.")
        if not self.config.model.is_file():
            issues.append(f"Whisper 모델이 없습니다: {self.config.model}")
        if need_vad and not (self.config.vad_model and self.config.vad_model.is_file()):
            issues.append(f"VAD 모델이 없습니다: {self.config.vad_model}")
        return issues

    def can_detect_regions(self) -> bool:
        return bool(self.vad_tool and self.config.vad_model and self.config.vad_model.is_file())

    def detect_regions(self, wav: Path, params: VadParams, cancel: threading.Event | None = None) -> list[SpeechRegion]:
        cmd = [self.vad_tool, "-f", str(wav), "-vm", str(self.config.vad_model), *vad_args(params), "-np"]
        result = procutil.run(cmd, cancel=cancel, timeout=600)
        if result.returncode != 0:
            raise WhisperError(f"음성 구간 탐지 실패 (종료 코드 {result.returncode}):\n{procutil.tail(result.stderr)}")
        return parse_vad_output(result.stdout.decode("utf-8", errors="replace"))

    def transcribe(
        self,
        pcm16: np.ndarray,
        *,
        use_vad: bool,
        params: VadParams,
        cancel: threading.Event | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """prompt 를 주면 설정의 힌트 프롬프트 대신 쓴다 (의사별 용어 목록을 넣은 힌트)."""
        issues = self.problems(need_vad=use_vad)
        if issues:
            raise WhisperError("\n".join(issues))
        started = time.monotonic()
        # 임시 WAV 는 이 함수가 끝나면 (성공·실패·취소 모두) 지운다.
        tmp = Path(tempfile.mkdtemp(prefix="clinic-scribe-"))
        try:
            wav = tmp / "input.wav"
            write_wav(wav, pcm16, WHISPER_SR)
            regions = self.detect_regions(wav, params, cancel) if self.can_detect_regions() else None
            if use_vad and regions == []:
                segments: list[TranscriptSegment] = []  # 음성 없음: Whisper 를 돌리지 않는다
            else:
                segments = self._run_whisper(wav, tmp / "out", use_vad, params, cancel, prompt)
            assign_regions(segments, regions, use_vad)
            return TranscriptionResult(
                segments=segments,
                regions=regions,
                vad_enabled=use_vad,
                vad_params=dataclasses.replace(params),
                audio_duration=duration_s(pcm16),
                elapsed=time.monotonic() - started,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _run_whisper(
        self, wav: Path, out_base: Path, use_vad: bool, params: VadParams, cancel: threading.Event | None,
        prompt: str | None = None,
    ) -> list[TranscriptSegment]:
        cfg = self.config
        cmd = [
            cfg.cli, "-m", str(cfg.model), "-f", str(wav),
            "-l", cfg.language or "ko",
            "-t", str(cfg.threads or default_threads()),
            "-oj", "-ojf", "-of", str(out_base),
        ]
        hint = (cfg.prompt if prompt is None else prompt).strip()
        if hint:
            cmd += ["--prompt", hint]
        if use_vad:
            cmd += ["--vad", "-vm", str(cfg.vad_model), *vad_args(params)]
        result = procutil.run(cmd, capture_stdout=False, cancel=cancel)
        if result.returncode != 0:
            raise WhisperError(f"whisper-cli 실패 (종료 코드 {result.returncode}):\n{procutil.tail(result.stderr)}")
        out_json = out_base.with_suffix(".json")
        if not out_json.is_file():
            raise WhisperError("whisper-cli 가 결과 파일을 만들지 않았습니다.")
        return parse_whisper_json(out_json.read_bytes())


def format_time(seconds: float) -> str:
    """원본 녹음 기준 시각 표시: 01:05.3 (1시간 이상이면 1:01:05.3)."""
    seconds = round(max(0.0, seconds), 1)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes):02d}:{secs:04.1f}"
