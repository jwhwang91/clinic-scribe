"""말한 사람 알아보기 (이 컴퓨터 에서 처리).

등록한 의사 목소리의 특징값(임베딩)과 발화마다의 특징값을 비교해, 가까우면 그 의사 이름을, 아니면 '환자'를 붙인다.
목소리 특징값은 생체인식정보다. 본인 동의를 받은 의사만 등록하고, 이 컴퓨터의 설정 폴더(voices.json)에만 저장한다.
환자 목소리는 비교에만 쓰고 저장하지 않는다.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from clinic_scribe.config import native_path, setup_command
from clinic_scribe.whisper import TranscriptSegment

PATIENT = "환자"  # 등록되지 않은 목소리 (대개 환자이고 보호자일 수도 있다)
SAMPLE_RATE = 16000
MIN_SEGMENT_S = 1.0  # 이보다 짧은 발화("네", "음")는 목소리로 판단하지 않는다
ENROLL_CHUNK_S = 4.0  # 등록 음성을 이 길이로 잘라 특징값을 평균한다
MIN_ENROLL_S = 10.0  # 등록에 필요한 최소 말소리 길이


class SpeakerError(Exception):
    pass


@dataclass
class Voice:
    name: str
    embedding: list[float]  # 정규화한 평균 특징값
    seconds: float  # 등록에 쓴 말소리 길이
    created_at: str
    specialty: str = ""  # 진료과 (따라 읽기 문장을 고를 때 쓴다)


def _normalize(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0.0 or not np.isfinite(norm):
        raise SpeakerError("목소리 특징값을 구하지 못했습니다.")
    return v / norm


def clean_name(name: str) -> str:
    name = " ".join(name.replace("<", "").replace(">", "").split())[:20]
    if not name:
        raise SpeakerError("이름을 입력하세요.")
    if name == PATIENT:
        raise SpeakerError(f"'{PATIENT}'는 등록되지 않은 목소리에 쓰는 이름이라 쓸 수 없습니다.")
    return name


class VoiceBook:
    """등록한 의사 목소리 목록 (voices.json)."""

    def __init__(self, path: Path, voices: list[Voice] | None = None) -> None:
        self.path = path
        self.voices: list[Voice] = voices or []

    @classmethod
    def load(cls, path: Path) -> VoiceBook:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            voices = [Voice(**item) for item in raw.get("voices", [])]
        except FileNotFoundError:
            voices = []
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise SpeakerError(f"등록된 목소리 파일을 읽지 못했습니다 ({path}): {exc}") from exc
        return cls(path, voices)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"voices": [asdict(v) for v in self.voices]}, ensure_ascii=False), encoding="utf-8")
        os.chmod(tmp, 0o600)  # 생체인식정보: 이 사용자만 읽게
        tmp.replace(self.path)

    def copy(self) -> VoiceBook:
        """작업 스레드에 넘길 사본 (작업 중에 등록·삭제가 바뀌어도 안전하게)."""
        return VoiceBook(self.path, list(self.voices))

    def names(self) -> list[str]:
        return [v.name for v in self.voices]

    def specialty(self, name: str) -> str:
        """등록한 의사의 진료과 (모르면 빈 글)."""
        return next((v.specialty for v in self.voices if v.name == name), "")

    def enroll(self, name: str, embeddings: list[np.ndarray], seconds: float, specialty: str = "") -> Voice:
        if not embeddings:
            raise SpeakerError("등록할 말소리가 없습니다.")
        name = clean_name(name)
        mean = _normalize(np.mean([_normalize(e) for e in embeddings], axis=0))
        voice = Voice(name, [round(float(x), 6) for x in mean], round(seconds, 1), datetime.now().strftime("%Y-%m-%d %H:%M"),
                      specialty)
        self.voices = [v for v in self.voices if v.name != name] + [voice]  # 같은 이름은 다시 등록
        return voice

    def remove(self, name: str) -> None:
        self.voices = [v for v in self.voices if v.name != name]

    def rename(self, old: str, new: str) -> str:
        """등록한 이름을 바꾼다 (목소리 특징값은 그대로). 바꾼 이름을 돌려준다."""
        new = clean_name(new)
        if new != old and new in self.names():
            raise SpeakerError(f"'{new}' 목소리가 이미 있습니다. 다른 이름을 쓰거나 그 목소리를 먼저 삭제하세요.")
        # 사본(작업 스레드가 쓰는 중일 수 있음)과 Voice 를 나눠 쓰므로 고치지 않고 바꿔 끼운다
        self.voices = [replace(v, name=new) if v.name == old else v for v in self.voices]
        return new

    def identify(self, embedding, threshold: float) -> tuple[str | None, float]:
        """가장 가까운 등록 목소리의 이름과 유사도. 유사도가 threshold 미만이면 이름 대신 None."""
        if not self.voices:
            return None, 0.0
        v = _normalize(embedding)
        score, name = max((float(v @ np.asarray(voice.embedding, dtype=np.float32)), voice.name) for voice in self.voices)
        return (name if score >= threshold else None), score


class SpeakerEncoder:
    """목소리 특징값 계산 (sherpa-onnx). 여러 작업 스레드에서 같이 쓰도록 잠근다."""

    def __init__(self, model_path: Path, num_threads: int = 2) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise SpeakerError(f"sherpa-onnx 가 설치되지 않았습니다. {setup_command()} 를 다시 실행하세요.") from exc
        if not model_path.is_file():
            raise SpeakerError(f"화자 인식 모델이 없습니다: {model_path}\n{setup_command()} 를 다시 실행하세요.")
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(  # Windows 한글 경로는 짧은 경로로
            model=native_path(model_path), num_threads=num_threads, debug=False, provider="cpu"
        )
        if not config.validate():
            raise SpeakerError(f"화자 인식 모델을 쓸 수 없습니다: {model_path}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._lock = threading.Lock()

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """16 kHz 모노 소리(int16 또는 -1~1 float32) → 정규화한 특징값."""
        audio = samples.astype(np.float32) / 32768.0 if samples.dtype == np.int16 else samples.astype(np.float32)
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=audio)
            stream.input_finished()
            return _normalize(self._extractor.compute(stream))


def speech_only(pcm16: np.ndarray, frame_s: float = 0.03) -> np.ndarray:
    """등록용: 조용한 부분을 빼고 말소리만 이어 붙인다(float32). 배경 소음보다 10 dB 이상 큰 소리를 말소리로 본다."""
    frame = int(frame_s * SAMPLE_RATE)
    count = len(pcm16) // frame
    if count == 0:
        return np.zeros(0, np.float32)
    frames = pcm16[: count * frame].astype(np.float32).reshape(count, frame)
    if pcm16.dtype == np.int16:
        frames /= 32768.0
    db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    voiced = db > max(float(np.percentile(db, 20)) + 10.0, -50.0)
    keep = voiced.copy()
    gap = int(0.2 / frame_s)  # 말 사이 0.2초 이하 끊김은 잇는다
    last = None
    for i in np.flatnonzero(voiced):
        if last is not None and 0 < i - last - 1 <= gap:
            keep[last + 1 : i] = True
        last = i
    return frames[keep].reshape(-1)


def enrollment_embeddings(encoder: SpeakerEncoder, speech: np.ndarray) -> list[np.ndarray]:
    """등록용 말소리를 ENROLL_CHUNK_S 조각으로 나눠 조각마다 특징값을 구한다 (평균은 VoiceBook.enroll)."""
    step = int(ENROLL_CHUNK_S * SAMPLE_RATE)
    chunks = [speech[i : i + step] for i in range(0, len(speech), step)]
    chunks = [c for c in chunks if len(c) >= step // 2] or [speech]
    return [encoder.embed(c) for c in chunks]


def label_segments(segments: list[TranscriptSegment], pcm16: np.ndarray, encoder: SpeakerEncoder, book: VoiceBook,
                   threshold: float, cancel: threading.Event | None = None) -> list[TranscriptSegment]:
    """발화마다 말한 사람을 붙인다: 등록 의사 이름 / PATIENT / None(짧거나 음성 미감지라 판단 안 함)."""
    out = []
    for seg in segments:
        if cancel is not None and cancel.is_set():
            return segments
        speaker = None
        start, end = max(0, int(seg.start * SAMPLE_RATE)), min(len(pcm16), int(seg.end * SAMPLE_RATE))
        if seg.speech_detected and seg.end - seg.start >= MIN_SEGMENT_S and end - start >= int(0.8 * MIN_SEGMENT_S * SAMPLE_RATE):
            name, _score = book.identify(encoder.embed(pcm16[start:end]), threshold)
            speaker = name or PATIENT
        out.append(replace(seg, speaker=speaker))
    return out


def main_doctor(segments: list[TranscriptSegment]) -> str | None:
    """가장 오래 말한 등록 의사 (진료 기록에 표시)."""
    talk: dict[str, float] = {}
    for seg in segments:
        if seg.speaker and seg.speaker != PATIENT:
            talk[seg.speaker] = talk.get(seg.speaker, 0.0) + seg.end - seg.start
    return max(talk, key=talk.__getitem__) if talk else None
