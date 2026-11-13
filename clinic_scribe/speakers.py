"""말한 사람 알아보기 (이 컴퓨터 에서 처리).

등록한 의사 목소리의 특징값(임베딩)과 발화마다의 특징값을 비교해, 가까우면 그 의사 이름을, 아니면 '환자'를 붙인다.
쉼 없이 끼어든 말(환자 말 도중의 의사 질문)은 말 덩어리 안을 짧은 창으로 비교해 사람이 바뀌는 낱말 경계에서 나눈다(split_turns).
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
from clinic_scribe.whisper import TranscriptSegment, ends_sentence, phrase_from_words

PATIENT = "환자"  # 등록되지 않은 목소리 (대개 환자이고 보호자일 수도 있다)
SAMPLE_RATE = 16000
MIN_SEGMENT_S = 1.0  # 이보다 짧은 발화("네", "음")는 목소리로 판단하지 않는다
# 쉼 없이 끼어든 말: 말 덩어리 안을 짧은 창으로 옮겨 가며 목소리를 비교해 말한 사람이 바뀌는 곳을 찾는다
TURN_WINDOW_S = 1.0  # 한 번에 비교하는 길이
TURN_HOP_S = 0.5  # 창을 옮기는 간격
MIN_TURN_S = 0.6  # 이보다 짧게 끼어든 말("네?")은 나누지 않는다 (목소리로 가리기 어렵다)
TURN_SHIFT_WORDS = 2  # 나누는 자리를 앞뒤로 이만큼 낱말을 옮겨 보며 두 사람 목소리가 가장 뚜렷이 갈리는 곳을 고른다
TURN_SENTENCE_BONUS = 0.1  # 목소리 차이가 이만큼(짧은 창의 흔들림 정도) 안이면 문장이 끝난 자리에서 나눈다 (짧은 첫 낱말·겹쳐 말한 곳)
QUESTION_WORDS = 4  # 짧게 끼어든 질문: '?' 낱말과 그 앞 낱말 몇 개까지를 한 구간으로 비교해 본다
# 목소리로 고른 나눌 자리가 문장 끝에서 이만큼 안이면 문장 끝에서 나눈다: 이보다 짧은 낱말(0.2초 "등")은 목소리로 누구 말인지
# 가를 수 없고, Whisper 낱말 시각은 빌드(CPU)마다 0.2초쯤 다르다 (Windows: "아파서요. 등 / 어느 쪽이요?").
# 문장을 끝낸 뒤 이어 한 말("그리고", 0.4초 넘음)은 그대로 둔다
SENTENCE_SNAP_S = 0.4
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

    def similarity(self, embedding) -> np.ndarray:
        """등록한 목소리마다의 유사도 (voices 순서)."""
        voices = np.asarray([v.embedding for v in self.voices], dtype=np.float32).reshape(len(self.voices), -1)
        return voices @ _normalize(embedding)

    def identify(self, embedding, threshold: float) -> tuple[str | None, float]:
        """가장 가까운 등록 목소리의 이름과 유사도. 유사도가 threshold 미만이면 이름 대신 None."""
        if not self.voices:
            return None, 0.0
        scores = self.similarity(embedding)
        best = int(np.argmax(scores))
        score = float(scores[best])
        return (self.voices[best].name if score >= threshold else None), score


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


def split_turns(phrase: TranscriptSegment, pcm16: np.ndarray, encoder: SpeakerEncoder, book: VoiceBook,
                threshold: float) -> list[TranscriptSegment]:
    """말 덩어리 안에서 말한 사람이 바뀌면(환자 말 도중에 쉼 없이 끼어든 의사 질문 등) 그 자리에서 나눈다.

    1) TURN_WINDOW_S 창을 옮겨 가며 등록 목소리와 비교해 낱말마다 말한 사람을 정하고,
    2) 너무 짧은 묶음은 이웃에 붙인 뒤, 나누는 자리를 앞뒤 낱말로 옮겨 보며 양쪽 목소리가 가장 뚜렷이 갈리는 곳을 고르고,
    3) 나눈 묶음 전체로 다시 비교해 같은 사람으로 나온 이웃은 도로 잇는다.
    한 사람만 말했으면 [phrase] 를 그대로 돌려준다. 겹쳐 말한 부분은 한 사람 말로 들어간다(마이크 하나로는 가를 수 없다).
    """
    words = [w for w in phrase.words if w.text.strip()]
    if len(words) < 2 or not phrase.speech_detected or not book.voices or phrase.end - phrase.start < 2 * MIN_TURN_S:
        return [phrase]
    names = book.names()

    def scores(start: float, end: float) -> np.ndarray:
        a, b = max(0, int(start * SAMPLE_RATE)), min(len(pcm16), int(end * SAMPLE_RATE))
        return book.similarity(encoder.embed(pcm16[a:b]))

    def label(s: np.ndarray) -> str:
        best = int(np.argmax(s))
        return names[best] if s[best] >= threshold else PATIENT

    # 낱말 사이에서 나눌 시각 (cuts[i]: i 번째 낱말 뒤)
    cuts, prev = [], phrase.start
    for left, right in zip(words, words[1:]):
        prev = min(max((left.end + right.start) / 2, prev), phrase.end)
        cuts.append(prev)

    def span(first: int, last: int) -> tuple[float, float]:
        return (phrase.start if first == 0 else cuts[first - 1]), (phrase.end if last == len(words) - 1 else cuts[last])

    # 1) 창마다 비교하고, 낱말은 가운데가 가장 가까운 창의 사람으로
    width = min(TURN_WINDOW_S, phrase.end - phrase.start)
    starts = list(np.arange(phrase.start, phrase.end - width + 1e-6, TURN_HOP_S))
    if phrase.end - width - starts[-1] > 0.05:
        starts.append(phrase.end - width)
    windows = [label(scores(t, t + width)) for t in starts]
    centres = np.asarray(starts) + width / 2
    who = [windows[int(np.argmin(np.abs(centres - (w.start + w.end) / 2)))] for w in words]
    # 환자 말 사이에 짧게 끼어든 질문("설사요?")은 창보다 짧아 창으로는 놓친다.
    # '?'로 끝나는 낱말에서 끝나는 짧은 구간을 직접 비교해 등록 목소리면 그 사람 말로 둔다.
    for i, w in enumerate(words):
        if who[i] != PATIENT or not w.text.rstrip().endswith("?"):
            continue
        found = None
        for j in range(i, max(i - QUESTION_WORDS, -1), -1):
            a, b = span(j, i)
            if who[j] != PATIENT or b - a > 2 * TURN_WINDOW_S:
                break
            if b - a >= MIN_TURN_S:
                s = scores(a, b)
                if label(s) != PATIENT and (found is None or s.max() > found[1]):
                    found = (j, float(s.max()), label(s))
        if found:
            who[found[0] : i + 1] = [found[2]] * (i + 1 - found[0])
    runs: list[list] = []  # [첫 낱말, 끝 낱말, 말한 사람]
    for i, name in enumerate(who):
        if runs and runs[-1][2] == name:
            runs[-1][1] = i
        else:
            runs.append([i, i, name])

    def coalesce() -> None:
        for k in range(len(runs) - 1, 0, -1):
            if runs[k][2] == runs[k - 1][2]:
                runs[k - 1][1] = runs[k][1]
                del runs[k]

    # 2) 짧은 묶음은 더 긴 이웃에 붙인다
    def length(run) -> float:
        a, b = span(run[0], run[1])
        return b - a

    def merge_short() -> None:
        while len(runs) > 1:
            k = min(range(len(runs)), key=lambda j: length(runs[j]))
            if length(runs[k]) >= MIN_TURN_S:
                return
            if k == 0 or (k + 1 < len(runs) and length(runs[k + 1]) > length(runs[k - 1])):
                runs[k + 1][0] = runs[k][0]
            else:
                runs[k - 1][1] = runs[k][1]
            del runs[k]
            coalesce()

    merge_short()
    if len(runs) == 1:
        return [phrase]

    # 나누는 자리 다듬기: 왼쪽은 앞사람, 오른쪽은 뒷사람 목소리에 가장 가까운 낱말 경계.
    # 양쪽 비교 구간은 같은 길이로, 두 묶음 안에서만 잡는다 (길이가 다르면 긴 쪽 유사도가 높게 나오고,
    # 짧게 끼어든 말 뒤의 다른 사람 말이 섞이면 흐려진다).
    def contrast(cut: float, size: float, before: str, after: str) -> float:
        left, right = scores(cut - size, cut), scores(cut, cut + size)
        value = 0.0
        if after != PATIENT:
            value += float(right[names.index(after)] - left[names.index(after)])
        if before != PATIENT:
            value += float(left[names.index(before)] - right[names.index(before)])
        return value

    for k in range(len(runs) - 1):
        a, b = runs[k], runs[k + 1]
        here = a[1]
        options = sorted(range(max(a[0], here - TURN_SHIFT_WORDS), min(b[1] - 1, here + TURN_SHIFT_WORDS) + 1),
                         key=lambda i: abs(i - here))  # 같으면 처음 자리
        lo, hi = span(a[0], a[1])[0], span(b[0], b[1])[1]
        options = [i for i in options if i == here or min(cuts[i] - lo, hi - cuts[i]) >= MIN_TURN_S]
        size = min([TURN_WINDOW_S] + [min(cuts[i] - lo, hi - cuts[i]) for i in options])
        if len(options) < 2 or size < MIN_TURN_S / 2:
            continue  # 비교할 자리가 없거나 너무 짧으면 비교가 흔들린다
        values = [contrast(cuts[i], size, a[2], b[2]) + TURN_SENTENCE_BONUS * ends_sentence(words[i].text) for i in options]
        best = options[int(np.argmax(values))]
        if not ends_sentence(words[best].text):
            near = [i for i in options if ends_sentence(words[i].text) and abs(cuts[i] - cuts[best]) <= SENTENCE_SNAP_S]
            if near:
                best = min(near, key=lambda i: abs(cuts[i] - cuts[best]))
        a[1], b[0] = best, best + 1
    merge_short()

    # 3) 짧은 묶음은 묶음 전체로 다시 비교한다 (창 몇 개가 잘못 나온 것이면 이웃과 같은 사람으로 나와 도로 이어진다).
    #    긴 묶음은 창 여러 개가 같은 사람을 가리켰으므로 그대로 둔다: 긴 구간 전체로 비교하면 놓친 짧은 끼어듦이
    #    섞여 유사도가 올라가 다른 사람으로 뒤집힐 수 있다.
    for run in runs:
        if length(run) < 2 * TURN_WINDOW_S:
            run[2] = label(scores(*span(run[0], run[1])))
    coalesce()
    if len(runs) == 1:
        return [phrase]
    parts = []
    for first, last, who in runs:
        part = phrase_from_words(words[first : last + 1], *span(first, last))
        parts.append(replace(part, speaker=who, speech_detected=phrase.speech_detected))
    return parts


def label_turns(phrases: list[TranscriptSegment], pcm16: np.ndarray, encoder: SpeakerEncoder, book: VoiceBook,
                threshold: float, cancel: threading.Event | None = None) -> list[TranscriptSegment]:
    """말 덩어리마다 말한 사람을 붙이고(label_segments), 덩어리 안에서 말한 사람이 바뀌면 나눈다(split_turns)."""
    out = []
    for phrase in label_segments(phrases, pcm16, encoder, book, threshold, cancel):
        if cancel is not None and cancel.is_set():
            return phrases
        out.extend(split_turns(phrase, pcm16, encoder, book, threshold))
    return out


def main_doctor(segments: list[TranscriptSegment]) -> str | None:
    """가장 오래 말한 등록 의사 (진료 기록에 표시)."""
    talk: dict[str, float] = {}
    for seg in segments:
        if seg.speaker and seg.speaker != PATIENT:
            talk[seg.speaker] = talk.get(seg.speaker, 0.0) + seg.end - seg.start
    return max(talk, key=talk.__getitem__) if talk else None
