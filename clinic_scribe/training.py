"""따라 읽기 학습 (약 1분): 의사가 화면에 나오는 짧은 문장을 하나씩 따라 읽는다.

읽을 문장(정답)을 알고 있으므로 같은 녹음으로
- 목소리를 등록하고(말한 사람 표시),
- Whisper 가 받아 적은 글을 정답과 맞춰 이 의사의 말이 어떻게 잘못 들리는지 찾아 딕션 사전(교정·용어)을 만든다.
학습 전과 후에 정확히 받아 적은 문장 수를 비교해 보여 준다. 녹음은 분석 뒤 버리고, Whisper 모델은 다시 학습시키지 않는다.
"""

from __future__ import annotations

import difflib
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clinic_scribe.diction import Correction, DictionBook, DictionError, apply_corrections, build_prompt, check_correction
from clinic_scribe.speakers import SAMPLE_RATE, SpeakerEncoder, enrollment_embeddings, speech_only
from clinic_scribe.whisper import TranscriptionResult

# 외래에서 자주 쓰는 말과 Whisper 가 자주 틀리는 약 이름·용량이 든 문장 (숫자는 Whisper 가 쓰는 대로 아라비아 숫자)
DEFAULT_PROMPTS = (
    "어디가 불편해서 오셨어요?",
    "언제부터 그러셨어요? 열이나 기침은 없으세요?",
    "배를 한번 눌러 볼게요. 여기 누르면 아프세요?",
    "트리메부틴은 배가 아플 때만 드세요.",
    "알마게이트 현탁액은 속 쓰릴 때 드세요.",
    "세티리진은 자기 전에 한 알 드세요.",
    "레바미피드와 모사프리드를 하루 3번 식후에 드세요.",
    "에소메프라졸 40mg은 아침 공복에 드세요.",
    "아세트아미노펜 500mg은 열날 때만 드세요.",
    "혈액검사랑 소변검사 결과 보고 다시 말씀드릴게요.",
    "증상이 계속되면 위내시경을 해 보는 게 좋겠습니다.",
    "약은 일주일 치 드릴게요. 안 나으면 다시 오세요.",
)
# 그 의사 사전의 용어로 만드는 문장
TERM_TEMPLATES = ("{} 처방해 드릴게요.", "오늘은 {} 이야기를 해 볼게요.")
GAP_S = 1.2  # 문장별 녹음을 이어 붙일 때 사이에 넣는 무음 (한 번에 전사하고 문장별로 다시 나눈다)
_IGNORED = re.compile(r"[\s.,?!·…'\"“”‘’()\[\]~-]")


def _norm(text: str) -> str:
    """띄어쓰기·문장부호·영문 대소문자를 무시한 비교용 글."""
    return _IGNORED.sub("", text).lower()


def build_prompts(vocabulary: list[str], count: int = len(DEFAULT_PROMPTS)) -> list[str]:
    """그 의사 사전의 용어로 만든 문장을 먼저(최대 절반), 나머지는 기본 문장으로 채운다."""
    terms = [t for t in vocabulary if _norm(t)][: count // 2]
    own = [TERM_TEMPLATES[i % len(TERM_TEMPLATES)].format(term) for i, term in enumerate(terms)]
    return (own + [p for p in DEFAULT_PROMPTS if p not in own])[:count]


def exact(expected: str, heard: str) -> bool:
    """띄어쓰기·문장부호를 빼고 같으면 정확히 받아 적은 것으로 본다."""
    return _norm(expected) == _norm(heard)


@dataclass(frozen=True)
class Finding:
    heard: str  # Whisper 가 받아 적은 말
    expected: str  # 화면에 나온 말 (의사가 읽은 말)
    prompt: str  # 그 문장


def _same_run(words: list[str], target: str, from_end: bool) -> int:
    """words 의 끝(또는 앞)에서 몇 낱말을 이어 붙여야 target 과 같아지는지. 같아지지 않으면 0."""
    joined, count = "", 0
    for word in reversed(words) if from_end else words:
        joined = _norm(word) + joined if from_end else joined + _norm(word)
        count += 1
        if joined == target:
            return count
        if len(joined) >= len(target):
            return 0
    return 0


def _trim_spacing_only_edges(got: list[str], want: list[str]) -> tuple[list[str], list[str]]:
    """바뀐 낱말 묶음의 앞뒤에서 띄어쓰기만 다른 낱말("속쓰릴" / "속 쓰릴")을 떼어 낸다."""
    for from_end in (True, False):
        while got and want:
            edge_got, edge_want = (got[-1], want[-1]) if from_end else (got[0], want[0])
            n_want = _same_run(want, _norm(edge_got), from_end)
            n_got = _same_run(got, _norm(edge_want), from_end) if not n_want else 0
            if n_want:
                got, want = (got[:-1], want[:-n_want]) if from_end else (got[1:], want[n_want:])
            elif n_got:
                got, want = (got[:-n_got], want[:-1]) if from_end else (got[n_got:], want[1:])
            else:
                break
    return got, want


def _trim_shared_ending(heard: str, expected: str) -> tuple[str, str]:
    """'현타객은 → 현탁액은'처럼 끝 글자(조사)가 같으면 떼어 '현타객 → 현탁액'으로 (다른 조사가 붙어도 고쳐지게)."""
    if len(heard) > 2 and len(expected) > 2 and heard[-1] == expected[-1] and not heard[-2].isspace() and not expected[-2].isspace():
        return heard[:-1], expected[:-1]
    return heard, expected


def find_misheard(expected: str, heard: str) -> list[Finding]:
    """화면 문장과 받아 적은 글을 낱말 단위로 맞춰 다르게 들린 부분을 찾는다 (띄어쓰기·문장부호 차이는 무시).

    빠지거나 더 들린 말은 고쳐 쓸 대상이 없어 넣지 않는다.
    """
    want = [w for w in expected.split() if _norm(w)]
    got = [w for w in heard.split() if _norm(w)]
    matcher = difflib.SequenceMatcher(a=[_norm(w) for w in got], b=[_norm(w) for w in want], autojunk=False)
    findings = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        heard_words, read_words = _trim_spacing_only_edges(got[i1:i2], want[j1:j2])
        wrong = " ".join(heard_words).strip(".,?!… ")
        right = " ".join(read_words).strip(".,?!… ")
        if not wrong or not right or _norm(wrong) == _norm(right):
            continue  # 띄어쓰기만 다르다
        wrong, right = _trim_shared_ending(wrong, right)
        findings.append(Finding(wrong, right, expected))
    return findings


def join_clips(clips: list[np.ndarray], gap_s: float = GAP_S) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """문장별 녹음(16 kHz)을 무음을 사이에 두고 이어 붙인다. 각 문장의 (시작, 끝) 시각도 돌려준다."""
    gap = np.zeros(int(gap_s * SAMPLE_RATE), np.int16)
    parts, spans, cursor = [gap], [], gap_s
    for clip in clips:
        clip = clip.astype(np.int16)
        spans.append((cursor, cursor + len(clip) / SAMPLE_RATE))
        cursor += len(clip) / SAMPLE_RATE + gap_s
        parts += [clip, gap]
    return np.concatenate(parts), spans


def split_by_prompts(prompts: list[str], heard: str) -> list[str]:
    """받아 적은 글 전체를 화면 문장들과 낱말 단위로 맞춰 문장별로 나눈다.

    Whisper 의 낱말 시각은 쉼 뒤에서 1초 넘게 앞당겨지기도 해서, 시각 대신 정답 문장과의 맞춤으로 나눈다.
    """
    want: list[str] = []
    owner: list[int] = []
    for index, prompt in enumerate(prompts):
        for w in prompt.split():
            if _norm(w):
                want.append(_norm(w))
                owner.append(index)
    got = [w for w in heard.split() if _norm(w)]
    texts: list[list[str]] = [[] for _ in prompts]
    if not want:
        return ["" for _ in prompts]
    matcher = difflib.SequenceMatcher(a=[_norm(w) for w in got], b=want, autojunk=False)
    for _tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for i in range(i1, i2):
            if j2 > j1:  # 같거나 바뀐 말: 짝이 되는 정답 낱말의 문장
                j = j1 + (i - i1) * (j2 - j1) // (i2 - i1)
            else:  # 더 들린 말: 바로 앞 정답 낱말의 문장
                j = max(j1 - 1, 0)
            texts[owner[min(j, len(owner) - 1)]].append(got[i])
    return [" ".join(words) for words in texts]


@dataclass
class TrainingResult:
    prompts: list[str]
    heard_before: list[str]  # 기본 힌트로 받아 적은 글 (학습 전)
    heard_after: list[str]  # 찾은 교정·용어를 넣어 다시 받아 적은 글 (학습 후)
    findings: list[Finding]
    embeddings: list[np.ndarray]  # 목소리 특징값 (화자 인식 모델이 없으면 빈 목록)
    speech_seconds: float
    voice_issue: str | None = None  # 목소리를 등록하지 못한 이유

    @property
    def correct_before(self) -> int:
        return sum(exact(p, h) for p, h in zip(self.prompts, self.heard_before))

    @property
    def correct_after(self) -> int:
        return sum(exact(p, h) for p, h in zip(self.prompts, self.heard_after))


def corrections_from(findings: list[Finding]) -> list[Correction]:
    rules = []
    for finding in findings:
        try:
            wrong, right = check_correction(finding.heard, finding.expected)
        except DictionError:
            continue  # 한 글자짜리 등은 교정으로 쓸 수 없다
        rules.append(Correction(wrong, right))
    return rules


def analyze(prompts: list[str], clips: list[np.ndarray], transcribe: Callable[[np.ndarray, str | None], TranscriptionResult],
            encoder: Callable[[], SpeakerEncoder], base_prompt: str,
            cancel: threading.Event | None = None) -> TrainingResult:
    """따라 읽은 녹음을 분석한다: 학습 전 전사 → 다르게 들린 말 찾기 → 그 교정·용어로 다시 전사(학습 후) → 목소리 특징값."""
    audio, _spans = join_clips(clips)

    def heard(result: TranscriptionResult) -> list[str]:
        return split_by_prompts(prompts, " ".join(seg.text for seg in result.segments))

    before = heard(transcribe(audio, None))
    findings = [f for prompt, said in zip(prompts, before) for f in find_misheard(prompt, said)]

    doctor = "따라 읽기"  # 학습 후 확인용 임시 사전의 이름
    rules = corrections_from(findings)
    trial = DictionBook(Path(), {})
    trial.set_entry(doctor, [r.right for r in rules], rules)
    after_result = transcribe(audio, build_prompt(base_prompt, trial, doctor) if rules else None)
    for seg in after_result.segments:
        seg.speaker = doctor  # 모두 이 의사가 읽은 말이다
    after_result.segments = apply_corrections(after_result.segments, trial)
    after = heard(after_result)

    speech = speech_only(audio)
    embeddings: list[np.ndarray] = []
    issue = None
    if cancel is None or not cancel.is_set():
        try:
            embeddings = enrollment_embeddings(encoder(), speech)
        except Exception as exc:  # 화자 인식 모델이 없어도 딕션 학습은 쓸 수 있다
            issue = str(exc)
    return TrainingResult(list(prompts), before, after, findings, embeddings, len(speech) / SAMPLE_RATE, issue)
