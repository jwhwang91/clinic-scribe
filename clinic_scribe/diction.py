"""의사별 딕션 사전: 자주 쓰는 용어와 '잘못 들린 말 → 바른 말' 교정 (이 컴퓨터에서 처리).

- 용어: 그 의사가 진료한 녹음을 전사할 때 Whisper 힌트 앞에 넣어 처음부터 더 잘 알아듣게 한다.
- 교정: 전사 뒤 그 의사의 발화에서(공통 교정은 모든 발화에서) 잘못 들린 말을 바른 말로 바꾼다. 원래 들린 말은 남긴다.
- 전사문에서 고쳐 쓰면 여기에 쌓이고 다음 진료부터 자동으로 적용된다(고치면 배우기). 교정한 바른 말은 용어로도 쓴다.
녹음을 저장하거나 Whisper 모델 자체를 다시 학습시키지는 않는다.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from clinic_scribe.speakers import PATIENT, SAMPLE_RATE, SpeakerEncoder, VoiceBook, speech_only
from clinic_scribe.whisper import TranscriptSegment

COMMON = ""  # 모든 발화에 적용하는 공통 사전
MAX_TERM_CHARS = 40
MAX_PROMPT_CHARS = 400  # Whisper 힌트는 224 토큰까지라 길게 넣지 않는다


class DictionError(Exception):
    pass


@dataclass
class Correction:
    wrong: str
    right: str


@dataclass
class Entry:
    vocabulary: list[str] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.vocabulary and not self.corrections


def clean_term(text: str) -> str:
    return " ".join(text.replace("<", "").replace(">", "").split())[:MAX_TERM_CHARS]


def _key(text: str) -> str:
    """띄어쓰기·대소문자를 무시한 비교용 키."""
    return "".join(text.split()).lower()


def check_correction(wrong: str, right: str) -> tuple[str, str]:
    """교정으로 쓸 수 있는지 확인하고 정리한 (잘못 들린 말, 바른 말)을 돌려준다."""
    wrong, right = clean_term(wrong), clean_term(right)
    if len(_key(wrong)) < 2:
        raise DictionError("잘못 들린 말은 두 글자 이상이어야 합니다.")
    if not right:
        raise DictionError("바른 말을 입력하세요.")
    if _key(wrong) == _key(right):
        raise DictionError("잘못 들린 말과 바른 말이 같습니다.")
    return wrong, right


def _dedupe(terms) -> list[str]:
    seen, out = set(), []
    for term in terms:
        term = clean_term(term)
        if term and _key(term) not in seen:
            seen.add(_key(term))
            out.append(term)
    return out


class DictionBook:
    """딕션 사전 파일 (diction.json): 공통("") + 의사 이름별 용어·교정."""

    def __init__(self, path: Path, entries: dict[str, Entry] | None = None) -> None:
        self.path = path
        self.entries: dict[str, Entry] = entries or {}

    @classmethod
    def load(cls, path: Path) -> DictionBook:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            entries = {
                str(name): Entry(list(item.get("vocabulary", [])), [Correction(**c) for c in item.get("corrections", [])])
                for name, item in raw.get("entries", {}).items()
            }
        except FileNotFoundError:
            entries = {}
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise DictionError(f"딕션 사전 파일을 읽지 못했습니다 ({path}): {exc}") from exc
        return cls(path, entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"entries": {name: asdict(entry) for name, entry in self.entries.items() if not entry.empty}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def copy(self) -> DictionBook:
        """작업 스레드에 넘길 사본."""
        return DictionBook(self.path, {k: Entry(list(e.vocabulary), list(e.corrections)) for k, e in self.entries.items()})

    @property
    def empty(self) -> bool:
        return all(entry.empty for entry in self.entries.values())

    def doctors(self) -> list[str]:
        """사전이 있는 의사 이름 (공통 제외)."""
        return [name for name, entry in self.entries.items() if name != COMMON and not entry.empty]

    def entry(self, scope: str) -> Entry:
        return self.entries.get(scope) or Entry()

    def set_entry(self, scope: str, vocabulary: list[str], corrections: list[Correction]) -> int:
        """한 사전을 통째로 바꾼다. 쓸 수 없는 교정(두 글자 미만·빈 칸·같은 말)은 빼고 그 개수를 돌려준다."""
        rules: dict[str, Correction] = {}
        dropped = 0
        for rule in corrections:
            try:
                wrong, right = check_correction(rule.wrong, rule.right)
            except DictionError:
                dropped += 1
                continue
            rules[_key(wrong)] = Correction(wrong, right)  # 같은 말은 나중 것이 이긴다
        self.entries[scope] = Entry(_dedupe(vocabulary), list(rules.values()))
        return dropped

    def rename_scope(self, old: str, new: str) -> None:
        """의사 이름이 바뀌면 그 의사의 사전도 옮긴다 (새 이름에 이미 사전이 있으면 합친다)."""
        if old == new or old not in self.entries:
            return
        moved = self.entries.pop(old)
        if new in self.entries:
            kept = self.entries[new]
            self.set_entry(new, kept.vocabulary + moved.vocabulary, kept.corrections + moved.corrections)
        else:
            self.entries[new] = moved

    def add_correction(self, scope: str, wrong: str, right: str, add_term: bool = True) -> Correction:
        """교정 하나를 더한다 (전사문에서 고쳐 쓰기). 바른 말은 용어 목록에도 넣어 다음부터 처음부터 바르게 받아 적게 한다."""
        wrong, right = check_correction(wrong, right)
        entry = self.entry(scope)
        rules = [c for c in entry.corrections if _key(c.wrong) != _key(wrong)] + [Correction(wrong, right)]
        self.set_entry(scope, entry.vocabulary + ([right] if add_term else []), rules)
        return Correction(wrong, right)


def _compile(rules: list[Correction]):
    if not rules:
        return None
    ordered = sorted(rules, key=lambda c: len(_key(c.wrong)), reverse=True)  # 겹치면 긴 말이 먼저
    gap = r"\s*"  # 띄어쓰기는 있어도 없어도 맞는다
    parts = [
        f"(?P<r{i}>{gap.join(re.escape(ch) for ch in rule.wrong if not ch.isspace())})"
        for i, rule in enumerate(ordered)
    ]
    return re.compile("|".join(parts), re.IGNORECASE), ordered


def _rules_for(book: DictionBook, doctor: str | None) -> list[Correction]:
    merged = {_key(c.wrong): c for c in book.entry(COMMON).corrections}
    if doctor:
        merged.update({_key(c.wrong): c for c in book.entry(doctor).corrections})  # 의사 사전이 공통보다 우선
    return list(merged.values())


def apply_corrections(segments: list[TranscriptSegment], book: DictionBook) -> list[TranscriptSegment]:
    """교정 사전을 적용한다: 공통 교정은 모든 발화, 의사 교정은 그 의사로 표시된 발화에.

    항상 Whisper 가 받아 적은 원문에서 다시 계산하므로 여러 번 불러도 결과가 같고, 바꾼 말이 다시 바뀌지 않는다.
    """
    compiled: dict[str | None, object] = {}
    out = []
    for seg in segments:
        source = seg.original_text if seg.original_text is not None else seg.text
        doctor = seg.speaker if seg.speaker and seg.speaker != PATIENT else None
        if doctor not in compiled:
            compiled[doctor] = _compile(_rules_for(book, doctor))
        rules = compiled[doctor]
        applied: list[str] = []
        text = source
        if rules is not None:
            pattern, ordered = rules

            def swap(match: re.Match) -> str:
                rule = ordered[int(match.lastgroup[1:])]
                applied.append(f"{rule.wrong} → {rule.right}")
                return rule.right

            text = pattern.sub(swap, source)
        if applied:
            out.append(replace(seg, text=text, original_text=source, corrections=applied))
        elif seg.original_text is not None:
            out.append(replace(seg, text=source, original_text=None, corrections=[]))  # 교정이 지워진 경우
        else:
            out.append(seg)
    return out


def build_prompt(base: str, book: DictionBook, doctor: str | None) -> str:
    """Whisper 힌트: "이 진료 의사의 용어, 공통 용어(교정의 바른 말 포함). 설정의 기본 힌트". 길면 용어 뒤쪽을 뺀다.

    용어 목록을 마침표로 끝내야 한다. 쉼표로 기본 힌트에 이어 붙이면 Whisper 가 여러 사람 말을 문장부호 없이
    한 줄로 합쳐 버려(가상 진료 실험), 말한 사람 표시가 무너졌다.
    """
    base = base.strip()
    terms: list[str] = []
    for scope in ([doctor] if doctor else []) + [COMMON]:
        entry = book.entry(scope)
        terms += entry.vocabulary + [c.right for c in entry.corrections]
    base_key = _key(base)
    budget = MAX_PROMPT_CHARS - (len(base) + 2 if base else 0) - 1
    words, keys = "", []
    for term in _dedupe(terms):
        key = _key(term)
        if key in base_key or any(key.startswith(k) and len(key) - len(k) <= 2 for k in keys):
            continue  # 기본 힌트에 있는 말, 조사만 붙은 말("트리메부틴은")은 뺀다
        candidate = f"{words}, {term}" if words else term
        if len(candidate) > budget:
            break
        words, keys = candidate, keys + [key]
    return f"{words}. {base}" if words and base else (f"{words}." if words else base)


def guess_doctor(pcm16: np.ndarray, encoder: SpeakerEncoder, voices: VoiceBook, threshold: float,
                 chunk_s: float = 3.0, max_chunks: int = 24) -> str | None:
    """이 진료의 의사: 말소리를 3초 조각으로 나눠 등록한 목소리와 비교한 다수결(두 조각 이상 맞아야 한다)."""
    if not voices.voices:
        return None
    speech = speech_only(pcm16)
    step = int(chunk_s * SAMPLE_RATE)
    starts = list(range(0, len(speech) - step + 1, step))
    if len(starts) > max_chunks:
        starts = [starts[i * len(starts) // max_chunks] for i in range(max_chunks)]
    votes: Counter[str] = Counter()
    for start in starts:
        name, _score = voices.identify(encoder.embed(speech[start : start + step]), threshold)
        if name:
            votes[name] += 1
    if not votes:
        return None
    name, count = votes.most_common(1)[0]
    return name if count >= min(2, len(starts)) else None
