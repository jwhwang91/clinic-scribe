"""자동 진료 나누기: 스페이스바로 진료를 끝내는 것을 깜빡했을 때의 대비책.

- 다음 환자 신호("들어오세요" 등)가 실시간 전사에 나오면 그 말이 시작하기 전에서 진료를 나눈다(그 말은 다음 진료로).
- 끝인사("안녕히 가세요" 등)가 나오면 그 말이 끝난 뒤에서 나눈다.
- 한 문장에 둘 다 있으면 끝인사 쪽으로 본다(그 문장까지 앞 진료).
- 말이 정해진 시간 이상 없으면 마지막 발화 뒤에서 나눈다(이 판단은 MainWindow 가 한다).
화자 구분이 없으므로 누가 말했든 신호로 본다. 나눈 앞 진료는 바로 전사·요약하고 녹음은 이어진다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NOISE = re.compile(r"[\s.,!?~…·'\"“”‘’]")


@dataclass(frozen=True)
class Cue:
    kind: str  # "next"(다음 환자 신호) / "end"(끝인사)
    phrase: str


def parse_cues(text: str) -> list[str]:
    """설정 문자열(쉼표·줄바꿈 구분) → 신호 목록."""
    return [p.strip() for p in re.split(r"[,\n]", text or "") if p.strip()]


def _compact(text: str) -> str:
    return _NOISE.sub("", text)


def find_cue(text: str, next_cues: list[str], end_cues: list[str]) -> Cue | None:
    """띄어쓰기·문장부호를 무시하고 신호를 찾는다. 끝인사를 먼저 본다."""
    compact = _compact(text)
    if not compact:
        return None
    for phrase in end_cues:
        if _compact(phrase) and _compact(phrase) in compact:
            return Cue("end", phrase)
    for phrase in next_cues:
        if _compact(phrase) and _compact(phrase) in compact:
            return Cue("next", phrase)
    return None
