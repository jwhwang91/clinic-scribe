"""검토용 차트 초안: 구조, 화면·복사용 텍스트, 전사문 대조 자동 점검."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime

from clinic_scribe.whisper import TranscriptSegment, format_time

SECTIONS: tuple[tuple[str, str], ...] = (
    ("chief_complaint_hpi", "주호소·현병력"),
    ("exam_findings", "대화에서 확인된 진찰·검사"),
    ("assessment_plan", "의사가 언급한 평가·계획"),
    ("needs_review", "확인 필요"),
)
CLINICAL_KEYS = ("chief_complaint_hpi", "exam_findings", "assessment_plan")
SOURCES = ("환자", "의사", "불명확")

REVIEW_BANNER = "[검토용 초안] 자동 생성된 차트 초안입니다. 의료진이 전사문과 대조해 확인·수정한 뒤에만 사용하세요."
SPEAKER_NOTE = "※ 전사문에는 화자 표시가 없어 환자·의사 구분은 발화 내용으로 추정했습니다."
EMPTY_SECTION = "대화에서 확인된 내용 없음"


@dataclass
class ChartItem:
    text: str
    source: str = ""  # 환자 / 의사 / 불명확 (확인 필요 항목은 비어 있을 수 있음)
    evidence: str = ""  # 근거가 된 전사문 구절
    reason: str = ""  # 확인 필요 사유
    auto: bool = False  # 앱의 자동 점검이 추가한 항목
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChartDraft:
    sections: dict[str, list[ChartItem]]
    model: str = ""
    based_on: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    kind_label = "차트 초안"

    @classmethod
    def from_structured(cls, data: object, **meta) -> ChartDraft:
        """Claude 의 structured_output(JSON 스키마 결과)을 검증해 초안으로 만든다."""
        if not isinstance(data, dict):
            raise ValueError("응답이 JSON 객체가 아닙니다.")
        sections: dict[str, list[ChartItem]] = {}
        for key, _title in SECTIONS:
            raw_items = data.get(key)
            if not isinstance(raw_items, list):
                raise ValueError(f"응답에 '{key}' 항목 목록이 없습니다.")
            items = []
            for raw in raw_items:
                if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
                    raise ValueError(f"'{key}' 항목 형식이 올바르지 않습니다.")
                source = str(raw.get("source", "")).strip()
                items.append(
                    ChartItem(
                        text=str(raw["text"]).strip(),
                        source=source if source in SOURCES else "",
                        evidence=str(raw.get("evidence", "")).strip(),
                        reason=str(raw.get("reason", "")).strip(),
                    )
                )
            sections[key] = items
        return cls(sections=sections, **meta)

    # ---- 표시 -------------------------------------------------------------
    def to_text(self, include_evidence: bool = True) -> str:
        lines = [REVIEW_BANNER, ""]
        for key, title in SECTIONS:
            lines.append(f"■ {title}")
            items = self.sections.get(key, [])
            if not items:
                lines.append(f"- {EMPTY_SECTION}")
            for item in items:
                lines.append(f"- {_item_line(item)}")
                if include_evidence and item.evidence:
                    lines.append(f'    근거: "{item.evidence}"')
            lines.append("")
        lines.append(SPEAKER_NOTE)
        lines.append(f"(검토용 초안 · {self.created_at} · {self.model} · {self.based_on})")
        return "\n".join(lines)

    def to_html(self, include_evidence: bool = True) -> str:
        e = html.escape
        parts = [
            '<div style="background:#33240f;border:1px solid #f0b46a;padding:6px 8px;color:#fbbf24;">'
            f"<b>{e(REVIEW_BANNER)}</b></div>"
        ]
        if not any(self.sections.get(key) for key in CLINICAL_KEYS):
            parts.append(
                '<p style="color:#93a8f8;">대화에서 진료 관련 내용을 찾지 못했습니다. 차트에는 증상·진찰·평가·계획만 정리하므로 '
                "일상 대화는 비어 보이는 것이 정상입니다. 말한 내용 전체는 왼쪽 '전사문' 탭에서 확인하세요.</p>"
            )
        for key, title in SECTIONS:
            parts.append(f'<h3 style="margin:12px 0 4px 0;">{e(title)}</h3><ul style="margin-top:0;">')
            items = self.sections.get(key, [])
            if not items:
                parts.append(f'<li style="color:#8b93a1;">{e(EMPTY_SECTION)}</li>')
            for item in items:
                color = "#fdba74" if item.auto else "inherit"
                body = f'<span style="color:{color};">{e(_item_line(item))}</span>'
                if item.warnings:
                    body += f' <span style="color:#f87171;">⚠ {e("; ".join(item.warnings))}</span>'
                if include_evidence and item.evidence:
                    body += f'<br><span style="color:#8b93a1;font-size:small;">근거: “{e(item.evidence)}”</span>'
                parts.append(f"<li>{body}</li>")
            parts.append("</ul>")
        parts.append(f'<p style="color:#8b93a1;">{e(SPEAKER_NOTE)}<br>'
                     f"(검토용 초안 · {e(self.created_at)} · {e(self.model)} · {e(self.based_on)})</p>")
        return "".join(parts)


def _item_line(item: ChartItem) -> str:
    text = item.text
    if item.reason and item.reason not in text:
        text = f"{text} — {item.reason}"
    if item.source and not item.auto:
        text = f"{text} ({item.source})"
    return text


# ---- 전사문 대조 자동 점검 ------------------------------------------------

_NATIVE_UNITS = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "석": 3, "넷": 4, "네": 4, "넉": 4,
                 "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
_NATIVE_TENS = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70,
                "여든": 80, "아흔": 90}
# 고유어 기간 표현 (사흘 → 3일 등). 일주일은 1주·7일 모두 허용.
_NATIVE_DAYS = {"하루": [1], "이틀": [2], "사흘": [3], "나흘": [4], "닷새": [5], "엿새": [6], "이레": [7],
                "여드레": [8], "아흐레": [9], "열흘": [10], "보름": [15], "일주일": [1, 7], "한달": [1, 30]}
_COUNTERS = "번째|번|개월|개|알|정|포|잔|병|시간|시|달|주|살|차례|가지|군데|명|회|장|봉|캡슐|방울|숟가락|스푼|분|일|년|세|도|층"
_NATIVE_RE = re.compile(
    rf"({'|'.join(_NATIVE_TENS)})?\s?({'|'.join(sorted(_NATIVE_UNITS, key=len, reverse=True))})?\s?(?:{_COUNTERS})"
)
_SINO_RE = re.compile(
    rf"([일이삼사오육륙칠팔구십백천만]+)\s?(?:밀리그램|밀리|미리|그램|mg|센티|퍼센트|프로|씨씨|cc|키로|킬로|kg|{_COUNTERS})"
)
_SINO_DIGITS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
_SINO_UNITS = {"십": 10, "백": 100, "천": 1000}
_DIGITS_RE = re.compile(r"\d+(?:\.\d+)?")


def _parse_sino(word: str) -> int:
    total = section = num = 0
    for ch in word:
        if ch in _SINO_DIGITS:
            num = _SINO_DIGITS[ch]
        elif ch in _SINO_UNITS:
            section += (num or 1) * _SINO_UNITS[ch]
            num = 0
        elif ch == "만":
            total += ((section + num) or 1) * 10000
            section = num = 0
    return total + section + num


def transcript_numbers(text: str) -> set[float]:
    """전사문에 나온 수: 아라비아 숫자 + 고유어/한자어 수 표현(넉넉하게 포함)."""
    values = {float(n) for n in _DIGITS_RE.findall(text.replace(",", ""))}
    compact = text.replace(" ", "")
    for word, nums in _NATIVE_DAYS.items():
        if word in compact:
            values.update(float(n) for n in nums)
    for tens, unit in _NATIVE_RE.findall(text):
        if tens or unit:
            values.add(float(_NATIVE_TENS.get(tens, 0) + _NATIVE_UNITS.get(unit, 0)))
    for word in _SINO_RE.findall(text):
        values.add(float(_parse_sino(word)))
    return values


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()


def evidence_found(evidence: str, transcript_norm: str) -> bool:
    """근거 인용이 전사문에 있는지. '…'로 줄인 인용은 조각마다 확인하고, 조각의 80% 이상이 연속으로 맞으면 인정."""
    from difflib import SequenceMatcher

    pieces = [_normalize(p) for p in re.split(r"\.\.\.|…", evidence)]
    pieces = [p for p in pieces if p]
    if not pieces:
        return False
    for piece in pieces:
        if piece in transcript_norm:
            continue
        match = SequenceMatcher(None, piece, transcript_norm, autojunk=False).find_longest_match(
            0, len(piece), 0, len(transcript_norm)
        )
        if match.size < 0.8 * len(piece):
            return False
    return True


def verify_draft(draft: ChartDraft, segments: list[TranscriptSegment]) -> None:
    """전사문과 대조해 경고를 달고, 필요한 항목을 '확인 필요'에 자동으로 추가한다.

    LLM 결과를 고치지는 않는다. 근거 없는 내용, 전사문에 없는 숫자, 전사 단계의 불확실성을 드러낼 뿐이다.
    """
    transcript = " ".join(s.text for s in segments)
    transcript_norm = _normalize(transcript)
    numbers = transcript_numbers(transcript)
    review = draft.sections.setdefault("needs_review", [])

    for key in CLINICAL_KEYS:
        for item in draft.sections.get(key, []):
            if not item.evidence:
                item.warnings.append("근거 인용 없음")
            elif not evidence_found(item.evidence, transcript_norm):
                item.warnings.append("근거 인용을 전사문에서 찾지 못함")
            for n in _DIGITS_RE.findall(item.text.replace(",", "")):
                if float(n) not in numbers:
                    item.warnings.append(f"전사문에 없는 숫자 '{n}'")
            if item.warnings:
                review.append(ChartItem(text=f"[자동 점검] {item.text}", reason="; ".join(item.warnings), auto=True))

    suspicious = [s for s in segments if not s.speech_detected]
    if suspicious:
        where = ", ".join(format_time(s.start) for s in suspicious[:5])
        review.append(ChartItem(
            text=f"[자동 점검] 음성이 감지되지 않은 곳의 전사 {len(suspicious)}건 ({where})",
            reason="무음에서 생긴 Whisper 환각일 수 있음. 해당 내용을 근거로 쓰지 말 것",
            auto=True,
        ))
    for seg in segments:
        if seg.uncertain_terms:
            review.append(ChartItem(
                text=f"[자동 점검] 전사 신뢰도 낮은 표기: {', '.join(seg.uncertain_terms)} ({format_time(seg.start)})",
                reason="약명·숫자·용량을 녹음으로 확인",
                auto=True,
            ))
