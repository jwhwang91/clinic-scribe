"""주요 내용 요약: 진료든 일상 대화든 주요 내용을 주제별 개조식으로 정리한다.

나중에 실제 의무기록 칸으로 옮기기 쉽도록 "한 줄 요약 + 주제별 주요 내용 + 할 일 + 확인 필요"로 둔다.
ChartDraft 와 같은 모양(to_text / to_html / based_on / kind_label)이라 화면·복사·내보내기를 그대로 쓴다.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime

GENERAL_BANNER = "[자동 요약] 녹음된 대화의 주요 내용을 자동으로 정리했습니다. 음성 인식 오류가 있을 수 있으니 전사문과 대조해 확인하세요."


@dataclass
class KeyPointGroup:
    category: str
    items: list[str]


@dataclass
class ConversationSummary:
    one_line: str
    key_points: list[KeyPointGroup]
    action_items: list[str]
    needs_review: list[str]
    model: str = ""
    based_on: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    kind_label = "주요 내용 요약"

    @classmethod
    def from_structured(cls, data: object, **meta) -> ConversationSummary:
        if not isinstance(data, dict):
            raise ValueError("응답이 JSON 객체가 아닙니다.")
        if not isinstance(data.get("one_line"), str):
            raise ValueError("응답에 'one_line' 문자열이 없습니다.")
        groups = data.get("key_points")
        if not isinstance(groups, list):
            raise ValueError("응답에 'key_points' 목록이 없습니다.")
        key_points = []
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("category"), str) or not _strings(group.get("items")):
                raise ValueError("'key_points' 의 묶음 형식이 올바르지 않습니다.")
            items = [item.strip() for item in group["items"] if item.strip()]
            if items:
                key_points.append(KeyPointGroup(group["category"].strip() or "기타", items))
        lists = {}
        for key in ("action_items", "needs_review"):
            if not _strings(data.get(key)):
                raise ValueError(f"응답의 '{key}' 가 문자열 목록이 아닙니다.")
            lists[key] = [item.strip() for item in data[key] if item.strip()]
        return cls(one_line=data["one_line"].strip(), key_points=key_points, **lists, **meta)

    def to_text(self, include_evidence: bool = True) -> str:
        lines = [GENERAL_BANNER, "", "■ 한 줄 요약", self.one_line or "-", "", "■ 주요 내용"]
        for group in self.key_points:
            lines.append(f"[{group.category}]")
            lines += [f"- {item}" for item in group.items]
        if not self.key_points:
            lines.append("- 없음")
        for title, items in (("할 일·결정 사항", self.action_items), ("확인 필요", self.needs_review)):
            lines += ["", f"■ {title}"] + ([f"- {item}" for item in items] or ["- 없음"])
        lines += ["", f"(자동 요약 · {self.created_at} · {self.model} · {self.based_on})"]
        return "\n".join(lines)

    def to_html(self, include_evidence: bool = True) -> str:
        e = html.escape
        parts = [
            '<div style="background:#1b2240;border:1px solid #3b4a8a;padding:6px 8px;color:#c7d2fe;">'
            f"<b>{e(GENERAL_BANNER)}</b></div>",
            f'<p style="font-size:large;"><b>{e(self.one_line or "요약할 내용이 없습니다.")}</b></p>',
            '<h3 style="margin:12px 0 4px 0;">주요 내용</h3>',
        ]
        for group in self.key_points:
            parts.append(f'<p style="margin:8px 0 2px 0;color:#93a8f8;"><b>{e(group.category)}</b></p><ul style="margin-top:0;">')
            parts += [f"<li>{e(item)}</li>" for item in group.items]
            parts.append("</ul>")
        if not self.key_points:
            parts.append('<p style="color:#8b93a1;">없음</p>')
        for title, items in (("할 일·결정 사항", self.action_items), ("확인 필요", self.needs_review)):
            parts.append(f'<h3 style="margin:12px 0 4px 0;">{e(title)}</h3><ul style="margin-top:0;">')
            parts += [f"<li>{e(item)}</li>" for item in items] or ['<li style="color:#8b93a1;">없음</li>']
            parts.append("</ul>")
        parts.append(f'<p style="color:#8b93a1;">(자동 요약 · {e(self.created_at)} · {e(self.model)} · {e(self.based_on)})</p>')
        return "".join(parts)


def _strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
