"""씨차트 연결 점검: 씨차트(진료 차트 프로그램) 화면에 보이는 환자 이름을 이 앱이 읽을 수 있는지 확인한다 (Windows).

씨차트에는 아무것도 입력하거나 누르지 않는다. 화면 낭독기가 쓰는 Windows 화면 읽기(UI Automation)와
창 글자(WM_GETTEXT)로 열린 창의 글자를 읽어, 사용자가 적은 환자 이름이 어느 칸에 있는지 찾는다.
다른 환자를 열었을 때 같은 자리에서 새 이름이 읽히면, 나중에 그 자리에서 이름을 가져와 진료 기록을 환자별로 나눌 수 있다.

환자 이름은 이 컴퓨터 메모리에서만 쓴다. 결과에는 그 이름을 '[환자 이름]'으로, 다른 글자는 모양만
(한글 ○ · 숫자 # · 영문 x) 남기고 화면에서 흔히 쓰는 낱말(환자명·접수·대기 등)만 그대로 둔다.
결과 파일을 다른 사람에게 보여 줘도 환자 정보가 드러나지 않게 하기 위해서다.

응답 없는 프로그램의 창을 읽다가 앱이 멈추지 않게, 따로 띄운 프로세스에서 읽는다
(python -m clinic_scribe.emr_probe: 이름은 명령줄이 아니라 표준 입력으로 받고, 가린 결과만 JSON 으로 내보낸다).
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from clinic_scribe import __version__, procutil

PLACEHOLDER = "[환자 이름]"
BUDGET_S = 60.0  # 한 번 점검에 창을 읽는 시간
FOCUS_ELEMENTS = 20000  # 씨차트로 보이는 창(제목에 이름이 있거나 프로그램 이름이 비슷한 창)에서 읽을 요소 수
OTHER_ELEMENTS = 2000  # 그 밖의 창
MAX_DEPTH = 40
MAX_CHILDREN = 400  # 한 칸 아래 요소 수 (긴 표는 앞부분만)
TEXT_LIMIT = 80  # 결과에 남기는 글자 수 (가린 뒤)
TREE_LINES = 1500  # 결과 파일의 화면 구조: 창 하나에 남기는 줄 수
KEYWORDS = ("seechart", "씨차트", "chart", "차트", "emr")
SKIP_CLASSES = {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW", "Windows.UI.Core.CoreWindow"}
REMOTE_CLASSES = {"TscShellContainerClass": "원격 데스크톱", "RAIL_WINDOW": "원격 앱(RemoteApp)",
                  "Transparent Windows Client": "Citrix"}

SOURCE_LABELS = {"title": "창 제목", "name": "화면 읽기(이름)", "value": "화면 읽기(값)", "legacy": "화면 읽기(MSAA 값)",
                 "text": "창 글자"}
CONTROL_TYPES = {
    50000: "Button", 50001: "Calendar", 50002: "CheckBox", 50003: "ComboBox", 50004: "Edit", 50005: "Hyperlink",
    50006: "Image", 50007: "ListItem", 50008: "List", 50009: "Menu", 50010: "MenuBar", 50011: "MenuItem",
    50012: "ProgressBar", 50013: "RadioButton", 50014: "ScrollBar", 50015: "Slider", 50016: "Spinner",
    50017: "StatusBar", 50018: "Tab", 50019: "TabItem", 50020: "Text", 50021: "ToolBar", 50022: "ToolTip",
    50023: "Tree", 50024: "TreeItem", 50025: "Custom", 50026: "Group", 50027: "Thumb", 50028: "DataGrid",
    50029: "DataItem", 50030: "Document", 50031: "SplitButton", 50032: "Window", 50033: "Pane", 50034: "Header",
    50035: "HeaderItem", 50036: "Table", 50037: "TitleBar", 50038: "Separator", 50039: "SemanticZoom", 50040: "AppBar",
}
CONTROL_LABELS = {
    "Edit": "입력칸", "Text": "글자", "Window": "창", "Pane": "패널", "ListItem": "목록 줄", "List": "목록",
    "DataItem": "표 줄", "DataGrid": "표", "Table": "표", "TitleBar": "제목 표시줄", "StatusBar": "상태 표시줄",
    "TreeItem": "트리 항목", "Tree": "트리", "Custom": "사용자 정의 칸", "Document": "문서", "Group": "묶음",
    "Tab": "탭", "TabItem": "탭 항목", "Button": "버튼", "ComboBox": "콤보 상자", "Header": "머리글",
    "HeaderItem": "머리글 칸", "MenuItem": "메뉴 항목", "ToolBar": "도구 모음", "HWND": "Win32 창",
}

# 화면에서 흔히 쓰는 낱말: 결과에 그대로 남긴다 (환자 정보가 아니고, 화면 구조를 알아보는 데 쓴다)
_SAFE_HANGUL = frozenset("""
씨차트 차트 환자 환자명 수진자 수진자명 성명 이름 성함 차트번호 등록번호 환자번호 주민번호 주민등록번호 생년월일
나이 연령 성별 남자 여자 연락처 전화 전화번호 휴대폰 핸드폰 주소 보험 보험유형 보험구분 건강보험 의료급여 일반
접수 접수일 접수번호 접수시간 대기 대기자 대기중 대기실 진료 진료중 진료실 진료과 진료일 진료비 완료 수납 미수
처방 처방전 약품 약품명 용량 용법 일수 횟수 투약 주사 처치 상병 상병명 상병코드 진단 진단명 증상 주증상 주호소
현병력 과거력 가족력 알레르기 부작용 메모 기록 경과 소견 검사 결과 영상 판독 예약 예약일 내원 내원일 초진 재진
신환 구환 일자 날짜 시간 오늘 어제 확인 취소 저장 닫기 조회 검색 신규 등록 수정 삭제 출력 인쇄 파일 편집 보기
도구 도움말 설정 종료 다음 이전 목록 번호 순번 순서 구분 상태 담당 담당의 의사 원장 원장님 간호사 직원 메시지
알림 호출 보류 수가 비고 바이탈 혈압 체온 맥박 호흡 체중 신장 혈당 전체 선택 적용 입력 추가 이동 열기 새로고침
최소화 최대화 복원 병원 의원 내과 서류 진단서 소견서 의뢰서 영수증 본인부담금 합계 금액 카드 현금 환불 검진 접종
가상차트
""".split())
# 한 글자 낱말은 따로 있을 때만 남긴다: 이어 붙이면 이름(예: 남세일)과 구별할 수 없다
_SAFE_SHORT = frozenset("남 여 세 년 월 일 시 분 초 명 건 회 번 원 개 약 키 예 네 님".split())
_SAFE_LATIN = frozenset("""
ok cancel yes no save close file edit view help tools window emr ocs pacs soap rx dx tx cc pi bp bt pr rr bst bmi
id seechart chart name patient date time age sex m f kg cm am pm
""".split())
_TOKENS = re.compile(r"[가-힣]+|[A-Za-z]+|\d+|\s+|.", re.S)
_NAME_SHAPE = re.compile(r"\s*([가-힣]{2,4})(?:\s*(?:님|씨))?(?=$|[\s(\[/,·:-]|\d)")
_LONG_DIGITS = re.compile(r"\d{4,}")


def supported() -> bool:
    return sys.platform == "win32"


# ---------------------------------------------------------------------------- 가리기
def name_pattern(name: str) -> re.Pattern | None:
    """적은 이름을 찾는 식: 화면에 '김 민 준'처럼 띄어 써 있어도 찾는다. 두 글자보다 짧으면 None."""
    chars = [c for c in name if not c.isspace()]
    if len(chars) < 2:
        return None
    return re.compile(r"\s*".join(re.escape(c) for c in chars))


def _safe_compound(run: str) -> bool:
    """두 글자 이상인 낱말만으로 이어진 말인가 (예: 대기환자목록)."""
    ok = [True] + [False] * len(run)
    for end in range(2, len(run) + 1):
        ok[end] = any(ok[start] and run[start:end] in _SAFE_HANGUL for start in range(end - 1))
    return ok[-1]


def _safe_hangul(run: str) -> bool:
    return run in _SAFE_HANGUL or run in _SAFE_SHORT or _safe_compound(run)


def _mask_plain(text: str) -> str:
    out = []
    for token in _TOKENS.findall(text):
        first = token[0]
        if "가" <= first <= "힣":
            out.append(token if _safe_hangul(token) else "○" * len(token))
        elif first.isascii() and first.isalpha():
            out.append(token if token.lower() in _SAFE_LATIN else "x" * len(token))
        elif first.isdigit():
            out.append("#" * len(token))
        elif token.isspace():
            out.append(" ")  # 줄바꿈·탭은 한 칸으로
        elif first.isalpha():
            out.append("○")  # 한자·자모 등 다른 글자
        else:
            out.append(token)  # 문장부호·기호
    return "".join(out)


def mask(text: str, name_re: re.Pattern | None = None) -> str:
    """환자 이름은 [환자 이름]으로, 다른 글자는 모양만 남긴다. 화면에서 흔히 쓰는 낱말은 그대로 둔다."""
    if not text:
        return ""
    parts = name_re.split(text) if name_re is not None else [text]
    masked = PLACEHOLDER.join(_mask_plain(part) for part in parts).strip()
    if len(masked) > TEXT_LIMIT:
        masked = f"{masked[:TEXT_LIMIT]}…({len(masked)}자)"
    return masked


def mask_identifier(text: str, name_re: re.Pattern | None = None, limit: int = 60) -> str:
    """프로그램 안의 이름(클래스·AutomationId)·오류 글: 영문은 남기고, 긴 숫자(차트번호 등)와 한글은 가린다."""
    if not text:
        return ""
    if name_re is not None:
        text = name_re.sub(PLACEHOLDER, text)
    text = _LONG_DIGITS.sub(lambda m: "#" * len(m.group()), text)
    text = re.sub(r"[가-힣]+", lambda m: m.group() if _safe_hangul(m.group()) else "○" * len(m.group()), text)
    return text[:limit]


def is_screen_word(text: str) -> bool:
    """화면에서 흔히 쓰는 낱말인가 (환자명·접수 등: 사람 이름이 아니다)."""
    return _safe_hangul(text)


def looks_like_name(text: str) -> bool:
    """사람 이름처럼 보이는 글자인가 (두~네 글자 한글로 시작하는 짧은 글자, 화면 낱말이 아님). 대기 목록을 알아보는 단서."""
    match = _NAME_SHAPE.match(text)
    return bool(match) and len(text.strip()) <= 24 and not _safe_hangul(match.group(1))


# ---------------------------------------------------------------------------- 결과 만들기 (가리기 전 → 가린 결과)
@dataclass
class TopWindow:
    hwnd: int
    pid: int
    exe: str
    class_name: str
    title: str  # 가리기 전 제목: 이 프로세스 안에서만 쓴다
    elevated: bool | None = None  # 관리자 권한으로 실행 중인가 (모르면 None)
    minimized: bool = False
    rect: list[int] | None = None


def new_result() -> dict:
    return {"version": 1, "when": datetime.now().isoformat(timespec="seconds"), "app": __version__,
            "os": platform.platform(), "error": "", "uia": "", "elevated": None, "elapsed": 0.0,
            "out_of_time": False, "windows": [], "hits": []}


def _distinct(texts: dict) -> dict:
    """같은 글자는 한 번만: 이름 → 값 → MSAA 값 → 창 글자 순서로."""
    seen, out = set(), {}
    for source in ("name", "value", "legacy", "text"):
        text = (texts.get(source) or "").strip()
        if text and text not in seen:
            seen.add(text)
            out[source] = text
    return out


def _step_label(step, name_re) -> str:
    control, class_name, automation_id, index = step
    label = control + (f" {mask_identifier(class_name, name_re)}" if class_name else "")
    if automation_id:
        label += f" #{mask_identifier(automation_id, name_re)}"
    return f"{label}[{index}]"


def window_result(top: TopWindow, nodes: list[dict], name_re: re.Pattern, *, method: str, truncated: bool,
                  error: str = "", focus: bool = False) -> tuple[dict, list[dict]]:
    """창 하나에서 읽은 요소(가리기 전)를 가린 결과로 바꾸고, 환자 이름이 있는 자리를 찾는다."""
    window_class = mask_identifier(top.class_name, name_re)
    title_hit = bool(name_re.search(top.title))
    entry = {
        "exe": top.exe, "pid": top.pid, "class": window_class, "title": mask(top.title, name_re), "title_hit": title_hit,
        "elevated": top.elevated, "minimized": top.minimized, "focus": focus, "scanned": True, "method": method,
        "error": error, "elements": len(nodes), "truncated": truncated,
        "frameworks": sorted({n["framework"] for n in nodes if n.get("framework")}), "nodes": [],
    }
    hits = []
    if title_hit:
        hits.append({"exe": top.exe, "window_class": window_class, "kind": "title", "sources": ["title"],
                     "control": "Window", "class": window_class, "id": "", "text": entry["title"], "rect": top.rect,
                     "selected": None, "row": None, "col": None, "path": [], "key": []})
    for node in nodes:
        texts = _distinct(node.get("texts", {}))
        # 맨 위 창의 이름은 창 제목과 같다 (위에서 따로 셌다)
        found = [s for s, t in texts.items() if name_re.search(t) and not (node["depth"] == 0 and s == "name")]
        masked = {s: mask(t, name_re) for s, t in texts.items()}
        out = {"d": node["depth"], "c": node["control"], "k": mask_identifier(node.get("class", ""), name_re),
               "id": mask_identifier(node.get("id", ""), name_re), "t": masked, "hit": found,
               "nl": not found and any(looks_like_name(t) for t in texts.values()),
               "sel": node.get("selected"), "more": bool(node.get("more"))}
        entry["nodes"].append(out)
        if found:
            hits.append({
                "exe": top.exe, "window_class": window_class, "kind": "element", "sources": found,
                "control": node["control"], "class": out["k"], "id": out["id"], "text": masked[found[0]],
                "rect": node.get("rect"), "selected": node.get("selected"), "row": node.get("row"),
                "col": node.get("col"), "path": [_step_label(step, name_re) for step in node["path"]],
                "key": [[c, mask_identifier(k, name_re), mask_identifier(i, name_re), n] for c, k, i, n in node["path"]],
            })
    return entry, hits


def skipped_window(top: TopWindow, name_re: re.Pattern, focus: bool) -> dict:
    entry, _hits = window_result(top, [], name_re, method="", truncated=False, focus=focus)
    entry["scanned"] = False
    return entry


def add_window(result: dict, entry: dict, hits: list[dict]) -> None:
    index = len(result["windows"])
    result["windows"].append(entry)
    for hit in hits:
        hit["window"] = index
        result["hits"].append(hit)


# ---------------------------------------------------------------------------- 판단 · 결과 파일
def location_key(hit: dict) -> tuple:
    """이름을 찾은 자리 (프로그램 · 창 종류 · 칸까지의 경로): 점검끼리 같은 자리인지 비교한다."""
    return hit["exe"].lower(), hit["window_class"], hit["kind"], json.dumps(hit["key"])


def found_runs(runs: list[dict]) -> list[dict]:
    return [run for run in runs if not run.get("error") and run["hits"]]


def stable_locations(runs: list[dict]) -> list[dict]:
    """이름을 찾은 점검(환자)마다 이름이 나온 같은 자리: 환자가 바뀌어도 그 자리에서 새 이름이 읽힌다.

    못 찾은 점검(이름을 잘못 적었거나 다시 해 본 것)은 빼고 비교한다.
    """
    found = found_runs(runs)
    if len(found) < 2:
        return []
    common = set.intersection(*({location_key(hit) for hit in run["hits"]} for run in found))
    return [hit for hit in found[-1]["hits"] if location_key(hit) in common]


def _missing_reasons(run: dict) -> list[str]:
    reasons = []
    if run.get("uia") != "ok":
        reasons.append("Windows 화면 읽기(UI Automation)를 쓰지 못해 창 글자만 읽었습니다.")
    scanned = [w for w in run["windows"] if w["scanned"]]
    windows = [w for w in scanned if w["focus"]] or scanned  # 씨차트로 보이는 창이 있으면 그 창으로 판단한다
    if any(w["class"] in REMOTE_CLASSES for w in run["windows"]):
        reasons.append("씨차트가 원격 데스크톱 창 안에서 열려 있으면 그 안의 글자는 읽을 수 없습니다.")
    # 관리자 권한 프로그램의 창은 일반 권한 프로그램이 창 테두리 말고는 읽지 못한다
    if run.get("elevated") is not True and any(w["elevated"] is not False and w["elements"] <= 3 for w in windows):
        reasons.append("씨차트가 관리자 권한으로 실행 중이면 읽지 못합니다: clinic-scribe 를 마우스 오른쪽 단추 → "
                       "'관리자 권한으로 실행'으로 열어 다시 해 보세요.")
    if run.get("out_of_time"):
        reasons.append("시간 안에 모든 창을 읽지 못했습니다: 씨차트 말고 다른 프로그램 창을 닫고 다시 해 보세요.")
    reasons.append("이름을 화면에 보이는 그대로 적었는지 확인해 주세요. 그래도 못 찾으면 결과 파일을 보내 주세요: "
                   "씨차트가 화면 글자를 밖으로 내보내지 않는 방식이면 화면 글자 인식(OCR) 같은 다른 방법을 찾아야 합니다.")
    return reasons


def verdict(runs: list[dict]) -> tuple[str, str]:
    """(상태, 안내). 상태: done 환자가 바뀌어도 읽힘 · found 찾음 · missing 못 찾음 · error 점검 못 함."""
    last = runs[-1]
    if last.get("error"):
        return "error", last["error"]
    stable = stable_locations(runs)
    if stable:
        return "done", (f"환자가 바뀌어도 같은 자리 {len(stable)}곳에서 이름이 읽힙니다. "
                        "이 자리에서 환자 이름을 가져와 진료 기록을 환자별로 나눌 수 있습니다.")
    if last["hits"]:
        if len(found_runs(runs)) >= 2:
            return "found", "이름은 찾았지만 두 환자의 이름이 나온 자리가 달랐습니다. 결과 파일을 보내 주시면 살펴보겠습니다."
        return "found", ("이름을 찾았습니다. 씨차트에서 다른 환자를 열고 그 이름으로 한 번 더 찾으면, "
                         "환자가 바뀔 때도 같은 자리에서 읽히는지 확인합니다.")
    return "missing", "이름을 찾지 못했습니다. " + " ".join(_missing_reasons(last))


def describe_hit(hit: dict, detail: bool = True) -> str:
    """이름을 찾은 곳 한 줄. detail 이면 프로그램·읽은 방법·위치까지."""
    if hit["kind"] == "title":
        text = f"창 제목 “{hit['text']}”"
        return f"{hit['exe']} · {text}" if detail else text
    what = CONTROL_LABELS.get(hit["control"], hit["control"])
    names = " · ".join(x for x in (hit["class"], f"#{hit['id']}" if hit["id"] else "") if x)
    text = f"{what}{f' ({names})' if names else ''} “{hit['text']}”"
    if hit.get("selected"):
        text += " · 선택된 줄"
    if hit.get("row") is not None and hit.get("col") is not None:
        text += f" · 표 {hit['row'] + 1}행 {hit['col'] + 1}열"
    if not detail:
        return text
    rect = hit.get("rect")
    where = f" · 위치 {rect[0]},{rect[1]} 크기 {rect[2]}×{rect[3]}" if rect else ""
    sources = ", ".join(SOURCE_LABELS.get(s, s) for s in hit["sources"])
    return f"{hit['exe']} · {text} · {sources}{where}"


def _elevation(value) -> str:
    return {True: "관리자", False: "일반", None: "알 수 없음"}[value]


def describe_window(window: dict) -> str:
    bits = [window["exe"] or "(알 수 없는 프로그램)", window["class"] or "(클래스 없음)",
            f"“{window['title']}”" if window["title"] else "(제목 없음)"]
    if window["scanned"]:
        bits.append(f"요소 {window['elements']:,}개" + (" (더 있음)" if window["truncated"] else ""))
        bits.append({"UIA": "화면 읽기", "Win32": "창 글자"}.get(window["method"], window["method"]))
        if window["frameworks"]:
            bits.append("틀 " + "/".join(window["frameworks"]))
    else:
        bits.append("읽지 않음")
    bits.append(f"권한 {_elevation(window['elevated'])}")
    if window["minimized"]:
        bits.append("최소화")
    if window["class"] in REMOTE_CLASSES:
        bits.append(f"{REMOTE_CLASSES[window['class']]} 창")
    if window["error"]:
        bits.append(window["error"])
    return " · ".join(bits)


def _tree_windows(run: dict) -> list[dict]:
    """결과 파일에 화면 구조를 남길 창: 이름을 찾은 프로그램의 창, 못 찾았으면 씨차트로 보이는 창 (없으면 읽은 창 모두)."""
    scanned = [w for w in run["windows"] if w["scanned"]]
    pids = {run["windows"][hit["window"]]["pid"] for hit in run["hits"]}
    if pids:
        return [w for w in scanned if w["pid"] in pids]
    return [w for w in scanned if w["focus"]] or scanned


def _node_line(node: dict) -> str:
    parts = [node["c"]]
    if node["k"]:
        parts.append(node["k"])
    if node["id"]:
        parts.append(f"#{node['id']}")
    labels = {"name": "", "value": "값=", "legacy": "MSAA값=", "text": "창글자="}
    parts += [f"{labels.get(source, source + '=')}“{text}”" for source, text in node["t"].items()]
    if node["hit"]:
        parts.append("★ 지금 환자 이름")
    elif node["nl"]:
        parts.append("· 이름 모양")
    if node.get("sel"):
        parts.append("· 선택됨")
    if node.get("more"):
        parts.append("· 아래 요소 더 있음(읽지 않음)")
    return "  " * min(node["d"], 30) + " ".join(parts)


def format_report(runs: list[dict]) -> str:
    """결과 파일: 판단, 점검마다 이름을 찾은 곳과 읽은 창, 이름이 나온 프로그램의 화면 구조(가린 글자)."""
    first = runs[0]
    lines = [
        "clinic-scribe 씨차트 연결 점검 결과",
        f"앱 {first.get('app', '')} · {first.get('os', '')} · clinic-scribe 권한 {_elevation(first.get('elevated'))}",
        "환자 이름은 [환자 이름]으로, 다른 글자는 모양만(한글 ○ · 숫자 # · 영문 x) 남겼습니다. "
        "화면에서 흔히 쓰는 낱말(환자명·접수·대기 등)만 그대로 둡니다.",
        "",
        f"■ 판단: {verdict(runs)[1]}",
    ]
    for number, run in enumerate(runs, 1):
        lines += ["", f"■ 점검 {number} ({run.get('when', '')}, {run.get('elapsed', 0)}초)"]
        if run.get("error"):
            lines.append(f"  점검하지 못했습니다: {run['error']}")
            continue
        if run.get("uia") != "ok":
            lines.append(f"  화면 읽기(UI Automation)를 쓰지 못해 창 글자만 읽었습니다: {run.get('uia')}")
        lines.append(f"  이름을 찾은 곳 {len(run['hits'])}곳")
        for index, hit in enumerate(run["hits"], 1):
            lines.append(f"   {index}) {describe_hit(hit)}")
            if hit["path"]:
                lines.append(f"      경로: {' › '.join(hit['path'])}")
        scanned = [w for w in run["windows"] if w["scanned"]]
        lines.append(f"  읽은 창 {len(scanned)}개 · 요소 {sum(w['elements'] for w in scanned):,}개"
                     + (" · 시간이 모자라 다 읽지 못함" if run.get("out_of_time") else ""))
        lines += [f"   - {describe_window(w)}" for w in run["windows"]]
    found = found_runs(runs)
    if len(found) >= 2:
        stable = stable_locations(runs)
        lines += ["", f"■ 이름을 찾은 점검 {len(found)}번 모두 이름이 나온 같은 자리 {len(stable)}곳"]
        lines += [f"   - {describe_hit(hit)}" for hit in stable]
    last = runs[-1]
    if not last.get("error"):
        lines += ["", "■ 화면 구조 (마지막 점검, 가린 글자. ★ 지금 환자 이름이 있는 자리)"]
        for window in _tree_windows(last):
            lines.append(f"  [{describe_window(window)}]")
            nodes = window["nodes"]
            lines += ["    " + _node_line(node) for node in nodes[:TREE_LINES]]
            if len(nodes) > TREE_LINES:
                lines.append(f"    … 요소 {len(nodes) - TREE_LINES:,}개 더")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- Windows: 창 목록 · 창 글자
def _as_str(value) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_bool(value) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_rect(value) -> list[int] | None:
    try:
        left, top, width, height = (int(v) for v in value)
    except (TypeError, ValueError, OverflowError):
        return None
    return [left, top, width, height]


class _Win32:
    """창 목록과 창 글자 (user32 · kernel32 · advapi32, ctypes). 다른 프로그램에는 읽는 메시지만 보낸다."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes, self.wintypes = ctypes, wintypes
        self.user32 = u = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.advapi32 = a = ctypes.WinDLL("advapi32", use_last_error=True)
        try:
            self.dwmapi = ctypes.WinDLL("dwmapi")
        except OSError:
            self.dwmapi = None
        hwnd, lparam = wintypes.HWND, wintypes.LPARAM
        self.WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, hwnd, lparam)
        u.EnumWindows.argtypes = [self.WNDENUMPROC, lparam]
        u.EnumChildWindows.argtypes = [hwnd, self.WNDENUMPROC, lparam]
        u.IsWindowVisible.argtypes = [hwnd]
        u.IsIconic.argtypes = [hwnd]
        u.GetWindowThreadProcessId.argtypes = [hwnd, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowThreadProcessId.restype = wintypes.DWORD
        u.GetClassNameW.argtypes = [hwnd, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowTextW.argtypes = [hwnd, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowRect.argtypes = [hwnd, ctypes.POINTER(wintypes.RECT)]
        u.GetDlgCtrlID.argtypes = [hwnd]
        u.GetAncestor.argtypes = [hwnd, wintypes.UINT]
        u.GetAncestor.restype = hwnd
        u.SendMessageTimeoutW.argtypes = [hwnd, wintypes.UINT, wintypes.WPARAM, lparam, wintypes.UINT, wintypes.UINT,
                                          ctypes.POINTER(ctypes.c_size_t)]
        u.SendMessageTimeoutW.restype = lparam
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                 ctypes.POINTER(wintypes.DWORD)]
        a.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        a.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                          ctypes.POINTER(wintypes.DWORD)]
        if self.dwmapi is not None:
            self.dwmapi.DwmGetWindowAttribute.argtypes = [hwnd, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
            self.dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
        self._processes: dict[int, tuple[str, bool | None]] = {}

    def _enum(self, parent: int | None) -> list[int]:
        handles: list[int] = []
        callback = self.WNDENUMPROC(lambda hwnd, _lparam: handles.append(hwnd) is None)
        if parent is None:
            self.user32.EnumWindows(callback, 0)
        else:
            self.user32.EnumChildWindows(parent, callback, 0)
        return [h for h in handles if h]

    def top_windows(self) -> list[TopWindow]:
        """보이는 창 (화면 맨 위에 있는 창부터)."""
        tops = []
        for hwnd in self._enum(None):
            class_name = self.class_name(hwnd)
            if not self.user32.IsWindowVisible(hwnd) or self._cloaked(hwnd) or class_name in SKIP_CLASSES:
                continue
            rect = self.rect(hwnd)
            minimized = bool(self.user32.IsIconic(hwnd))
            if not minimized and (rect[2] <= 1 or rect[3] <= 1):
                continue
            pid = self.wintypes.DWORD()
            self.user32.GetWindowThreadProcessId(hwnd, self.ctypes.byref(pid))
            exe, elevated = self.process(pid.value)
            tops.append(TopWindow(hwnd, pid.value, exe, class_name, self.title(hwnd), elevated, minimized, rect))
        return tops

    def class_name(self, hwnd: int) -> str:
        buf = self.ctypes.create_unicode_buffer(256)
        self.user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    def title(self, hwnd: int) -> str:
        # 다른 프로그램의 맨 위 창: 메시지를 보내지 않고 제목을 읽는다 (응답 없는 프로그램이어도 멈추지 않는다)
        buf = self.ctypes.create_unicode_buffer(512)
        self.user32.GetWindowTextW(hwnd, buf, 512)
        return buf.value

    def window_text(self, hwnd: int, limit: int = 1000) -> str:
        """칸의 글자 (WM_GETTEXT). 응답 없는 프로그램이면 0.2초 만에 포기한다."""
        size = self.ctypes.c_size_t()
        abort_if_hung = 0x0002
        if not self.user32.SendMessageTimeoutW(hwnd, 0x000E, 0, 0, abort_if_hung, 200, self.ctypes.byref(size)):
            return ""  # WM_GETTEXTLENGTH
        length = min(int(size.value), limit)
        if length <= 0:
            return ""
        buf = self.ctypes.create_unicode_buffer(length + 1)
        if not self.user32.SendMessageTimeoutW(hwnd, 0x000D, length + 1, self.ctypes.addressof(buf), abort_if_hung, 200,
                                               self.ctypes.byref(size)):
            return ""  # WM_GETTEXT
        return buf.value

    def rect(self, hwnd: int) -> list[int]:
        r = self.wintypes.RECT()
        if not self.user32.GetWindowRect(hwnd, self.ctypes.byref(r)):
            return [0, 0, 0, 0]
        return [r.left, r.top, r.right - r.left, r.bottom - r.top]

    def _cloaked(self, hwnd: int) -> bool:
        """보이지 않게 숨긴 창 (다른 가상 데스크톱, 백그라운드 앱)."""
        if self.dwmapi is None:
            return False
        value = self.wintypes.DWORD()
        result = self.dwmapi.DwmGetWindowAttribute(hwnd, 14, self.ctypes.byref(value), self.ctypes.sizeof(value))
        return result == 0 and bool(value.value)  # DWMWA_CLOAKED

    def process(self, pid: int) -> tuple[str, bool | None]:
        """(실행 파일 이름, 관리자 권한인가)."""
        if pid not in self._processes:
            handle = self.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                self._processes[pid] = ("", None)
            else:
                try:
                    size = self.wintypes.DWORD(1024)
                    buf = self.ctypes.create_unicode_buffer(1024)
                    ok = self.kernel32.QueryFullProcessImageNameW(handle, 0, buf, self.ctypes.byref(size))
                    self._processes[pid] = (os.path.basename(buf.value) if ok else "", self._elevated(handle))
                finally:
                    self.kernel32.CloseHandle(handle)
        return self._processes[pid]

    def _elevated(self, process_handle) -> bool | None:
        token = self.wintypes.HANDLE()
        if not self.advapi32.OpenProcessToken(process_handle, 0x0008, self.ctypes.byref(token)):  # TOKEN_QUERY
            return None
        try:
            value, size = self.wintypes.DWORD(), self.wintypes.DWORD()
            if not self.advapi32.GetTokenInformation(token, 20, self.ctypes.byref(value), self.ctypes.sizeof(value),
                                                     self.ctypes.byref(size)):  # TokenElevation
                return None
            return bool(value.value)
        finally:
            self.kernel32.CloseHandle(token)

    def own_elevation(self) -> bool | None:
        return self._elevated(self.kernel32.GetCurrentProcess())

    def walk(self, hwnd: int, deadline: float, limit: int) -> tuple[list[dict], bool]:
        """화면 읽기를 못 쓸 때: 창 안의 Win32 칸과 그 글자."""
        root = {"depth": 0, "control": "Window", "class": self.class_name(hwnd), "id": "", "framework": "Win32",
                "rect": self.rect(hwnd), "texts": {}, "path": ()}
        nodes, by_hwnd, counts = [root], {hwnd: root}, {}
        for child in self._enum(hwnd):
            if len(nodes) >= limit or time.monotonic() > deadline:
                return nodes, True
            parent = by_hwnd.get(self.user32.GetAncestor(child, 1), root)  # GA_PARENT
            index = counts[id(parent)] = counts.get(id(parent), -1) + 1
            class_name = self.class_name(child)
            control_id = self.user32.GetDlgCtrlID(child)
            step = ("HWND", class_name, str(control_id) if control_id else "", index)
            node = {"depth": parent["depth"] + 1, "control": "HWND", "class": class_name, "id": step[2],
                    "framework": "Win32", "rect": self.rect(child), "texts": {"text": self.window_text(child)},
                    "path": parent["path"] + (step,)}
            by_hwnd[child] = node
            nodes.append(node)
        return nodes, False


# ---------------------------------------------------------------------------- Windows: 화면 읽기 (UI Automation)
_UIA_PROPERTIES = {  # (comtypes 가 만든 상수 이름, 값)
    "name": ("UIA_NamePropertyId", 30005), "control": ("UIA_ControlTypePropertyId", 30003),
    "class": ("UIA_ClassNamePropertyId", 30012), "id": ("UIA_AutomationIdPropertyId", 30011),
    "framework": ("UIA_FrameworkIdPropertyId", 30024), "rect": ("UIA_BoundingRectanglePropertyId", 30001),
    "hwnd": ("UIA_NativeWindowHandlePropertyId", 30020), "value": ("UIA_ValueValuePropertyId", 30045),
    "legacy": ("UIA_LegacyIAccessibleValuePropertyId", 30093),
    "selected": ("UIA_SelectionItemIsSelectedPropertyId", 30079),
    "row": ("UIA_GridItemRowPropertyId", 30064), "col": ("UIA_GridItemColumnPropertyId", 30065),
}


class _Uia:
    """화면 낭독기가 쓰는 Windows 화면 읽기 (comtypes). 요소마다 필요한 속성을 한 번에 받아 온다."""

    def __init__(self) -> None:
        import comtypes.client

        self.module = module = comtypes.client.GetModule("UIAutomationCore.dll")
        self.automation = comtypes.client.CreateObject(module.CUIAutomation, interface=module.IUIAutomation)
        self.ids = {key: getattr(module, constant, value) for key, (constant, value) in _UIA_PROPERTIES.items()}
        self.cache = self.automation.CreateCacheRequest()
        for prop in self.ids.values():
            self.cache.AddProperty(prop)
        self.cache.TreeFilter = self.automation.RawViewCondition
        self.walker = self.automation.RawViewWalker

    def cached(self, element, key: str):
        try:
            return element.GetCachedPropertyValue(self.ids[key])
        except Exception:
            return None

    def identity(self, element) -> tuple[str, str, str]:
        """(종류, 클래스, AutomationId): 경로 한 칸을 알아보는 값."""
        control = CONTROL_TYPES.get(_as_int(self.cached(element, "control")), "Element")
        return control, _as_str(self.cached(element, "class")), _as_str(self.cached(element, "id"))

    def root(self, hwnd: int):
        return self.automation.ElementFromHandleBuildCache(hwnd, self.cache)

    def children(self, element, limit: int = MAX_CHILDREN) -> list:
        kids: list = []
        try:
            child = self.walker.GetFirstChildElementBuildCache(element, self.cache)
            while child and len(kids) < limit:
                kids.append(child)
                child = self.walker.GetNextSiblingElementBuildCache(child, self.cache)
        except Exception:
            pass
        return kids

    def walk(self, hwnd: int, deadline: float, limit: int, win: _Win32) -> tuple[list[dict], bool]:
        root = self.root(hwnd)
        nodes: list[dict] = []
        stopped = False

        def visit(element, depth: int, index: int, parent_path: tuple) -> None:
            nonlocal stopped
            if len(nodes) >= limit or time.monotonic() > deadline:
                stopped = True
                return
            node = self._node(element, depth, index, parent_path, win)
            nodes.append(node)
            if depth >= MAX_DEPTH:
                return
            try:
                child = self.walker.GetFirstChildElementBuildCache(element, self.cache)
            except Exception:
                return
            count = 0
            while child and not stopped:
                if count >= MAX_CHILDREN:
                    node["more"] = True
                    return
                visit(child, depth + 1, count, node["path"])
                count += 1
                try:
                    child = self.walker.GetNextSiblingElementBuildCache(child, self.cache)
                except Exception:
                    return

        visit(root, 0, 0, ())
        return nodes, stopped

    def _node(self, element, depth: int, index: int, parent_path: tuple, win: _Win32) -> dict:
        def get(key: str):
            return self.cached(element, key)

        control, class_name, automation_id = self.identity(element)
        hwnd = _as_int(get("hwnd")) or 0
        texts = {"name": _as_str(get("name")), "value": _as_str(get("value")), "legacy": _as_str(get("legacy"))}
        if hwnd and depth > 0:
            texts["text"] = win.window_text(hwnd)
        return {"depth": depth, "control": control, "class": class_name, "id": automation_id,
                "framework": _as_str(get("framework")), "rect": _as_rect(get("rect")), "texts": texts,
                "selected": _as_bool(get("selected")), "row": _as_int(get("row")), "col": _as_int(get("col")),
                "more": False, "path": parent_path + ((control, class_name, automation_id, index),)}


# ---------------------------------------------------------------------------- 점검
def _error_text(exc: BaseException, name_re: re.Pattern | None) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return (name_re.sub(PLACEHOLDER, text) if name_re is not None else text)[:300]


def _keyword(top: TopWindow) -> bool:
    text = f"{top.exe} {top.title}".lower()
    return any(word in text for word in KEYWORDS)


def scan(name: str, exclude_pids=(), budget_s: float = BUDGET_S) -> dict:
    """열린 창에서 환자 이름을 찾는다. 돌려주는 결과에는 가린 글자만 있다.

    제목에 이름이 있는 창, 씨차트로 보이는 창(프로그램 이름·제목)부터 읽고, 이름을 찾으면 그 프로그램의 창만 마저 읽는다.
    """
    result = new_result()
    name_re = name_pattern(name)
    if name_re is None:
        result["error"] = "환자 이름을 두 글자 이상 적어 주세요."
        return result
    if not supported():
        result["error"] = "씨차트 연결 점검은 Windows 에서만 할 수 있습니다."
        return result
    started = time.monotonic()
    deadline = started + budget_s
    win = _Win32()
    result["elevated"] = win.own_elevation()
    try:
        uia: _Uia | None = _Uia()
        result["uia"] = "ok"
    except Exception as exc:
        uia = None
        result["uia"] = _error_text(exc, name_re)
    excluded = {os.getpid(), *exclude_pids}
    tops = [top for top in win.top_windows() if top.pid not in excluded]
    focus = {top.hwnd: bool(name_re.search(top.title)) or _keyword(top) for top in tops}
    tops.sort(key=lambda top: (not name_re.search(top.title), not focus[top.hwnd]))
    found_pids: set[int] = set()
    for top in tops:
        if (found_pids and top.pid not in found_pids) or time.monotonic() > deadline:
            result["out_of_time"] |= not found_pids
            add_window(result, skipped_window(top, name_re, focus[top.hwnd]), [])
            continue
        limit = FOCUS_ELEMENTS if focus[top.hwnd] else OTHER_ELEMENTS
        nodes, truncated, method, error = None, False, "", ""
        if uia is not None:
            try:
                nodes, truncated = uia.walk(top.hwnd, deadline, limit, win)
                method = "UIA"
            except Exception as exc:
                error = "화면 읽기 실패: " + _error_text(exc, name_re)
        if nodes is None:
            nodes, truncated = win.walk(top.hwnd, deadline, limit)
            method = "Win32"
        entry, hits = window_result(top, nodes, name_re, method=method, truncated=truncated, error=error,
                                    focus=focus[top.hwnd])
        add_window(result, entry, hits)
        if hits:
            found_pids.add(top.pid)
    result["elapsed"] = round(time.monotonic() - started, 1)
    return result


REPO = Path(__file__).resolve().parent.parent


def child_python() -> str:
    """창을 읽는 프로세스를 띄울 Python. 결과를 표준 출력으로 받으려고 pythonw 대신 python 을 쓴다 (콘솔 창은 띄우지 않는다)."""
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe" and python.with_name("python.exe").is_file():
        python = python.with_name("python.exe")
    return str(python)


def run_probe(name: str, cancel: threading.Event | None = None, budget_s: float = BUDGET_S) -> dict:
    """따로 띄운 프로세스에서 scan 을 돌린다: 응답 없는 프로그램의 창을 읽다가 앱이 멈추지 않게."""
    request = json.dumps({"name": name, "exclude_pids": [os.getpid()], "budget_s": budget_s}).encode("ascii")
    done = procutil.run([child_python(), "-m", "clinic_scribe.emr_probe"], stdin_data=request, cwd=str(REPO),
                        timeout=budget_s + 60, cancel=cancel)
    try:
        return json.loads(done.stdout.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        name_re = name_pattern(name)
        detail = procutil.tail(done.stderr)
        raise RuntimeError("점검 프로세스가 결과를 내지 못했습니다. "
                           + (name_re.sub(PLACEHOLDER, detail) if name_re is not None else detail)) from None


def main() -> int:
    """표준 입력 {"name", "exclude_pids", "budget_s"} → 표준 출력 가린 결과(JSON, ASCII)."""
    sys.coinit_flags = 0  # COINIT_MULTITHREADED: 화면 읽기 클라이언트 권장 (comtypes 를 처음 불러올 때 쓰인다)
    name_re = None
    try:
        request = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
        name = str(request.get("name", ""))
        name_re = name_pattern(name)
        result = scan(name, [int(pid) for pid in request.get("exclude_pids", [])],
                      float(request.get("budget_s", BUDGET_S)))
    except Exception as exc:
        result = new_result()
        result["error"] = "점검 중 오류: " + _error_text(exc, name_re)
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=True).encode("ascii"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
