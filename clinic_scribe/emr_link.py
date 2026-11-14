"""씨차트 연결: 녹음할 때 씨차트에 열린 환자 이름을 읽고, 요약이 끝나면 씨차트 진료 기록 칸 끝에 넣는다 (Windows).

어디를 읽고 어디에 넣을지는 병원 컴퓨터에서 '씨차트 연결 점검'(emr_probe)으로 정한다.
- 환자 이름 자리: 두 환자로 점검했을 때 같은 자리에서 이름이 읽힌 곳. 그 자리의 글자 모양(가린 글자, 예: '[환자 이름] (남/##세)')
  에서 이름이 있던 부분만 떼어 읽는다.
- 진료 기록 칸: 씨차트의 그 칸에 표시 글자(@@@@)를 적고 찾은 입력칸.
설정에는 화면 구조(프로그램 · 창 종류 · 칸까지의 경로)와 가린 글자만 두고, 환자 이름은 두지 않는다.

넣을 때는 지금 씨차트에 열린 환자가 녹음하며 읽은 환자와 같은지 먼저 확인하고, 칸 끝에 덧붙이기만 한다 (이미 적힌 내용은
건드리지 않고, 되돌리기(Ctrl+Z)도 된다). 씨차트의 저장은 누르지 않는다: 의사가 읽고 고친 뒤 직접 저장한다.

응답 없는 프로그램 때문에 앱이 멈추지 않게 따로 띄운 프로세스에서 한다 (python -m clinic_scribe.emr_link: 요청은 표준 입력,
결과는 표준 출력 JSON(ASCII)). 환자 이름과 진료 내용은 명령줄과 로그에 남기지 않는다.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading

from clinic_scribe import emr_probe, procutil
from clinic_scribe.emr_probe import CONTROL_LABELS, PLACEHOLDER

MARKER = "@@@@"  # 진료 기록 칸을 찾을 때 씨차트의 그 칸에 적는 표시 글자
MAX_NAME_PLACES = 3
MAX_NOTE = 200_000  # 진료 기록 칸에서 읽는 글자 수 (덧붙인 글 확인용)
READ_TIMEOUT_S = 30.0
WRITE_TIMEOUT_S = 40.0
FIELD_CONTROLS = ("Edit", "Document")
SOURCES = ("title", "name", "value", "legacy", "text")
# 넣지 못한 이유 (표시창 · 상태 표시줄에 짧게)
REASONS = {
    "no_patient": "씨차트에서 지금 환자 이름을 읽지 못했습니다",
    "mismatch": "씨차트에 지금 열린 환자가 다릅니다",
    "no_field": "씨차트 진료 기록 칸을 찾지 못했습니다",
    "refused": "그 칸에는 넣을 수 없습니다",
    "failed": "넣은 글이 칸에 보이지 않습니다",
    "error": "씨차트 연결 중 오류",
}

GWL_STYLE, ES_MULTILINE, ES_READONLY = -16, 0x0004, 0x0800
WM_GETTEXTLENGTH, EM_GETSEL, EM_SETSEL, EM_REPLACESEL, EM_GETLIMITTEXT = 0x000E, 0x00B0, 0x00B1, 0x00C2, 0x00D5
SMTO_ABORTIFHUNG = 0x0002
UIA_VALUE_PATTERN = 10002

_NAME = r"[가-힣](?:\s?[가-힣]){1,4}|[A-Za-z][A-Za-z.' -]{0,38}[A-Za-z]"
_TEMPLATE = re.compile(re.escape(PLACEHOLDER) + r"|○+|#+|[가-힣]+|[A-Za-z]+|\s+|.", re.S)
_CUT = re.compile(r"…\(\d+자\)$")  # emr_probe.mask 가 긴 글자를 줄인 표시
_SAMPLE_NAME = "가나다"


class EmrLinkError(Exception):
    pass


def supported() -> bool:
    return emr_probe.supported()


# ---------------------------------------------------------------------------- 이름 떼어 읽기
def name_regex(template: str) -> re.Pattern | None:
    """가린 글자 모양 → 이름을 떼어 읽는 식. 이름 자리가 없으면 None.

    '[환자 이름]' 바로 옆 글자는 그대로 맞추고, 떨어진 글자는 같은 종류(한글 · 숫자 · 영문)면 맞는다고 본다:
    성별 · 나이처럼 환자마다 다른 글자가 있어도 읽히게.
    """
    text = template.strip()
    cut = bool(_CUT.search(text))  # 길어서 뒤를 줄인 글자: 앞부분만 맞춘다
    tokens = _TEMPLATE.findall(_CUT.sub("", text))
    if PLACEHOLDER not in tokens:
        return None
    parts, named = [], False
    for i, token in enumerate(tokens):
        touching = PLACEHOLDER in tokens[max(0, i - 1):i] + tokens[i + 1:i + 2]
        first = token[0]
        if token == PLACEHOLDER:
            parts.append("(?P=name)" if named else f"(?P<name>{_NAME})")
            named = True
        elif token.isspace():
            parts.append(r"\s+")
        elif first == "#":
            parts.append(r"\d+")
        elif first == "○":
            parts.append(rf"\S{{{len(token)}}}" if touching else r"\S+?")
        elif "가" <= first <= "힣":
            parts.append(re.escape(token) if touching else "[가-힣]+")
        elif first.isascii() and first.isalpha():
            parts.append(re.escape(token) if touching and set(token) != {"x"} else "[A-Za-z]+")
        else:
            parts.append(re.escape(token))
    return re.compile("".join(parts) + ("" if cut else r"\Z"))


def read_name(template: str, text: str) -> str:
    """정한 자리의 글자에서 환자 이름만. 읽지 못했거나 화면 낱말(환자명 등)이면 ""."""
    regex = name_regex(template)
    match = regex.match(" ".join(text.split())) if regex is not None else None
    if match is None:
        return ""
    name = " ".join(match.group("name").split())
    return "" if emr_probe.is_screen_word(name.replace(" ", "")) else name


def name_key(name: str) -> str:
    return "".join(name.split())


def same_patient(a: str, b: str) -> bool:
    """띄어쓰기만 다른 이름은 같은 환자로 본다."""
    return bool(name_key(a)) and name_key(a) == name_key(b)


def agreed_name(reads: list[str]) -> str:
    """녹음하는 동안 씨차트에서 읽은 이름이 모두 같으면 그 이름. 다르면(그 사이 환자가 바뀜) 또는 못 읽었으면 ""."""
    return reads[-1] if reads and len({name_key(name) for name in reads}) == 1 else ""


def _sample(masked: str) -> str:
    """가린 글자 → 같은 모양의 보기 글자 (이름 자리는 '가나다'): 다른 환자의 글자 모양으로도 이름이 읽히는지 볼 때."""
    text = masked.replace(PLACEHOLDER, "\0").replace("○", "가").replace("#", "1")
    text = re.sub(r"(?<![A-Za-z])x+(?![A-Za-z])", lambda m: "a" * len(m.group()), text)
    return text.replace("\0", _SAMPLE_NAME)


# ---------------------------------------------------------------------------- 점검 결과 → 자리
def _place(hit: dict, method: str) -> dict:
    return {"exe": hit["exe"], "window_class": hit["window_class"], "method": method, "kind": hit["kind"],
            "source": "title" if hit["kind"] == "title" else hit["sources"][0], "key": hit["key"]}


def name_places(runs: list[dict]) -> list[dict]:
    """두 환자 이상으로 점검해 같은 자리에서 이름이 읽힌 곳 → 환자 이름을 읽을 자리 (믿을 만한 곳부터).

    이름만 있는 칸, 다른 글자와 섞인 칸, 창 제목 순서로 두고, 점검한 환자마다 같은 글자 모양으로 이름을 떼어 낼 수 없는 곳은 뺀다.
    """
    found = emr_probe.found_runs(runs)
    places = []
    for hit in emr_probe.stable_locations(runs):
        where = emr_probe.location_key(hit)
        texts = [h["text"] for run in found for h in run["hits"] if emr_probe.location_key(h) == where]
        if not all(read_name(hit["text"], _sample(text)) == _SAMPLE_NAME for text in texts):
            continue
        method = found[-1]["windows"][hit["window"]]["method"] or "Win32"
        places.append({**_place(hit, method), "template": hit["text"]})
    places.sort(key=lambda place: (place["kind"] == "title", place["template"] != PLACEHOLDER))
    return places[:MAX_NAME_PLACES]


def _edit_class(class_name: str) -> bool:
    return any(word in class_name.lower() for word in ("edit", "memo"))


def field_places(run: dict) -> list[dict]:
    """진료 기록 칸에 표시 글자를 적고 찾은 점검 결과 → 요약을 넣을 칸 후보 (입력칸만)."""
    places = []
    for hit in run.get("hits", []):
        if hit["kind"] == "element" and (hit["control"] in FIELD_CONTROLS
                                         or (hit["control"] == "HWND" and _edit_class(hit["class"]))):
            places.append(_place(hit, run["windows"][hit["window"]]["method"] or "Win32"))
    return places


def describe_place(place: dict) -> str:
    if place["kind"] == "title":
        return f"{place['exe']} 창 제목"
    control, class_name, automation_id, _index = place["key"][-1]
    names = " · ".join(x for x in (class_name, f"#{automation_id}" if automation_id else "") if x)
    what = "입력칸" if control in FIELD_CONTROLS or (control == "HWND" and _edit_class(class_name)) else control
    return f"{place['exe']} {CONTROL_LABELS.get(what, what)}" + (f" ({names})" if names else "")


def _valid(place, name: bool) -> bool:
    try:
        ok = (isinstance(place["exe"], str) and bool(place["exe"]) and isinstance(place["window_class"], str)
              and place["method"] in ("UIA", "Win32") and place["kind"] in ("title", "element")
              and place["source"] in SOURCES and isinstance(place["key"], list)
              and all(isinstance(step, list) and len(step) == 4 and all(isinstance(x, str) for x in step[:3])
                      and isinstance(step[3], int) for step in place["key"]))
    except (KeyError, TypeError):
        return False
    return ok and (not name or (isinstance(place.get("template"), str) and name_regex(place["template"]) is not None))


def usable_names(places) -> list[dict]:
    """설정에 둔 환자 이름 자리 중 쓸 수 있는 것."""
    return [place for place in places if _valid(place, name=True)] if isinstance(places, list) else []


def usable_field(place) -> dict | None:
    return place if isinstance(place, dict) and _valid(place, name=False) else None


def field_report(run: dict | None) -> str:
    """결과 파일에 덧붙이는 '진료 기록 칸 찾기' (가린 글자. 표시 글자는 [표시 글자])."""
    if run is None:
        return ""
    lines = ["", f"■ 진료 기록 칸 찾기 (표시 글자 {MARKER} · {run.get('when', '')}, {run.get('elapsed', 0)}초)"]
    if run.get("error"):
        return "\n".join(lines + [f"  찾지 못했습니다: {run['error']}"]) + "\n"
    lines.append(f"  표시 글자를 찾은 곳 {len(run['hits'])}곳")
    for index, hit in enumerate(run["hits"], 1):
        lines.append(f"   {index}) {emr_probe.describe_hit(hit).replace(PLACEHOLDER, '[표시 글자]')}")
        if hit["path"]:
            lines.append(f"      경로: {' › '.join(hit['path'])}")
    places = field_places(run)
    lines.append(f"  → 요약을 넣을 칸: {describe_place(places[0])}" if places else "  → 입력칸에서 찾지 못했습니다")
    if not run["hits"]:
        lines += [f"   - {emr_probe.describe_window(window)}" for window in run["windows"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- Windows: 찾기 · 읽기 · 덧붙이기
def same_identifier(saved: str, actual: str) -> bool:
    """저장한 (가린) 클래스 · AutomationId 와 지금 화면의 값이 같은가. '[환자 이름]'이 든 값은 그 자리를 아무 글자로 본다."""
    masked = emr_probe.mask_identifier(actual)
    if PLACEHOLDER not in saved:
        return masked == saved
    return re.fullmatch(".+?".join(re.escape(part) for part in saved.split(PLACEHOLDER)), masked) is not None


def _matches(step: list, identity: tuple[str, str, str]) -> bool:
    control, class_name, automation_id = identity
    return step[0] == control and same_identifier(step[1], class_name) and same_identifier(step[2], automation_id)


def _pick(step: list, kids: list, identity) -> object | None:
    """경로 한 칸: 점검 때와 같은 순번의 요소. 순번이 바뀌었으면 같은 모양의 요소가 하나뿐일 때만."""
    index = step[3]
    if index < len(kids) and _matches(step, identity(kids[index])):
        return kids[index]
    same = [kid for kid in kids if _matches(step, identity(kid))]
    return same[0] if len(same) == 1 else None


def _squash(text: str) -> str:
    return "".join(text.split())


def _joined(before: str, text: str) -> str:
    """칸 끝에 덧붙일 글: 앞 내용과 빈 줄 하나를 두고, 줄바꿈은 Windows 식(CRLF)으로."""
    body = "\r\n".join(text.strip("\r\n").splitlines())
    if not before.strip():
        return body
    tail = before[len(before.rstrip("\r\n")):]
    breaks = tail.count("\n") or tail.count("\r")
    return "\r\n" * max(0, 2 - breaks) + body


def _verified(before: str, after: str, text: str) -> dict:
    if _squash(after).startswith(_squash(before)) and _squash(after).endswith(_squash(text)):
        return {"status": "ok"}
    return {"status": "failed", "error": REASONS["failed"]}


class _Screen:
    """씨차트 창에서 정한 자리를 찾아 읽고, 진료 기록 칸 끝에 덧붙인다 (따로 띄운 프로세스 안에서만)."""

    def __init__(self, exclude_pids=()) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.win = emr_probe._Win32()
        u = self.win.user32
        u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        u.GetWindowLongW.restype = ctypes.c_long
        u.IsWindowEnabled.argtypes = [wintypes.HWND]
        u.RealGetWindowClassW.argtypes = [wintypes.HWND, wintypes.LPWSTR, wintypes.UINT]
        u.RealGetWindowClassW.restype = wintypes.UINT
        self.excluded = {os.getpid(), *exclude_pids}
        self._tops: list | None = None
        self._uia = None
        self._uia_tried = False

    def uia(self):
        if not self._uia_tried:
            self._uia_tried = True
            try:
                self._uia = emr_probe._Uia()
            except Exception:
                self._uia = None
        return self._uia

    def tops(self, place: dict) -> list:
        """정한 프로그램 · 창 종류의 창 (화면 맨 앞 창부터)."""
        if self._tops is None:
            self._tops = [top for top in self.win.top_windows() if top.pid not in self.excluded]
        return [top for top in self._tops
                if top.exe.lower() == place["exe"].lower() and same_identifier(place["window_class"], top.class_name)]

    def find(self, place: dict, top):
        """정한 자리 (화면 읽기 요소 또는 창 핸들). 화면 구조가 달라 찾지 못하면 None."""
        if place["kind"] == "title":
            return top.hwnd
        key = place["key"]
        if place["method"] != "UIA":
            return self._find_hwnd(top.hwnd, key)
        uia = self.uia()
        if uia is None or not key:
            return None
        node = uia.root(top.hwnd)
        if not _matches(key[0], uia.identity(node)):
            return None
        for step in key[1:]:
            node = _pick(step, uia.children(node), uia.identity)
            if node is None:
                return None
        return node

    def _find_hwnd(self, hwnd: int, key: list) -> int | None:
        """창 글자로 읽은 자리: 점검(emr_probe._Win32.walk)과 같은 순서로 칸을 세어 같은 경로의 칸."""
        win = self.win
        paths: dict[int, tuple] = {hwnd: ()}
        counts: dict[int, int] = {}
        for child in win._enum(hwnd):
            parent = win.user32.GetAncestor(child, 1)  # GA_PARENT
            if parent not in paths:
                parent = hwnd
            index = counts[parent] = counts.get(parent, -1) + 1
            control_id = win.user32.GetDlgCtrlID(child)
            path = paths[parent] + ((("HWND", win.class_name(child), str(control_id) if control_id else ""), index),)
            paths[child] = path
            if len(path) == len(key) and all(step[3] == i and _matches(step, ident) for step, (ident, i) in zip(key, path)):
                return child
        return None

    def text(self, place: dict, top, target) -> str:
        source = place["source"]
        if source == "title":
            return self.win.title(top.hwnd)
        if place["method"] != "UIA":
            return self.win.window_text(target)
        uia = self.uia()
        if source == "text":
            hwnd = emr_probe._as_int(uia.cached(target, "hwnd")) or 0
            return self.win.window_text(hwnd) if hwnd else ""
        return emr_probe._as_str(uia.cached(target, source))

    def patient(self, places: list[dict]) -> str:
        """정한 자리에서 읽은 환자 이름 (화면 맨 앞 씨차트 창부터). 못 읽으면 ""."""
        for place in places:
            for top in self.tops(place):
                try:
                    target = self.find(place, top)
                    name = read_name(place["template"], self.text(place, top, target)) if target is not None else ""
                except Exception:
                    continue
                if name:
                    return name
        return ""

    # ---- 진료 기록 칸 ----------------------------------------------------------
    def _field(self, field: dict):
        for top in self.tops(field):
            try:
                target = self.find(field, top)
            except Exception:
                continue
            if target is not None:
                return target
        return None

    def check(self, field: dict) -> dict:
        """진료 기록 칸을 찾아 덧붙일 수 있는지만 본다 (쓰지 않는다)."""
        target = self._field(field)
        if target is None:
            return {"status": "no_field"}
        writer, reason = self._writer(field, target)
        return {"status": "ok"} if writer is not None else {"status": "refused", "error": reason}

    def insert(self, names: list[dict], field: dict, text: str, expected: str) -> dict:
        """지금 씨차트에 열린 환자가 expected 이면 진료 기록 칸 끝에 text 를 덧붙인다."""
        current = self.patient(names)
        if not current:
            return {"status": "no_patient"}
        if not same_patient(current, expected):
            return {"status": "mismatch"}
        target = self._field(field)
        if target is None:
            return {"status": "no_field"}
        writer, reason = self._writer(field, target)
        if writer is None:
            return {"status": "refused", "error": reason}
        return writer(text)

    def _writer(self, field: dict, target):
        """(글을 덧붙이는 함수, "") 또는 (None, 넣을 수 없는 이유)."""
        hwnd = target if field["method"] != "UIA" else emr_probe._as_int(self.uia().cached(target, "hwnd")) or 0
        if hwnd and self._edit_like(hwnd):
            reason = self._edit_issue(hwnd)
            return (None, reason) if reason else ((lambda text: self._append_edit(hwnd, text)), "")
        if field["method"] == "UIA":
            pattern = self._value_pattern(target)
            if pattern is None:
                return None, "글을 넣을 수 없는 칸입니다"
            if pattern.CurrentIsReadOnly:
                return None, "읽기 전용 칸입니다"
            return (lambda text: self._append_value(pattern, text)), ""
        return None, "글을 넣을 수 없는 칸입니다"

    def _edit_like(self, hwnd: int) -> bool:
        buf = self.ctypes.create_unicode_buffer(256)
        self.win.user32.RealGetWindowClassW(hwnd, buf, 256)
        return _edit_class(self.win.class_name(hwnd)) or _edit_class(buf.value)

    def _edit_issue(self, hwnd: int) -> str:
        style = self.win.user32.GetWindowLongW(hwnd, GWL_STYLE) & 0xFFFFFFFF
        if not style & ES_MULTILINE:
            return "한 줄짜리 칸이라 넣지 않습니다"
        if style & ES_READONLY or not self.win.user32.IsWindowEnabled(hwnd):
            return "지금은 쓸 수 없는 칸입니다 (읽기 전용)"
        return ""

    def _send(self, hwnd: int, message: int, wparam: int = 0, lparam: int = 0) -> int | None:
        """응답 없는 프로그램이면 2초 만에 포기한다 (None)."""
        result = self.ctypes.c_size_t()
        if not self.win.user32.SendMessageTimeoutW(hwnd, message, wparam, lparam, SMTO_ABORTIFHUNG, 2000,
                                                   self.ctypes.byref(result)):
            return None
        return int(result.value)

    def _append_edit(self, hwnd: int, text: str) -> dict:
        """Win32 입력칸: 끝에 커서를 두고 '선택 바꾸기'로 넣는다 (손으로 붙여 넣은 것과 같아 씨차트가 바뀐 줄 안다)."""
        before = self.win.window_text(hwnd, MAX_NOTE)
        payload = _joined(before, text)
        length = self._send(hwnd, WM_GETTEXTLENGTH)
        if length is None:
            return {"status": "failed", "error": "씨차트가 응답하지 않습니다"}
        limit = self._send(hwnd, EM_GETLIMITTEXT)
        if limit and length + len(payload) > limit:
            return {"status": "refused", "error": "칸에 남은 자리가 모자랍니다"}
        selection = self._send(hwnd, EM_GETSEL)
        self._send(hwnd, EM_SETSEL, length, length)
        buf = self.ctypes.create_unicode_buffer(payload)
        self._send(hwnd, EM_REPLACESEL, 1, self.ctypes.addressof(buf))  # 1: 되돌리기(Ctrl+Z) 할 수 있게
        if selection is not None and length < 0x10000:  # 의사가 두었던 커서 자리로
            self._send(hwnd, EM_SETSEL, selection & 0xFFFF, selection >> 16 & 0xFFFF)
        return _verified(before, self.win.window_text(hwnd, MAX_NOTE), text)

    def _value_pattern(self, element):
        uia = self.uia()
        try:
            unknown = element.GetCurrentPattern(UIA_VALUE_PATTERN)
            return unknown.QueryInterface(uia.module.IUIAutomationValuePattern) if unknown else None
        except Exception:
            return None

    def _append_value(self, pattern, text: str) -> dict:
        """창 핸들이 없는 칸(화면 읽기로만 보이는 칸): 지금 글 + 덧붙일 글로 바꾼다."""
        before = pattern.CurrentValue or ""
        combined = before + _joined(before, text)
        if len(combined) > MAX_NOTE:
            return {"status": "refused", "error": "칸에 남은 자리가 모자랍니다"}
        pattern.SetValue(combined)
        return _verified(before, pattern.CurrentValue or "", text)


# ---------------------------------------------------------------------------- 앱에서 부르기 (따로 띄운 프로세스)
def _scrub(text: str) -> str:
    """오류 글: 한글(이름 · 진료 내용)과 긴 숫자를 가린다."""
    return emr_probe.mask_identifier(text, None, 300)


def _request(request: dict, cancel: threading.Event | None, timeout: float) -> dict:
    payload = json.dumps({**request, "exclude_pids": [os.getpid()]}, ensure_ascii=True).encode("ascii")
    done = procutil.run([emr_probe.child_python(), "-m", "clinic_scribe.emr_link"], stdin_data=payload,
                        cwd=str(emr_probe.REPO), timeout=timeout, cancel=cancel)
    try:
        result = json.loads(done.stdout.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        result = None
    if not isinstance(result, dict):
        raise EmrLinkError("씨차트 연결 프로세스가 결과를 내지 못했습니다. " + _scrub(procutil.tail(done.stderr)))
    return result


def _status(result: dict) -> dict:
    status = result.get("status")
    return {"status": status if status == "ok" or status in REASONS else "error", "error": str(result.get("error") or "")}


def run_read(places: list[dict], cancel: threading.Event | None = None) -> str:
    """씨차트에 지금 열린 환자 이름 (못 읽으면 "")."""
    patient = _request({"command": "read", "names": places}, cancel, READ_TIMEOUT_S).get("patient")
    return patient if isinstance(patient, str) else ""


def run_check(field: dict, cancel: threading.Event | None = None) -> dict:
    """진료 기록 칸을 찾아 덧붙일 수 있는지 (쓰지 않는다). {"status": ok · no_field · refused · error, "error"}."""
    return _status(_request({"command": "check", "field": field}, cancel, READ_TIMEOUT_S))


def run_insert(names: list[dict], field: dict, text: str, expected: str,
               cancel: threading.Event | None = None) -> dict:
    """같은 환자인지 확인하고 진료 기록 칸 끝에 text 를 덧붙인다.

    {"status": ok · no_patient · mismatch · no_field · refused · failed · error, "error"}. 씨차트의 저장은 누르지 않는다.
    """
    request = {"command": "insert", "names": names, "field": field, "text": text, "expected": expected}
    return _status(_request(request, cancel, WRITE_TIMEOUT_S))


def _handle(request: dict) -> dict:
    if not supported():
        return {"status": "error", "error": "씨차트 연결은 Windows 에서만 할 수 있습니다."}
    screen = _Screen([int(pid) for pid in request.get("exclude_pids", [])])
    command = request.get("command")
    names = usable_names(request.get("names"))
    if command == "read":
        return {"status": "ok", "patient": screen.patient(names)}
    field = usable_field(request.get("field"))
    if command not in ("check", "insert"):
        return {"status": "error", "error": "알 수 없는 요청입니다."}
    if field is None:
        return {"status": "no_field"}
    if command == "check":
        return screen.check(field)
    return screen.insert(names, field, str(request.get("text", "")), str(request.get("expected", "")))


def main() -> int:
    """표준 입력 {"command": read · check · insert, ...} → 표준 출력 결과(JSON, ASCII)."""
    sys.coinit_flags = 0  # COINIT_MULTITHREADED (emr_probe.main 과 같다)
    try:
        result = _handle(json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}"))
    except Exception as exc:
        result = {"status": "error", "error": _scrub(f"{type(exc).__name__}: {exc}")}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=True).encode("ascii"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
