"""녹음 버튼(전역 단축키): 씨차트 등 다른 프로그램이 앞에 있어도 키 하나로 녹음을 시작·종료한다.

USB 버튼·풋페달은 컴퓨터에 키보드로 보이고, 전용 프로그램으로 누를 때 보낼 키를 정한다. F13 처럼 보통 키보드에 없는 키로
정하면 다른 프로그램과 겹치지 않는다. 키 이름은 Qt 표기("F13", "Ctrl+Alt+R")를 쓴다 (macOS 의 Ctrl 은 ⌘, Meta 는 ⌃).

- Windows: RegisterHotKey. 따로 도는 스레드의 메시지 루프가 WM_HOTKEY 를 받는다 (누르고 있어도 한 번만, MOD_NOREPEAT).
- macOS: Carbon RegisterEventHotKey. 손쉬운 사용 권한 없이 쓸 수 있고, 앱의 이벤트 루프(화면 스레드)에서 불린다.
- Linux: 아직 지원하지 않는다.
callback 은 Windows 에서는 다른 스레드에서 불리므로, 화면을 바꾸려면 Qt 신호로 넘긴다.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Callable

DEFAULT_KEY = "F13"
MODIFIERS = ("Ctrl", "Alt", "Shift", "Meta")
_FUNCTION = {f"F{n}" for n in range(1, 25)}
_ALONE = {f"F{n}" for n in range(13, 25)} | {"Pause", "ScrollLock"}  # 따로 눌러도 되는 키 (보통 키보드에 없거나 안 쓰는 키)

# Windows 가상 키 코드
_WIN_KEYS = {**{f"F{n}": 0x6F + n for n in range(1, 25)}, "Pause": 0x13, "ScrollLock": 0x91,
             **{chr(c): c for c in range(ord("A"), ord("Z") + 1)}, **{str(d): 0x30 + d for d in range(10)}}
_WIN_MODS = {"Alt": 0x1, "Ctrl": 0x2, "Shift": 0x4, "Meta": 0x8}
# macOS 키 코드 (Events.h 의 kVK_*). F21~F24·Pause·ScrollLock 은 mac 키보드에 없다
_MAC_KEYS = {
    "F1": 0x7A, "F2": 0x78, "F3": 0x63, "F4": 0x76, "F5": 0x60, "F6": 0x61, "F7": 0x62, "F8": 0x64, "F9": 0x65,
    "F10": 0x6D, "F11": 0x67, "F12": 0x6F, "F13": 0x69, "F14": 0x6B, "F15": 0x71, "F16": 0x6A, "F17": 0x40,
    "F18": 0x4F, "F19": 0x50, "F20": 0x5A,
    "A": 0x00, "S": 0x01, "D": 0x02, "F": 0x03, "H": 0x04, "G": 0x05, "Z": 0x06, "X": 0x07, "C": 0x08, "V": 0x09,
    "B": 0x0B, "Q": 0x0C, "W": 0x0D, "E": 0x0E, "R": 0x0F, "Y": 0x10, "T": 0x11, "1": 0x12, "2": 0x13, "3": 0x14,
    "4": 0x15, "6": 0x16, "5": 0x17, "9": 0x19, "7": 0x1A, "8": 0x1C, "0": 0x1D, "O": 0x1F, "U": 0x20, "I": 0x22,
    "P": 0x23, "L": 0x25, "J": 0x26, "K": 0x28, "N": 0x2D, "M": 0x2E,
}
_MAC_MODS = {"Ctrl": 0x0100, "Shift": 0x0200, "Alt": 0x0800, "Meta": 0x1000}  # cmdKey · shiftKey · optionKey · controlKey


class HotkeyError(Exception):
    pass


def parse(text: str) -> tuple[frozenset[str], str]:
    """"Ctrl+Alt+R" → ({"Ctrl", "Alt"}, "R"). 쓸 수 없는 키면 ValueError (화면에 그대로 보일 설명)."""
    parts = [p.strip() for p in (text or "").split(",")[0].split("+") if p.strip()]
    if not parts:
        raise ValueError("키를 정해 주세요")
    *mods, key = parts
    key = key.upper() if len(key) == 1 else key
    unknown = [m for m in mods if m not in MODIFIERS]
    if unknown or key not in _WIN_KEYS:
        raise ValueError(f"'{text}' 는 쓸 수 없는 키입니다. F13~F24, Pause, ScrollLock 이나 Ctrl·Alt 를 함께 누르는 키를 쓰세요")
    held = frozenset(mods)
    if key not in _ALONE and not held & {"Ctrl", "Alt", "Meta"}:
        if key in _FUNCTION:
            raise ValueError(f"{key} 는 씨차트 같은 다른 프로그램이 쓰는 경우가 많습니다. F13~F24 를 쓰거나 Ctrl·Alt 와 함께 쓰세요")
        raise ValueError(f"'{key}' 하나만으로는 글을 쓸 때마다 녹음이 시작됩니다. Ctrl·Alt 와 함께 쓰세요")
    return held, key


def describe(text: str) -> str:
    """화면에 보일 키 이름 (macOS 는 ⌘·⌥·⌃·⇧)."""
    mods, key = parse(text)
    if sys.platform == "darwin":
        symbols = {"Meta": "⌃", "Alt": "⌥", "Shift": "⇧", "Ctrl": "⌘"}
        return "".join(symbols[m] for m in ("Meta", "Alt", "Shift", "Ctrl") if m in mods) + key
    return "+".join([m for m in ("Ctrl", "Alt", "Shift", "Meta") if m in mods] + [key])


class GlobalHotkey:
    """start() 로 등록하고 stop() 으로 푼다. 다른 프로그램이 이미 쓰는 키면 start() 가 HotkeyError."""

    def __init__(self, text: str, callback: Callable[[], None]) -> None:
        self.mods, self.key = parse(text)
        self.text = text
        self.callback = callback
        if sys.platform == "win32":
            self._impl: _WindowsHotkey | _MacHotkey = _WindowsHotkey(self)
        elif sys.platform == "darwin":
            if self.key not in _MAC_KEYS:
                raise HotkeyError(f"{self.key} 는 Mac 키보드에 없는 키입니다. F13~F20 이나 ⌘·⌥ 와 함께 누르는 키를 쓰세요")
            self._impl = _MacHotkey(self)
        else:
            raise HotkeyError("Linux 에서는 아직 녹음 버튼을 쓸 수 없습니다")

    def start(self) -> None:
        self._impl.start()

    def stop(self) -> None:
        self._impl.stop()

    def _fire(self) -> None:
        try:
            self.callback()
        except Exception:  # 단축키 처리 실패가 메시지 루프를 멈추지 않게
            pass


class _WindowsHotkey:
    WM_HOTKEY, WM_QUIT, MOD_NOREPEAT = 0x0312, 0x0012, 0x4000

    def __init__(self, owner: GlobalHotkey) -> None:
        self.owner = owner
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._error = ""

    def start(self) -> None:
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), name="clinic-scribe-hotkey", daemon=True)
        self._thread.start()
        if not ready.wait(5):
            raise HotkeyError("단축키를 등록하지 못했습니다 (시간 초과)")
        if self._error:
            self._thread.join(2)
            raise HotkeyError(self._error)

    def _run(self, ready: threading.Event) -> None:
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT,
                                        wintypes.UINT]
        self._thread_id = kernel32.GetCurrentThreadId()
        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)  # 이 스레드의 메시지 큐를 만든다 (stop 의 WM_QUIT 을 받게)
        owner = self.owner
        mods = sum(_WIN_MODS[m] for m in owner.mods) | self.MOD_NOREPEAT
        if not user32.RegisterHotKey(None, 1, mods, _WIN_KEYS[owner.key]):
            code = ctypes.get_last_error()
            self._error = (f"{owner.text} 키를 쓸 수 없습니다: 다른 프로그램이 이미 쓰고 있습니다 (오류 {code}). 다른 키로 정하세요"
                           if code == 1409 else f"{owner.text} 키를 등록하지 못했습니다 (오류 {code})")
            ready.set()
            return
        ready.set()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == self.WM_HOTKEY:
                    owner._fire()
        finally:
            user32.UnregisterHotKey(None, 1)

    def stop(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.PostThreadMessageW(self._thread_id, self.WM_QUIT, 0, 0)
        self._thread.join(2)


def _fourcc(code: bytes) -> int:
    return int.from_bytes(code, "big")


class _MacHotkey:
    """Carbon RegisterEventHotKey (손쉬운 사용 권한이 필요 없다). 앱의 이벤트 루프가 화면 스레드에서 처리한다."""

    _HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)

    class _EventTypeSpec(ctypes.Structure):
        _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]

    class _HotKeyID(ctypes.Structure):
        _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]

    def __init__(self, owner: GlobalHotkey) -> None:
        self.owner = owner
        self._handler = ctypes.c_void_p()
        self._hotkey = ctypes.c_void_p()
        self._proc = None

    def _carbon(self):
        carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
        carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
        carbon.InstallEventHandler.argtypes = [ctypes.c_void_p, self._HANDLER, ctypes.c_ulong,
                                               ctypes.POINTER(self._EventTypeSpec), ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_void_p)]
        carbon.InstallEventHandler.restype = ctypes.c_int32
        carbon.RegisterEventHotKey.argtypes = [ctypes.c_uint32, ctypes.c_uint32, self._HotKeyID, ctypes.c_void_p,
                                               ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        carbon.RegisterEventHotKey.restype = ctypes.c_int32
        carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
        carbon.RemoveEventHandler.argtypes = [ctypes.c_void_p]
        return carbon

    def start(self) -> None:
        carbon = self._carbon()
        target = carbon.GetApplicationEventTarget()
        spec = self._EventTypeSpec(_fourcc(b"keyb"), 5)  # kEventClassKeyboard · kEventHotKeyPressed
        self._proc = self._HANDLER(lambda _call, _event, _data: self.owner._fire() or 0)
        status = carbon.InstallEventHandler(target, self._proc, 1, ctypes.byref(spec), None, ctypes.byref(self._handler))
        if status != 0:
            raise HotkeyError(f"단축키를 받을 준비를 하지 못했습니다 (오류 {status})")
        owner = self.owner
        mods = sum(_MAC_MODS[m] for m in owner.mods)
        status = carbon.RegisterEventHotKey(_MAC_KEYS[owner.key], mods, self._HotKeyID(_fourcc(b"clsc"), 1), target, 0,
                                            ctypes.byref(self._hotkey))
        if status != 0:
            carbon.RemoveEventHandler(self._handler)
            self._handler = ctypes.c_void_p()
            raise HotkeyError(f"{describe(owner.text)} 키를 쓸 수 없습니다: 다른 프로그램이 이미 쓰고 있을 수 있습니다 (오류 {status})")

    def stop(self) -> None:
        carbon = self._carbon()
        if self._hotkey:
            carbon.UnregisterEventHotKey(self._hotkey)
            self._hotkey = ctypes.c_void_p()
        if self._handler:
            carbon.RemoveEventHandler(self._handler)
            self._handler = ctypes.c_void_p()
