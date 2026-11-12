"""테스트용 가짜 진료 차트 창 (Windows): 씨차트 연결 점검이 다른 프로그램의 창에서 환자 이름을 찾는지 확인한다.

python fake_emr_window.py <지금 환자> <대기 환자>...
가상 환자 이름만 쓴다. 창 제목·환자명 칸·대기 목록에 이름이 보인다. 표준 입력으로 'show <이름>'을 받으면
다른 환자를 연 것처럼 제목과 환자명 칸을 바꾸고 'shown', 'quit' 이면 창을 닫는다. 창이 뜨면 'ready'를 출력한다.
"""

import ctypes
import sys
import threading
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR),
    ]


user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
user32.RegisterClassW.restype = wintypes.ATOM
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                                   wintypes.HINSTANCE, wintypes.LPVOID]
user32.CreateWindowExW.restype = wintypes.HWND
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SendMessageW.restype = LRESULT
user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.restype = LRESULT
user32.PostQuitMessage.argtypes = [ctypes.c_int]
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE

WS_OVERLAPPEDWINDOW, WS_VISIBLE, WS_CHILD, WS_BORDER, WS_VSCROLL = 0x00CF0000, 0x10000000, 0x40000000, 0x00800000, 0x00200000
WM_DESTROY, WM_CLOSE, LB_ADDSTRING = 0x0002, 0x0010, 0x0180


@WNDPROC
def window_proc(hwnd, message, wparam, lparam):
    if message == WM_DESTROY:
        user32.PostQuitMessage(0)
        return 0
    return user32.DefWindowProcW(hwnd, message, wparam, lparam)


def title(patient: str) -> str:
    return f"가상차트 - {patient} (남/45세)"


def main() -> None:
    patient, waiting = sys.argv[1], sys.argv[2:]
    instance = kernel32.GetModuleHandleW(None)
    window_class = WNDCLASSW(lpfnWndProc=window_proc, hInstance=instance, hbrBackground=6,  # COLOR_WINDOW + 1
                             lpszClassName="FakeEmrMainWindow")
    if not user32.RegisterClassW(ctypes.byref(window_class)):
        raise ctypes.WinError(ctypes.get_last_error())

    def create(class_name, text, style, x, y, width, height, parent=None, control_id=None):
        hwnd = user32.CreateWindowExW(0, class_name, text, style, x, y, width, height, parent, control_id, instance, None)
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        return hwnd

    main_window = create("FakeEmrMainWindow", title(patient), WS_OVERLAPPEDWINDOW | WS_VISIBLE, 80, 80, 640, 420)
    child = WS_CHILD | WS_VISIBLE
    create("STATIC", "환자명", child, 16, 16, 80, 24, main_window, 1001)
    name_edit = create("EDIT", patient, child | WS_BORDER, 100, 16, 160, 24, main_window, 1002)
    create("STATIC", "대기 환자", child, 16, 56, 80, 24, main_window, 1003)
    waiting_list = create("LISTBOX", None, child | WS_BORDER | WS_VSCROLL, 100, 56, 160, 120, main_window, 1004)
    for name in waiting:
        buffer = ctypes.create_unicode_buffer(name)
        user32.SendMessageW(waiting_list, LB_ADDSTRING, 0, ctypes.addressof(buffer))

    def listen() -> None:
        for raw in sys.stdin.buffer:
            line = raw.decode("utf-8").strip()
            if line.startswith("show "):
                user32.SetWindowTextW(main_window, title(line[5:]))  # 창을 만든 스레드가 메시지로 바꾼다
                user32.SetWindowTextW(name_edit, line[5:])
                print("shown", flush=True)
            elif line == "quit":
                break
        user32.PostMessageW(main_window, WM_CLOSE, 0, 0)

    threading.Thread(target=listen, daemon=True).start()
    print("ready", flush=True)
    message = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(message))
        user32.DispatchMessageW(ctypes.byref(message))


if __name__ == "__main__":
    main()
