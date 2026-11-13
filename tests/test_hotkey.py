"""녹음 버튼(전역 단축키): 키 이름 읽기, 쓸 수 없는 키 거르기, Windows·macOS 에서 등록·해제 (Windows 는 키를 실제로 눌러 본다)."""

import sys
import threading

import pytest

from clinic_scribe import hotkey


def test_parse_and_describe_keys():
    assert hotkey.parse("F13") == (frozenset(), "F13")
    assert hotkey.parse("Ctrl+Alt+r") == (frozenset({"Ctrl", "Alt"}), "R")
    assert hotkey.parse("Pause") == (frozenset(), "Pause") and hotkey.parse("Ctrl+F5")[1] == "F5"
    for bad, reason in (("A", "글을 쓸 때마다"), ("F5", "다른 프로그램"), ("Shift+F5", "다른 프로그램"), ("", "정해 주세요"),
                        ("Ctrl+Esc", "쓸 수 없는 키"), ("Hyper+F13", "쓸 수 없는 키")):
        with pytest.raises(ValueError, match=reason):
            hotkey.parse(bad)
    assert hotkey.describe("Alt+Ctrl+F13") == ("⌥⌘F13" if sys.platform == "darwin" else "Ctrl+Alt+F13")


@pytest.mark.skipif(sys.platform not in ("win32", "darwin"), reason="녹음 버튼은 Windows·macOS")
def test_register_and_release_again():
    for _ in range(2):  # 푼 키는 다시 등록된다 (설정 창을 열고 닫을 때)
        key = hotkey.GlobalHotkey("Ctrl+Alt+F14", lambda: None)
        key.start()
        key.stop()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows: 키를 실제로 눌러 본다")
def test_windows_key_press_fires_and_a_taken_key_is_reported():
    import ctypes

    user32 = ctypes.WinDLL("user32")
    pressed = threading.Event()
    key = hotkey.GlobalHotkey("F13", pressed.set)
    key.start()
    try:
        with pytest.raises(hotkey.HotkeyError, match="이미 쓰고"):
            hotkey.GlobalHotkey("F13", lambda: None).start()  # 다른 프로그램이 이미 쓰는 키
        user32.keybd_event(0x7C, 0, 0, 0)  # F13 누르고
        user32.keybd_event(0x7C, 0, 2, 0)  # 떼기 (KEYEVENTF_KEYUP)
        assert pressed.wait(5)
    finally:
        key.stop()


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="Linux 는 아직 녹음 버튼이 없다")
def test_mac_has_no_f21():
    if sys.platform == "darwin":
        with pytest.raises(hotkey.HotkeyError, match="Mac 키보드"):
            hotkey.GlobalHotkey("F21", lambda: None)
    else:
        hotkey.GlobalHotkey("F21", lambda: None)  # Windows 는 F24 까지


# ---------------------------------------------------------------------------- 설정 창
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("PySide6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


@pytest.mark.gui
def test_settings_record_button_tab_checks_the_key(qapp):
    from PySide6.QtGui import QKeySequence

    from clinic_scribe.config import Settings
    from clinic_scribe.ui.settings_dialog import SettingsDialog

    settings = Settings()
    dialog = SettingsDialog(settings, hotkey_issue="F13 키를 쓸 수 없습니다: 다른 프로그램이 이미 쓰고 있습니다")
    assert "이미 쓰고" in dialog.hotkey_status.text()  # 지금 키를 등록하지 못한 이유
    dialog.hotkey_edit.setKeySequence(QKeySequence("F5"))
    assert "다른 프로그램" in dialog.hotkey_status.text()
    dialog.apply()
    assert settings.record_hotkey == "F13"  # 쓸 수 없는 키는 저장하지 않는다
    dialog.hotkey_edit.setKeySequence(QKeySequence("Ctrl+Alt+R"))
    dialog.auto_copy.setChecked(False)
    dialog.apply()
    assert settings.record_hotkey == "Ctrl+Alt+R" and settings.auto_copy_summary is False and settings.hotkey_enabled
