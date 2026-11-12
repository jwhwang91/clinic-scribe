"""씨차트 연결 점검: 환자 이름을 찾은 자리, 가린 결과(이름·다른 환자·진료 내용이 남지 않음), 점검 창.

화면을 실제로 읽는 부분은 Windows 에서만 확인한다 (가짜 진료 차트 창을 다른 프로세스로 띄운다).
가상 환자 이름만 쓴다.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from clinic_scribe import emr_probe
from clinic_scribe.procutil import ProcessResult

REPO = Path(__file__).resolve().parent.parent
HELPER = Path(__file__).with_name("fake_emr_window.py")
NAMES = ("김민준", "이서연", "박지호")
CHART_NUMBER = "20240917"
MEMO = "어제부터 명치가 아프고 검은 변을 봤어요"


def fake_scan(name: str, patient: str, waiting=NAMES) -> dict:
    """씨차트 창 하나를 화면 읽기로 읽었다고 치고 scan 과 같은 모양의 결과를 만든다 (가리기 전 요소 → window_result)."""
    name_re = emr_probe.name_pattern(name)
    top = emr_probe.TopWindow(101, 4321, "SeeChart.exe", "TfrmMain", f"씨차트 - {patient}(남/45세) 차트번호 {CHART_NUMBER}",
                              elevated=False, rect=[0, 0, 1280, 800])

    def node(path, control, class_name="", automation_id="", **texts):
        return {"depth": len(path) - 1, "control": control, "class": class_name, "id": automation_id,
                "framework": "Win32", "rect": [120, 88, 160, 24], "texts": texts, "path": path}

    root = (("Window", "TfrmMain", "", 0),)
    panel = root + (("Pane", "TPanel", "pnlPatient", 0),)
    waiting_list = root + (("List", "TListBox", "lstWaiting", 1),)
    nodes = [
        node(root, "Window", "TfrmMain", name=top.title),
        node(panel, "Pane", "TPanel", "pnlPatient"),
        node(panel + (("Text", "TLabel", "", 0),), "Text", "TLabel", name="환자명"),
        node(panel + (("Edit", "TEdit", "edtName", 1),), "Edit", "TEdit", "edtName", name="환자명", value=patient, text=patient),
        node(panel + (("Edit", "TEdit", "edtChart", 2),), "Edit", "TEdit", "edtChart", name="차트번호", value=CHART_NUMBER),
        node(panel + (("Document", "TMemo", "memSymptom", 3),), "Document", "TMemo", "memSymptom", value=MEMO),
        node(waiting_list, "List", "TListBox", "lstWaiting", name="대기 환자"),
    ]
    nodes += [node(waiting_list + (("ListItem", "", "", i),), "ListItem", name=person) for i, person in enumerate(waiting)]
    entry, hits = emr_probe.window_result(top, nodes, name_re, method="UIA", truncated=False, focus=True)
    result = emr_probe.new_result()
    result.update(uia="ok", elevated=False, elapsed=4.2)
    emr_probe.add_window(result, entry, hits)
    chat = emr_probe.TopWindow(202, 999, "KakaoTalk.exe", "EVA_Window", "이서연 님과의 대화")
    emr_probe.add_window(result, emr_probe.skipped_window(chat, name_re, False), [])
    return result


def assert_nothing_private(text: str) -> None:
    for private in (*NAMES, CHART_NUMBER, "명치", "검은 변"):
        assert private not in text


def test_mask_keeps_screen_words_and_hides_patient_details():
    name_re = emr_probe.name_pattern("김민준")
    assert emr_probe.mask(f"씨차트 - 김민준(남/45세) 차트번호 {CHART_NUMBER}", name_re) == "씨차트 - [환자 이름](남/##세) 차트번호 ########"
    assert emr_probe.mask("김 민 준 님", name_re) == "[환자 이름] 님"  # 띄어 써도 찾는다
    assert emr_probe.mask("대기환자목록: 이서연, 박지호") == "대기환자목록: ○○○, ○○○"
    assert emr_probe.mask("남세일") == "○○○"  # 한 글자 낱말(남·세·일)을 이어 붙인 이름은 남기지 않는다
    assert emr_probe.mask("JOHN SMITH\n010-1234-5678 OK") == "xxxx xxxxx ###-####-#### OK"
    assert emr_probe.mask(MEMO) == "○○○○ ○○○ ○○○ ○○ ○○ ○○○"
    assert emr_probe.mask("가" * 200).endswith("…(200자)")
    assert emr_probe.mask_identifier("edtName1", name_re) == "edtName1"
    assert emr_probe.mask_identifier("row_20240917_김민준", name_re) == "row_########_[환자 이름]"
    assert emr_probe.name_pattern("김") is None
    assert [emr_probe.looks_like_name(t) for t in ("이서연", "이서연 님", "박지호(여/7세)", "진료실", "환자명", "접수 완료")] == [
        True, True, True, False, False, False]


def test_finds_where_the_name_is_and_masks_everything_else():
    run = fake_scan("김민준", "김민준")
    assert [(h["control"], h["sources"]) for h in run["hits"]] == [
        ("Window", ["title"]),  # 창 제목
        ("Edit", ["value"]),  # 환자명 칸 (같은 글자인 창 글자는 한 번만)
        ("ListItem", ["name"]),  # 대기 목록의 그 환자 줄
    ]
    edit = run["hits"][1]
    assert edit["path"] == ["Window TfrmMain[0]", "Pane TPanel #pnlPatient[0]", "Edit TEdit #edtName[1]"]
    assert run["windows"][0]["title"] == "씨차트 - [환자 이름](남/##세) 차트번호 ########"
    nodes = run["windows"][0]["nodes"]
    assert [n["nl"] for n in nodes if n["c"] == "ListItem"] == [False, True, True]  # 다른 환자는 '이름 모양'으로만
    assert any(n["t"].get("name") == "환자명" for n in nodes)  # 화면 낱말은 그대로
    assert run["windows"][1]["scanned"] is False and run["windows"][1]["title"] == "○○○ ○○○ ○○"
    assert_nothing_private(json.dumps(run, ensure_ascii=False))


def test_same_place_for_two_patients_means_the_name_can_be_read():
    first, second = fake_scan("김민준", "김민준"), fake_scan("이서연", "이서연")
    assert emr_probe.verdict([first])[0] == "found"
    stable = emr_probe.stable_locations([first, second])
    assert [h["control"] for h in stable] == ["Window", "Edit"]  # 대기 목록 줄은 환자마다 자리가 다르다
    assert emr_probe.verdict([first, second])[0] == "done"
    typo = fake_scan("김민중", "이서연")  # 못 찾은 점검은 같은 자리 비교에서 뺀다
    assert emr_probe.stable_locations([first, typo, second]) == stable and emr_probe.verdict([first, typo])[0] == "missing"
    report = emr_probe.format_report([first, second])
    assert "같은 자리 2곳" in report and "★ 지금 환자 이름" in report and "edtName" in report
    assert "SeeChart.exe · 입력칸 (TEdit · #edtName) “[환자 이름]” · 화면 읽기(값)" in report
    assert "KakaoTalk.exe" in report and "읽지 않음" in report
    assert_nothing_private(report)


def test_missing_name_explains_admin_rights_and_remote_desktop():
    name_re = emr_probe.name_pattern("김민준")
    result = emr_probe.new_result()
    result.update(uia="ok", elevated=False)
    chart = emr_probe.TopWindow(1, 10, "SeeChart.exe", "TfrmMain", "씨차트", elevated=None)
    entry, hits = emr_probe.window_result(chart, [{"depth": 0, "control": "Window", "texts": {}, "path": ()}], name_re,
                                          method="UIA", truncated=False, focus=True)
    emr_probe.add_window(result, entry, hits)
    kind, message = emr_probe.verdict([result])
    assert kind == "missing" and "관리자 권한으로 실행" in message and "원격 데스크톱" not in message
    remote = emr_probe.TopWindow(2, 11, "mstsc.exe", "TscShellContainerClass", "원격 데스크톱 연결")
    emr_probe.add_window(result, *emr_probe.window_result(remote, [], name_re, method="UIA", truncated=False))
    assert "원격 데스크톱 창 안에서" in emr_probe.verdict([result])[1]


def test_run_probe_sends_the_name_on_stdin_not_the_command_line(monkeypatch):
    calls = []

    def fake_run(cmd, stdin_data=None, cwd=None, timeout=None, cancel=None, **_):
        calls.append((cmd, stdin_data, cwd))
        if len(calls) == 1:
            return ProcessResult(0, json.dumps(fake_scan("김민준", "김민준")).encode("ascii"), b"")
        return ProcessResult(1, b"", "Traceback (most recent call last):\nValueError: 김민준".encode("utf-8"))

    monkeypatch.setattr(emr_probe.procutil, "run", fake_run)
    assert emr_probe.run_probe("김민준")["hits"]
    cmd, stdin_data, cwd = calls[0]
    assert cmd[-2:] == ["-m", "clinic_scribe.emr_probe"] and Path(cwd) == REPO
    assert not any("김민준" in part for part in cmd) and json.loads(stdin_data)["name"] == "김민준"
    with pytest.raises(RuntimeError) as failed:
        emr_probe.run_probe("김민준")
    assert "[환자 이름]" in str(failed.value) and "김민준" not in str(failed.value)


def test_command_line_reads_stdin_and_prints_masked_json():
    request = json.dumps({"name": "김", "exclude_pids": [], "budget_s": 5}).encode("ascii")
    done = subprocess.run([sys.executable, "-m", "clinic_scribe.emr_probe"], input=request, capture_output=True,
                          cwd=REPO, timeout=60)
    result = json.loads(done.stdout.decode("ascii"))
    assert done.returncode == 0 and result["error"] == "환자 이름을 두 글자 이상 적어 주세요."
    if sys.platform != "win32":
        assert emr_probe.scan("김민준")["error"] == "씨차트 연결 점검은 Windows 에서만 할 수 있습니다."


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 화면 읽기(UI Automation)")
def test_reads_patient_name_from_another_program_window():
    """다른 프로세스의 창(가짜 진료 차트)에서 이름을 찾고, 다른 환자를 열면 같은 자리에서 새 이름을 찾는다."""
    pytest.importorskip("comtypes")
    helper = subprocess.Popen([sys.executable, str(HELPER), "김민준", "이서연", "김민준", "박지호"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert helper.stdout.readline().strip() == b"ready"
        first = emr_probe.run_probe("김민준")
        assert not first["error"] and first["uia"] == "ok", first["error"] or first["uia"]
        found = {(h["control"], tuple(h["sources"])) for h in first["hits"] if h["exe"].lower() == "python.exe"}
        assert ("Window", ("title",)) in found, first["hits"]
        assert any(c == "Edit" and ("value" in s or "text" in s) for c, s in found), first["hits"]  # 환자명 칸
        assert ("ListItem", ("name",)) in found, first["hits"]  # 대기 목록 줄
        report = emr_probe.format_report([first])
        assert "FakeEmrMainWindow" in report and "환자명" in report
        assert_nothing_private(json.dumps(first, ensure_ascii=False) + report)

        helper.stdin.write("show 이서연\n".encode("utf-8"))
        helper.stdin.flush()
        assert helper.stdout.readline().strip() == b"shown"
        second = emr_probe.run_probe("이서연")
        stable = emr_probe.stable_locations([first, second])
        assert {h["control"] for h in stable} >= {"Window", "Edit"}, (first["hits"], second["hits"])
        assert emr_probe.verdict([first, second])[0] == "done"
        assert_nothing_private(json.dumps(second, ensure_ascii=False) + emr_probe.format_report([first, second]))
    finally:
        helper.kill()
        helper.wait(timeout=10)


# ---------------------------------------------------------------------------- 점검 창
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("PySide6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def wait_until(qapp, condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.gui
def test_dialog_checks_two_patients_and_saves_a_masked_report(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from clinic_scribe.ui import theme
    from clinic_scribe.ui.emr_probe_dialog import EmrProbeDialog

    asked = []

    def probe(name, cancel):
        asked.append(name)
        if name == "박지호":  # 멈추기를 누를 때까지 읽는 중
            assert cancel.wait(10)
            from clinic_scribe.procutil import Cancelled
            raise Cancelled()
        if name == "김민중":  # 잘못 적은 이름: 화면 어디에도 없다
            return fake_scan(name, "김민준", waiting=("이서연",))
        return fake_scan(name, name)

    dialog = EmrProbeDialog(probe=probe)
    dialog.setStyleSheet(theme.stylesheet("EmrProbeDialog", EmrProbeDialog.STYLE))
    dialog.show()
    assert dialog.pill.text() == "점검 전" and not dialog.save_btn.isEnabled()
    dialog.name_edit.setText("김")
    dialog.find_btn.click()
    assert dialog.pill.text() == "이름 확인" and not asked

    dialog.name_edit.setText("김민중")
    dialog.find_btn.click()
    assert wait_until(qapp, lambda: dialog.runs)
    assert dialog.pill.text() == "못 찾음" and "관리자 권한" not in dialog.verdict.text()
    assert dialog.name_edit.text() == "김민중" and dialog.badges[0].text() == "1"  # 이름을 고쳐 다시 해 볼 수 있게 둔다

    dialog.name_edit.setText("김민준")
    dialog.find_btn.click()
    assert wait_until(qapp, lambda: len(dialog.runs) == 2)
    shown = dialog.result.toPlainText() + dialog.verdict.text() + dialog.detail.text()
    assert dialog.pill.text() == "찾음" and "[환자 이름]" in shown and dialog.find_btn.text() == "한 번 더 찾기"
    assert dialog.name_edit.text() == "" and dialog.save_btn.isEnabled()
    dialog.name_edit.setText("김 민준")
    dialog.find_btn.click()
    assert dialog.pill.text() == "같은 환자" and len(asked) == 2

    dialog.name_edit.setText("이서연")
    dialog.find_btn.click()
    assert wait_until(qapp, lambda: len(dialog.runs) == 3)
    shown = dialog.result.toPlainText() + dialog.verdict.text()
    assert dialog.pill.text() == "읽을 수 있음" and "같은 자리 2곳" in shown
    assert [badge.text() for badge in dialog.badges] == ["✓", "✓", "✓"]
    assert_nothing_private(shown + dialog.detail.text())

    target = tmp_path / "점검.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    dialog.save_btn.click()
    report = target.read_text(encoding="utf-8-sig")
    assert "★ 지금 환자 이름" in report and str(target) in dialog.detail.text()
    assert_nothing_private(report)
    dialog.copy_btn.click()
    assert qapp.clipboard().text() == report

    dialog.name_edit.setText("박지호")
    dialog.find_btn.click()
    assert dialog.find_btn.text() == "멈추기" and not dialog.name_edit.isEnabled()
    dialog.find_btn.click()
    assert wait_until(qapp, lambda: dialog.pill.text() == "멈춤")
    assert len(dialog.runs) == 3 and dialog.find_btn.text() == "한 번 더 찾기"
    dialog.reset_btn.click()
    assert not dialog.runs and dialog.pill.text() == "점검 전" and dialog.badges[0].text() == "1"
    dialog.close()


@pytest.mark.gui
def test_settings_has_seechart_tab_on_windows(qapp, monkeypatch):
    from PySide6.QtWidgets import QTabWidget

    from clinic_scribe.config import Settings
    from clinic_scribe.ui.settings_dialog import SettingsDialog

    titles = lambda dialog: [dialog.findChild(QTabWidget).tabText(i) for i in range(dialog.findChild(QTabWidget).count())]
    monkeypatch.setattr(emr_probe, "supported", lambda: False)
    assert "씨차트" not in titles(SettingsDialog(Settings()))
    monkeypatch.setattr(emr_probe, "supported", lambda: True)
    dialog = SettingsDialog(Settings())
    assert titles(dialog)[-1] == "씨차트" and dialog.emr_probe_btn.text() == "씨차트 연결 점검…"
