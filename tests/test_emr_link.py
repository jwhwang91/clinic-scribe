"""씨차트 연결: 환자 이름 자리에서 이름만 떼어 읽기, 점검 결과 → 자리, 진료 기록 칸 끝에 덧붙이기, 점검 창 · 설정 · 녹음 버튼.

화면을 실제로 읽고 쓰는 부분은 Windows 에서만 확인한다 (가짜 진료 차트 창을 다른 프로세스로 띄운다).
가상 환자 이름과 가상 요약만 쓴다.
"""

import json
import subprocess
import sys
import time

import pytest

from clinic_scribe import emr_link, emr_probe
from clinic_scribe.config import Settings
from clinic_scribe.procutil import ProcessResult
from tests.test_emr_probe import HELPER, REPO, assert_nothing_private, fake_scan

SUMMARY = "[검토용 초안] 가상 요약\n■ 주호소·현병력\n- 사흘 전부터 두통"


def name_places() -> list[dict]:
    return emr_link.name_places([fake_scan("김민준", "김민준"), fake_scan("이서연", "이서연")])


def marker_scan(in_label: bool = True) -> dict:
    """씨차트 진료 기록 칸(TMemo)에 표시 글자를 적고 찾았다고 치고 scan 과 같은 모양의 결과를 만든다."""
    marker_re = emr_probe.name_pattern(emr_link.MARKER)
    top = emr_probe.TopWindow(101, 4321, "SeeChart.exe", "TfrmMain", "씨차트 - 김민준(남/45세)", elevated=False)
    root = (("Window", "TfrmMain", "", 0),)
    memo = root + (("Document", "TMemo", "memChart", 3),)
    label = root + (("Text", "TLabel", "", 4),)

    def node(path, control, class_name="", automation_id="", **texts):
        return {"depth": len(path) - 1, "control": control, "class": class_name, "id": automation_id,
                "framework": "Win32", "rect": [0, 0, 10, 10], "texts": texts, "path": path}

    nodes = [node(root, "Window", "TfrmMain", name=top.title),
             node(memo, "Document", "TMemo", "memChart", value=f"지난 진료 기록\r\n{emr_link.MARKER}")]
    if in_label:  # 씨차트가 적은 글을 미리 보기 칸에도 보여 준다면: 글자만 보이는 칸은 고르지 않는다
        nodes.append(node(label, "Text", "TLabel", name=emr_link.MARKER))
    entry, hits = emr_probe.window_result(top, nodes, marker_re, method="UIA", truncated=False, focus=True)
    result = emr_probe.new_result()
    result.update(uia="ok", elevated=False, elapsed=3.0)
    emr_probe.add_window(result, entry, hits)
    return result


# ---------------------------------------------------------------------------- 이름 떼어 읽기 · 자리
@pytest.mark.parametrize("template, text, name", [
    ("[환자 이름]", "이서연", "이서연"),
    ("[환자 이름]", "환자명", ""),  # 환자를 열지 않아 칸 이름만 보일 때
    ("[환자 이름]", "", ""),
    ("가상차트 - [환자 이름] (남/##세)", "가상차트 - 이서연 (여/30세)", "이서연"),  # 성별 · 나이는 환자마다 다르다
    ("씨차트 - [환자 이름](남/##세) 차트번호 ########", "씨차트 - 박지호(여/7세) 차트번호 20240918", "박지호"),
    ("[환자 이름] 님", "김 민 준 님", "김 민 준"),
    ("[환자 이름]님", "이준님", "이준"),
    ("[환자 이름] (M/##)", "JOHN SMITH (F/45)", "JOHN SMITH"),
    ("○○○ [환자 이름] " + "○" * 70 + "…(120자)", "진료실 남궁민수 " + "가" * 90, "남궁민수"),  # 줄인 글자는 앞부분만
], ids=["name-only", "label-only", "empty", "title-sex-age", "title-chart-number", "spaced", "suffix", "latin", "cut"])
def test_reads_only_the_name_from_the_place_text(template, text, name):
    assert emr_link.read_name(template, text) == name


def test_same_patient_needs_every_read_to_agree():
    assert emr_link.name_regex("○○ ○○") is None  # 이름 자리가 없는 글자 모양
    assert emr_link.same_patient("김 민준", "김민준") and not emr_link.same_patient("", "")
    assert emr_link.agreed_name(["김민준", "김 민준"]) == "김 민준"
    assert emr_link.agreed_name(["김민준", "이서연"]) == "" and emr_link.agreed_name([]) == ""


def test_places_from_two_patients_keep_the_structure_but_no_names():
    places = name_places()
    assert [(p["kind"], p["template"]) for p in places] == [
        ("element", "[환자 이름]"),  # 이름만 있는 칸이 먼저
        ("title", "씨차트 - [환자 이름](남/##세) 차트번호 ########"),
    ]
    edit = places[0]
    assert edit["method"] == "UIA" and edit["source"] == "value" and edit["key"][-1] == ["Edit", "TEdit", "edtName", 1]
    assert emr_link.describe_place(edit) == "SeeChart.exe 입력칸 (TEdit · #edtName)"
    assert emr_link.describe_place(places[1]) == "SeeChart.exe 창 제목"
    assert emr_link.name_places([fake_scan("김민준", "김민준")]) == []  # 한 환자만으로는 정하지 않는다
    assert_nothing_private(json.dumps(places, ensure_ascii=False))


def test_chart_field_is_an_input_field_with_the_marker():
    run = marker_scan()
    assert {hit["control"] for hit in run["hits"]} == {"Document", "Text"}
    fields = emr_link.field_places(run)
    assert len(fields) == 1 and fields[0]["key"][-1] == ["Document", "TMemo", "memChart", 3]
    assert emr_link.describe_place(fields[0]) == "SeeChart.exe 입력칸 (TMemo · #memChart)"
    report = emr_link.field_report(run)
    assert "[표시 글자]" in report and "→ 요약을 넣을 칸: SeeChart.exe 입력칸 (TMemo · #memChart)" in report
    assert "지난" not in report and "김민준" not in report
    assert emr_link.field_places(emr_probe.new_result()) == []


def test_settings_keep_places_and_ignore_broken_ones(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings()
    settings.emr_name_places, settings.emr_note_place = name_places(), emr_link.field_places(marker_scan())[0]
    settings.save(path)
    loaded = Settings.load(path)
    assert loaded.emr_name_places == settings.emr_name_places and loaded.emr_note_place == settings.emr_note_place
    assert emr_link.usable_names(loaded.emr_name_places) == loaded.emr_name_places
    assert emr_link.usable_field(loaded.emr_note_place) == loaded.emr_note_place
    broken = dict(loaded.emr_name_places[0], template="○○○")  # 이름 자리가 없는 글자 모양
    assert emr_link.usable_names([broken, {"exe": "x"}, "?"]) == [] and emr_link.usable_names(None) == []
    assert emr_link.usable_field({}) is None and emr_link.usable_field(dict(loaded.emr_note_place, key="?")) is None
    path.write_text(json.dumps({"emr_name_places": {"bad": 1}, "emr_note_place": []}), encoding="utf-8")
    assert Settings.load(path).emr_name_places == [] and Settings.load(path).emr_note_place == {}


def test_requests_go_on_stdin_and_errors_hide_names(monkeypatch):
    calls = []
    answers = [{"status": "ok", "patient": "이서연"}, {"status": "mismatch"}, {"status": "weird"}]

    def fake_run(cmd, stdin_data=None, cwd=None, timeout=None, cancel=None, **_):
        calls.append((cmd, json.loads(stdin_data)))
        if answers:
            return ProcessResult(0, json.dumps(answers.pop(0)).encode("ascii"), b"")
        return ProcessResult(1, b"", "Traceback (most recent call last):\nValueError: 김민준 사흘 전부터 두통".encode("utf-8"))

    monkeypatch.setattr(emr_link.procutil, "run", fake_run)
    places, field = name_places(), emr_link.field_places(marker_scan())[0]
    assert emr_link.run_read(places) == "이서연"
    assert emr_link.run_insert(places, field, SUMMARY, "김민준") == {"status": "mismatch", "error": ""}
    assert emr_link.run_check(field)["status"] == "error"  # 모르는 답
    with pytest.raises(emr_link.EmrLinkError) as failed:
        emr_link.run_read(places)
    assert "김민준" not in str(failed.value) and "두통" not in str(failed.value) and "ValueError" in str(failed.value)
    cmd, request = calls[1]
    assert cmd[-2:] == ["-m", "clinic_scribe.emr_link"]
    assert request["command"] == "insert" and request["expected"] == "김민준" and request["text"] == SUMMARY
    assert not any("김민준" in part or "두통" in part for part in cmd)


def test_command_line_answers_json():
    request = json.dumps({"command": "read", "names": []}).encode("ascii")
    done = subprocess.run([sys.executable, "-m", "clinic_scribe.emr_link"], input=request, capture_output=True,
                          cwd=REPO, timeout=60)
    result = json.loads(done.stdout.decode("ascii"))
    if sys.platform == "win32":
        assert result == {"status": "ok", "patient": ""}
    else:
        assert result["status"] == "error" and "Windows" in result["error"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 화면 읽기(UI Automation)와 입력칸 쓰기")
def test_reads_the_patient_and_appends_the_summary_in_another_program():
    """다른 프로세스의 창(가짜 진료 차트): 두 환자로 점검해 이름 자리를 정하고, 진료 기록 칸을 찾고, 같은 환자일 때만 끝에 덧붙인다."""
    pytest.importorskip("comtypes")
    helper = subprocess.Popen([sys.executable, str(HELPER), "김민준", "이서연", "김민준", "박지호"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def send(line: str) -> bytes:
        helper.stdin.write(f"{line}\n".encode("utf-8"))
        helper.stdin.flush()
        return helper.stdout.readline().strip()

    def note() -> str:
        return json.loads(send("dump"))["note"]

    try:
        assert helper.stdout.readline().strip() == b"ready"
        first = emr_probe.run_probe("김민준")
        assert send("show 이서연") == b"shown"
        second = emr_probe.run_probe("이서연")
        names = emr_link.name_places([first, second])
        assert names and names[0]["kind"] == "element" and names[0]["template"] == "[환자 이름]", (first["hits"], second["hits"])
        assert_nothing_private(json.dumps(names, ensure_ascii=False))
        assert emr_link.run_read(names) == "이서연"

        assert send("note 지난 진료 기록\\n" + emr_link.MARKER) == b"noted"
        fields = emr_link.field_places(emr_probe.run_probe(emr_link.MARKER))
        assert len(fields) == 1, fields
        field = fields[0]
        assert emr_link.run_check(field) == {"status": "ok", "error": ""}
        name_field = dict(names[0])  # 한 줄짜리 환자명 칸은 진료 기록 칸으로 쓰지 않는다
        assert emr_link.run_check(name_field)["status"] == "refused"
        assert send("note 지난 진료 기록") == b"noted"  # 표시 글자를 지운다

        assert emr_link.run_insert(names, field, SUMMARY, "김민준")["status"] == "mismatch"  # 다른 환자가 열려 있다
        assert note() == "지난 진료 기록"
        assert emr_link.run_insert(names, field, SUMMARY, "이 서연") == {"status": "ok", "error": ""}
        assert note() == "지난 진료 기록\r\n\r\n" + SUMMARY.replace("\n", "\r\n")
        assert send("show 박지호") == b"shown"
        assert emr_link.run_insert(names, field, SUMMARY, "이서연")["status"] == "mismatch"
        assert note().count("가상 요약") == 1
    finally:
        helper.kill()
        helper.wait(timeout=10)


# ---------------------------------------------------------------------------- 점검 창 · 설정 · 녹음 버튼
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
def test_dialog_saves_the_name_places_and_the_chart_field(qapp):
    from clinic_scribe.ui import theme
    from clinic_scribe.ui.emr_probe_dialog import EmrProbeDialog

    settings = Settings()
    checks = []
    scans = {"김민준": fake_scan("김민준", "김민준"), "이서연": fake_scan("이서연", "이서연")}
    marker_runs = [marker_scan(), marker_scan(in_label=False)]

    def probe(name, cancel):
        if name == emr_link.MARKER:
            run = marker_runs.pop(0)
            if len(marker_runs) == 1:
                run["hits"] = [hit for hit in run["hits"] if hit["control"] == "Text"]  # 처음엔 글자만 보이는 곳
            return run
        return scans[name]

    def check(place, cancel):
        checks.append(place)
        return {"status": "ok", "error": ""}

    dialog = EmrProbeDialog(probe=probe, settings=settings, check=check)
    dialog.setStyleSheet(theme.stylesheet("EmrProbeDialog", EmrProbeDialog.STYLE))
    dialog.show()
    assert [badge.text() for badge in dialog.badges] == ["1", "2", "3", "4"] and dialog.field_btn.text() == "칸 찾기"
    for name in ("김민준", "이서연"):
        dialog.name_edit.setText(name)
        dialog.find_btn.click()
        assert wait_until(qapp, lambda: dialog.find_btn.text() != "멈추기" and dialog.runs and dialog._task is None)
    assert dialog.pill.text() == "읽을 수 있음" and "이 자리를 저장했습니다" in dialog.verdict.text()
    assert [p["kind"] for p in settings.emr_name_places] == ["element", "title"]
    assert Settings.load().emr_name_places == settings.emr_name_places  # 바로 저장된다

    dialog.field_btn.click()
    assert wait_until(qapp, lambda: dialog._task is None and dialog.field_run is not None)
    assert "글자만 보이는 곳" in dialog.field_status.text() and not settings.emr_note_place and not checks
    assert dialog.badges[3].text() == "4" and dialog.field_btn.text() == "다시 찾기"
    dialog.field_btn.click()
    assert wait_until(qapp, lambda: dialog._task is None and "정했습니다" in dialog.field_status.text())
    assert settings.emr_note_place["key"][-1] == ["Document", "TMemo", "memChart", 3] and checks == [settings.emr_note_place]
    assert f"{emr_link.MARKER} 는 지워" in dialog.field_status.text() and dialog.badges[3].text() == "✓"
    assert Settings.load().emr_note_place == settings.emr_note_place
    report = dialog._report()
    assert "진료 기록 칸 찾기" in report and "[표시 글자]" in report
    assert_nothing_private(report + dialog.result.toPlainText() + dialog.verdict.text() + json.dumps(settings.emr_name_places))
    dialog.reset_btn.click()
    assert "정해 둔 칸" in dialog.field_status.text() and dialog.badges[3].text() == "✓"  # 저장한 자리는 남는다
    dialog.close()


@pytest.mark.gui
def test_settings_tab_shows_the_saved_places_and_can_clear_them(qapp, monkeypatch):
    from clinic_scribe.ui.settings_dialog import SettingsDialog

    monkeypatch.setattr(emr_probe, "supported", lambda: True)
    settings = Settings()
    dialog = SettingsDialog(settings)
    assert dialog.emr_name_status.text().startswith("아직") and dialog.emr_insert.isChecked()
    assert not dialog.emr_clear_btn.isEnabled()
    settings.emr_name_places, settings.emr_note_place = name_places(), emr_link.field_places(marker_scan())[0]
    dialog = SettingsDialog(settings)
    assert dialog.emr_name_status.text() == "✓ SeeChart.exe 입력칸 (TEdit · #edtName) 외 1곳"
    assert dialog.emr_field_status.text() == "✓ SeeChart.exe 입력칸 (TMemo · #memChart)"
    dialog.emr_insert.setChecked(False)
    dialog.apply()
    assert settings.emr_insert_summary is False
    dialog.emr_clear_btn.click()
    assert settings.emr_name_places == [] and settings.emr_note_place == {} and Settings.load().emr_name_places == []
    assert dialog.emr_name_status.text().startswith("아직") and not dialog.emr_clear_btn.isEnabled()


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    import numpy as np

    from clinic_scribe import mic_permission, recorder
    from clinic_scribe.ui import main_window
    from tests.helpers import make_fake_cli
    from tests.test_gui import FAKE_DEVICE, FakeInputStream

    monkeypatch.setattr(main_window, "list_input_devices", lambda refresh=False: [FAKE_DEVICE])
    monkeypatch.setattr(mic_permission, "authorization_status", lambda: "authorized")
    monkeypatch.setattr(mic_permission, "lid_closed", lambda: False)
    monkeypatch.setattr(recorder.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(recorder.sd, "check_input_settings", lambda **kw: None)
    FakeInputStream.source = np.zeros(48000 * 5, np.float32) + 1e-3
    settings = Settings()
    settings.auto_summarize = False
    settings.claude_cli = make_fake_cli(tmp_path, "success")
    w = main_window.MainWindow(settings)
    w.interactive = False
    w.show()
    yield w
    w.shutdown()
    w.deleteLater()


@pytest.mark.gui
def test_record_button_reads_the_patient_and_puts_the_summary_into_seechart(window, qapp, monkeypatch):
    """씨차트 연결을 정했으면: 녹음을 시작할 때 환자 이름을 읽어 붙이고, 요약은 같은 환자일 때만 진료 기록 칸에 넣는다."""
    from clinic_scribe.chart import SECTIONS, ChartDraft
    from clinic_scribe.ui import main_window

    cues, reads, inserts = [], [], []
    answers = {"read": ["김민준", "이서연"], "insert": [{"status": "ok", "error": ""}, {"status": "mismatch", "error": ""}]}
    monkeypatch.setattr(main_window, "play_cue", cues.append)
    monkeypatch.setattr(emr_link, "supported", lambda: True)
    monkeypatch.setattr(emr_link, "run_read", lambda places, cancel=None: (reads.append(places), answers["read"].pop(0))[1])

    def run_insert(names, field, text, expected, cancel=None):
        inserts.append((names, field, text, expected))
        return answers["insert"].pop(0)

    monkeypatch.setattr(emr_link, "run_insert", run_insert)
    settings = window.settings
    settings.emr_name_places, settings.emr_note_place = name_places(), emr_link.field_places(marker_scan())[0]

    window._on_hotkey()  # 녹음 버튼: 시작하면서 씨차트 환자 이름을 읽는다
    rec = window._recording_session
    assert window.recorder.active
    assert wait_until(qapp, lambda: rec.patient == "김민준")
    assert reads == [settings.emr_name_places] and rec.emr_reads == ["김민준"]
    assert window.overlay.detail.startswith("김민준 · ")
    rec.emr_rechecked = True
    window._read_emr_patient(rec)  # 20초 뒤 다시 읽었더니 그 사이 다른 환자를 열었다
    assert wait_until(qapp, lambda: rec.patient == "이서연")
    assert emr_link.agreed_name(rec.emr_reads) == ""
    window._on_cancel_recording()

    def finished(patient, reads_seen):
        session = window._create_session("가상 녹음")
        session.via_hotkey, session.patient, session.emr_reads = True, patient, reads_seen
        session.draft = ChartDraft(sections={key: [] for key, _title in SECTIONS})
        return session

    window._deliver_hotkey_summary(finished("이서연", ["김민준", "이서연"]))  # 녹음하는 동안 환자가 바뀜: 넣지 않는다
    assert not inserts and window.overlay.kind == "warn" and "환자가 바뀌었습니다" in window.overlay.detail
    assert qapp.clipboard().text().startswith("[검토용 초안]") and cues[-1] == "error"

    session = finished("김민준", ["김민준", "김 민준"])
    window._deliver_hotkey_summary(session)
    assert window.overlay.text == "씨차트에 넣는 중…"
    assert wait_until(qapp, lambda: window.overlay.kind == "done")
    names, field, text, expected = inserts[0]
    assert expected == "김 민준" and text.startswith("[검토용 초안]") and field == settings.emr_note_place
    assert window.overlay.text == "씨차트에 넣었습니다" and "저장하세요" in window.overlay.detail and cues[-1] == "done"
    window._deliver_hotkey_summary(session)  # 같은 진료를 다시 요약했다: 또 넣지 않고 복사만 한다
    assert len(inserts) == 1 and window.overlay.text == "요약을 복사했습니다"

    window._deliver_hotkey_summary(finished("박지호", ["박지호"]))  # 그 사이 씨차트에서 다른 환자를 열었다
    assert wait_until(qapp, lambda: window.overlay.kind == "warn")
    assert "열린 환자가 다릅니다" in window.overlay.detail and "복사해 두었습니다" in window.overlay.detail

    settings.emr_insert_summary = False  # 넣기를 끄면 예전처럼 복사만 한다
    window._deliver_hotkey_summary(finished("최유나", ["최유나"]))
    assert len(inserts) == 2 and window.overlay.text == "요약을 복사했습니다"
