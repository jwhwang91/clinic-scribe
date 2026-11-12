"""씨차트 연결 점검 창: 씨차트에서 지금 보고 있는 환자 이름을 이 앱이 읽을 수 있는지 확인한다 (emr_probe).

환자 이름은 이 창의 입력칸과 메모리에만 있다. 화면의 결과와 결과 파일에는 '[환자 이름]'으로 가려 보여 준다.
"""

from __future__ import annotations

import html
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from clinic_scribe import emr_probe
from clinic_scribe.procutil import Cancelled
from clinic_scribe.ui import theme
from clinic_scribe.ui.tasks import Job, TaskRunner

# 판단 → (상태 표시, 글자), 안내 상자
STATUS = {"done": ("done", "읽을 수 있음"), "found": ("busy", "찾음"), "missing": ("warn", "못 찾음"),
          "error": ("error", "점검 못 함")}
CALLOUT = {"done": "ok", "found": "note", "missing": "warn", "error": "error"}
SHOWN_HITS = 8


class EmrProbeDialog(QDialog):
    STYLE = """
#dialogTitle { font-size: 18px; font-weight: 700; color: #f1f3f7; }
#stepBadge { background: #232838; color: #a5b4fc; border-radius: 11px; font-size: 12px; font-weight: 700; }
#stepBadge[done="true"] { background: #0f2b20; color: #4ade80; }
#stepText { font-size: 13px; color: #dfe3ea; }
"""

    def __init__(self, parent=None, probe: Callable[[str, threading.Event], dict] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("씨차트 연결 점검")
        self.setMinimumSize(700, 640)
        self.runs: list[dict] = []  # 점검 결과 (가린 글자만)
        self._probe = probe or emr_probe.run_probe
        self._seen: list[str] = []  # 앞에서 찾은 환자 이름: 같은 환자로 다시 찾지 않게 (메모리에만)
        self._pending = ""
        self._task = None
        self._started = 0.0
        self._runner = TaskRunner(self)
        self._runner.finished.connect(self._on_finished)
        self._ticker = QTimer(self)
        self._ticker.setInterval(500)
        self._ticker.timeout.connect(self._tick)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(22, 20, 22, 16)
        outer.setSpacing(12)
        title = QLabel("씨차트 연결 점검")
        title.setObjectName("dialogTitle")
        outer.addWidget(title)
        intro = QLabel("씨차트에서 지금 보고 있는 환자 이름을 이 앱이 읽을 수 있는지 확인합니다. "
                       "씨차트에는 아무것도 입력하거나 누르지 않고, 화면의 글자를 읽기만 합니다.")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        steps = QFrame()
        steps.setObjectName("card")
        steps_layout = QVBoxLayout(steps)
        steps_layout.setContentsMargins(16, 14, 16, 14)
        steps_layout.setSpacing(10)
        self.badges: list[QLabel] = []
        steps_layout.addLayout(self._step("씨차트에서 환자 한 명을 열어 두세요."))
        steps_layout.addLayout(self._step("그 환자 이름을 화면에 보이는 그대로 적고 찾기를 누르세요."))
        name_row = QHBoxLayout()
        name_row.setContentsMargins(32, 0, 0, 0)
        name_row.setSpacing(8)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("환자 이름 (예: 홍길동)")
        self.name_edit.returnPressed.connect(self._start)
        name_row.addWidget(self.name_edit, 1)
        self.find_btn = QPushButton("찾기")
        self.find_btn.setObjectName("primary")
        self.find_btn.setMinimumWidth(110)
        self.find_btn.clicked.connect(self._on_find_clicked)
        name_row.addWidget(self.find_btn)
        steps_layout.addLayout(name_row)
        privacy = QLabel("이름은 이 컴퓨터 안에서만 씁니다. 결과와 결과 파일에는 [환자 이름]으로, "
                         "다른 글자는 모양만(한글 ○ · 숫자 #) 남깁니다.")
        privacy.setObjectName("subtle")
        privacy.setWordWrap(True)
        privacy.setContentsMargins(32, 0, 0, 2)
        steps_layout.addWidget(privacy)
        steps_layout.addLayout(self._step("씨차트에서 다른 환자를 열고, 그 이름으로 한 번 더 찾으세요. "
                                          "환자가 바뀌어도 같은 자리에서 읽히는지 확인합니다."))
        outer.addWidget(steps)

        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        self.pill = QLabel()
        status_row.addWidget(self.pill, 0, Qt.AlignmentFlag.AlignVCenter)
        self.detail = QLabel()
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        status_row.addWidget(self.detail, 1)
        outer.addLayout(status_row)
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        self.verdict.hide()
        outer.addWidget(self.verdict)

        result_card = QFrame()
        result_card.setObjectName("card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(16, 8, 16, 8)
        self.result = QTextBrowser()
        self.result.document().setDefaultStyleSheet(theme.DOCUMENT_STYLE)
        self.result.setMinimumHeight(170)
        result_layout.addWidget(self.result)
        outer.addWidget(result_card, 1)

        bottom = QHBoxLayout()
        self.save_btn = QPushButton("결과 저장…")
        self.save_btn.setToolTip("가린 결과를 텍스트 파일로 저장합니다. 이 파일을 보내 주시면 연결 방법을 정합니다.")
        self.save_btn.clicked.connect(self._save)
        bottom.addWidget(self.save_btn)
        self.copy_btn = QPushButton("결과 복사")
        self.copy_btn.clicked.connect(self._copy)
        bottom.addWidget(self.copy_btn)
        self.reset_btn = QPushButton("처음부터")
        self.reset_btn.setObjectName("ghost")
        self.reset_btn.clicked.connect(self._reset)
        bottom.addWidget(self.reset_btn)
        bottom.addStretch(1)
        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.reject)
        bottom.addWidget(self.close_btn)
        outer.addLayout(bottom)
        for button in (self.find_btn, self.save_btn, self.copy_btn, self.reset_btn, self.close_btn):
            button.setAutoDefault(False)  # 이름 칸에서 Enter 는 찾기만 한다
        self._reset()

    def _step(self, text: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        badge = QLabel(str(len(self.badges) + 1))
        badge.setObjectName("stepBadge")
        badge.setFixedSize(22, 22)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badges.append(badge)
        row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        label = QLabel(text)
        label.setObjectName("stepText")
        label.setWordWrap(True)
        row.addWidget(label, 1)
        return row

    # ---- 화면 상태 ----------------------------------------------------------
    def _set_status(self, kind: str, label: str, detail: str) -> None:
        self.pill.setText(label)
        self.pill.setStyleSheet(theme.pill(kind, 12))
        self.detail.setText(detail)

    def _found(self) -> int:
        return sum(1 for run in self.runs if run["hits"])

    def _mark_steps(self) -> None:
        found = self._found()
        done = (found >= 1, found >= 1, found >= 2)
        for number, (badge, finished) in enumerate(zip(self.badges, done), 1):
            badge.setText("✓" if finished else str(number))
            badge.setProperty("done", finished)
            badge.style().unpolish(badge)
            badge.style().polish(badge)

    def _set_busy(self, busy: bool) -> None:
        self.name_edit.setEnabled(not busy)
        self.find_btn.setText("멈추기" if busy else ("한 번 더 찾기" if self._found() else "찾기"))
        for button in (self.save_btn, self.copy_btn):
            button.setEnabled(not busy and bool(self.runs))
        self.reset_btn.setEnabled(not busy and bool(self.runs))
        if busy:
            self._ticker.start()
            self._tick()
        else:
            self._ticker.stop()

    def _tick(self) -> None:
        seconds = time.monotonic() - self._started
        self._set_status("busy", "읽는 중", f"열린 창의 글자를 읽고 있습니다… {seconds:.0f}초 (길어도 1분 남짓)")

    def _reset(self) -> None:
        self.runs.clear()
        self._seen.clear()
        self.name_edit.clear()
        self.name_edit.setPlaceholderText("환자 이름 (예: 홍길동)")
        self.verdict.hide()
        self.result.setHtml('<p style="color:#6f7787">찾기를 누르면 이름을 찾은 곳이 여기에 나옵니다.</p>')
        self._set_status("idle", "점검 전", "씨차트와 이 창을 함께 띄워 두고 시작하세요.")
        self._mark_steps()
        self._set_busy(False)

    # ---- 점검 ---------------------------------------------------------------
    def _on_find_clicked(self) -> None:
        if self._task is not None:
            self._task.cancel.set()
        else:
            self._start()

    def _start(self) -> None:
        if self._task is not None:
            return
        name = self.name_edit.text().strip()
        key = "".join(name.split())
        if len(key) < 2:
            self._set_status("warn", "이름 확인", "환자 이름을 두 글자 이상 적어 주세요.")
            return
        if key in self._seen:
            self._set_status("warn", "같은 환자", "앞에서 찾은 환자입니다. 씨차트에서 다른 환자를 열고 그 이름으로 찾아 주세요.")
            return
        self._pending = key
        self._started = time.monotonic()
        self._task = self._runner.start(Job("emr_probe", 0, len(self.runs)), lambda cancel: self._probe(name, cancel))
        self._set_busy(True)

    def _on_finished(self, task, result, error) -> None:
        if task is not self._task:
            return
        self._task = None
        self._set_busy(False)
        if isinstance(error, Cancelled):
            self._set_status("idle", "멈춤", "점검을 멈췄습니다.")
            return
        if error is not None or result.get("error"):
            self._set_status("error", "점검 못 함", str(error) if error is not None else result["error"])
            return
        self.runs.append(result)
        if result["hits"]:  # 못 찾았으면 이름을 고치거나 같은 이름으로 다시 해 볼 수 있게 둔다
            self._seen.append(self._pending)
            self.name_edit.clear()
            self.name_edit.setPlaceholderText("다른 환자 이름")
        self._render()

    def _render(self) -> None:
        kind, message = emr_probe.verdict(self.runs)
        pill, label = STATUS[kind]
        last = self.runs[-1]
        scanned = sum(1 for w in last["windows"] if w["scanned"])
        self._set_status(pill, label, f"{len(self.runs)}번 점검 · 마지막 {last['elapsed']:.0f}초 동안 창 {scanned}개를 읽었습니다.")
        self.verdict.setText(message)
        self.verdict.setStyleSheet(theme.callout(CALLOUT[kind]))
        self.verdict.show()
        self.result.setHtml(self._html())
        self._mark_steps()
        self._set_busy(False)

    def _html(self) -> str:
        esc = html.escape
        parts = []
        stable = emr_probe.stable_locations(self.runs)
        if stable:
            items = "".join(f"<li>{esc(emr_probe.describe_hit(hit, detail=False))}</li>" for hit in stable)
            parts.append(f'<h3 style="color:#4ade80">✓ 환자가 바뀌어도 같은 자리 {len(stable)}곳</h3><ul>{items}</ul>')
        for number in range(len(self.runs), 0, -1):  # 최근 점검부터
            run = self.runs[number - 1]
            hits = run["hits"]
            programs = ", ".join(sorted({hit["exe"] for hit in hits}))
            head = f"점검 {number} · " + (f"이름을 찾은 곳 {len(hits)}곳 ({programs})" if hits else "이름을 찾지 못함")
            parts.append(f"<h3>{esc(head)}</h3>")
            if hits:
                items = "".join(f"<li>{esc(emr_probe.describe_hit(hit, detail=False))}</li>" for hit in hits[:SHOWN_HITS])
                if len(hits) > SHOWN_HITS:
                    items += f"<li>… {len(hits) - SHOWN_HITS}곳 더 (결과 파일에 모두 있습니다)</li>"
                parts.append(f"<ul>{items}</ul>")
            else:
                programs = sorted({w["exe"] or "(알 수 없음)" for w in run["windows"] if w["scanned"]})
                parts.append(f"<p>읽은 프로그램: {esc(', '.join(programs) or '없음')}</p>")
        return "".join(parts)

    # ---- 결과 파일 ------------------------------------------------------------
    def _save(self) -> None:
        desktop = Path.home() / "Desktop"
        folder = desktop if desktop.is_dir() else Path.home()
        default = folder / f"씨차트_연결_점검_{datetime.now():%Y%m%d_%H%M}.txt"
        path, _filter = QFileDialog.getSaveFileName(self, "점검 결과 저장", str(default), "텍스트 파일 (*.txt)")
        if not path:
            return
        try:
            Path(path).write_text(emr_probe.format_report(self.runs), encoding="utf-8-sig")  # 메모장에서도 한글이 보이게
        except OSError as exc:
            self.detail.setText(f"저장하지 못했습니다: {exc}")
            return
        self.detail.setText(f"결과를 저장했습니다: {path}")

    def _copy(self) -> None:
        QApplication.clipboard().setText(emr_probe.format_report(self.runs))
        self.detail.setText("결과를 복사했습니다. 메일이나 메신저에 붙여 넣어 보내 주세요.")

    def done(self, result: int) -> None:
        if self._task is not None:
            self._task.cancel.set()  # 창을 닫으면 점검 프로세스도 끝낸다
        super().done(result)
