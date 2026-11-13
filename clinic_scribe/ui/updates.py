"""자동 업데이트 화면: 30분마다 새 버전을 확인하는 감시자와 '업데이트' 창 (실제 일은 updater.py).

새 버전은 테스트를 통과한 것만 받아 두고, 앱을 다시 열 때 적용한다. '다시 열어 적용'은 녹음·처리 중이 아닐 때만 누를 수 있다.
"""

from __future__ import annotations

import html
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout

from clinic_scribe import updater
from clinic_scribe.config import Settings
from clinic_scribe.ui import theme


class UpdateWatcher(QObject):
    """앱이 열려 있는 동안 새 버전을 확인한다 (네트워크는 따로 도는 스레드에서)."""

    changed = Signal()
    _done = Signal(object)  # 다른 스레드 → 화면 스레드

    def __init__(self, settings: Settings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.result: updater.CheckResult | None = None
        self.checking = False
        self.notice = updater.take_notice()  # 방금 적용했거나 되돌린 업데이트
        self.current = updater.version_of("HEAD")
        self.blocked = updater.blocker()
        self._done.connect(self._on_done)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timer)
        if not self.blocked:
            self._timer.start(updater.FIRST_CHECK_S * 1000)

    @property
    def ready(self) -> updater.Version | None:
        return self.result.ready if self.result is not None else updater.ready_version()

    def _on_timer(self) -> None:
        if self.settings.auto_update:
            self.check_now()
        self._timer.start(updater.CHECK_INTERVAL_S * 1000)

    def check_now(self) -> None:
        if self.checking:
            return
        self.checking = True
        self.changed.emit()

        def run() -> None:
            try:
                result = updater.check()
            except Exception as exc:  # 확인 실패는 화면에 한 줄로만
                result = updater.CheckResult(None, 0, f"확인하지 못했습니다: {type(exc).__name__}")
            try:
                self._done.emit(result)
            except RuntimeError:
                pass  # 앱 종료 중

        threading.Thread(target=run, name="clinic-scribe-update", daemon=True).start()

    def _on_done(self, result: updater.CheckResult) -> None:
        self.checking = False
        self.result = result
        self.current = updater.version_of("HEAD")
        self.changed.emit()

    def acknowledge(self) -> None:
        """'업데이트됨' 알림을 봤다."""
        if self.notice is not None:
            self.notice = None
            self.changed.emit()

    def chip(self) -> tuple[str, str] | None:
        """위쪽 막대 단추 (글자, 상태 색) 또는 숨김."""
        if self.notice is not None and self.notice["kind"] == "rolled_back":
            return "업데이트 되돌림", "warn"
        if self.ready is not None:
            return "새 버전", "busy"
        if self.notice is not None:
            return "업데이트됨", "done"
        return None


class UpdateDialog(QDialog):
    restart_requested = Signal()  # '다시 열어 적용': 메인 창이 녹음·처리 중이 아닌지 보고 다시 연다

    def __init__(self, watcher: UpdateWatcher, can_restart: Callable[[], bool], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("업데이트")
        self.setMinimumSize(560, 420)
        self.watcher = watcher
        self._can_restart = can_restart
        outer = QVBoxLayout(self)
        outer.setContentsMargins(22, 20, 22, 16)
        outer.setSpacing(12)
        title = QLabel("업데이트")
        title.setObjectName("cardTitle")
        outer.addWidget(title)
        intro = QLabel("GitHub 에 올라온 새 버전 가운데 테스트를 모두 통과한 버전만 받아 두었다가, 앱을 다시 열 때 적용합니다. "
                       "진료 기록·설정·의사 목소리는 그대로 남습니다.")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        self.current = QLabel()
        self.current.setWordWrap(True)
        outer.addWidget(self.current)
        status_row = QHBoxLayout()
        self.pill = QLabel()
        status_row.addWidget(self.pill)
        self.status = QLabel()
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        status_row.addWidget(self.status, 1)
        outer.addLayout(status_row)
        self.changes = QTextBrowser()
        self.changes.document().setDefaultStyleSheet(theme.DOCUMENT_STYLE)
        outer.addWidget(self.changes, 1)
        buttons = QHBoxLayout()
        self.check_btn = QPushButton("지금 확인")
        self.check_btn.clicked.connect(watcher.check_now)
        buttons.addWidget(self.check_btn)
        buttons.addStretch(1)
        self.restart_btn = QPushButton("다시 열어 적용")
        self.restart_btn.setObjectName("primary")
        self.restart_btn.clicked.connect(self._on_restart)
        buttons.addWidget(self.restart_btn)
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        outer.addLayout(buttons)
        for button in (self.check_btn, self.restart_btn, close_btn):
            button.setAutoDefault(False)
        watcher.changed.connect(self._refresh)
        self._refresh()

    def _set_pill(self, kind: str, text: str) -> None:
        self.pill.setText(text)
        self.pill.setStyleSheet(theme.pill(kind, 12))

    def _refresh(self) -> None:
        w = self.watcher
        current = w.current
        self.current.setText(f"지금 버전: {current.describe()} — {current.subject}" if current else "지금 버전: 알 수 없음")
        ready = w.ready
        notice = w.notice
        rows = []
        if w.blocked:
            self._set_pill("idle", "자동 업데이트 꺼짐")
            self.status.setText(w.blocked)
        elif w.checking:
            self._set_pill("busy", "확인하는 중")
            self.status.setText("GitHub 에서 새 버전과 테스트 결과를 확인하고 있습니다…")
        elif ready is not None:
            self._set_pill("busy", "새 버전 준비됨")
            waiting = w.result.waiting if w.result is not None else 0
            self.status.setText(f"{ready.describe()} (테스트 통과). 다시 열면 적용됩니다."
                                + (f" 더 새 버전 {waiting}개는 테스트가 끝나면 받습니다." if waiting else ""))
            rows = updater.changes(ready.commit)
        elif w.result is not None and w.result.error:
            self._set_pill("warn", "확인 못 함")
            self.status.setText(w.result.error)
        elif w.result is not None:
            self._set_pill("done", "최신 버전")
            waiting = w.result.waiting
            self.status.setText("최신 버전입니다." + (f" 더 새 버전 {waiting}개는 테스트 중이거나 통과하지 못해 기다립니다."
                                                    if waiting else ""))
        else:
            self._set_pill("idle", "확인 전")
            self.status.setText("'지금 확인'을 누르거나, 앱이 열려 있으면 30분마다 확인합니다.")
        if notice is not None and ready is None and not w.checking:
            if notice["kind"] == "rolled_back":
                self._set_pill("warn", "되돌림")
                self.status.setText("새 버전이 창을 띄우지 못해 이전 버전으로 되돌렸습니다. 그 버전은 다시 받지 않습니다.")
            else:
                self._set_pill("done", "업데이트됨")
                self.status.setText("새 버전으로 업데이트했습니다.")
        if rows:
            items = "".join(f"<li>{html.escape(row)}</li>" for row in rows[:30])
            self.changes.setHtml(f"<h3>바뀐 것</h3><ul>{items}</ul>")
        else:
            self.changes.setHtml('<p style="color:#6f7787">새 버전이 준비되면 바뀐 것이 여기에 나옵니다.</p>')
        self.check_btn.setEnabled(not w.checking and not w.blocked)
        self.restart_btn.setEnabled(ready is not None and self._can_restart())
        self.restart_btn.setToolTip("" if self._can_restart() else "녹음·전사·요약이 끝난 뒤에 누를 수 있습니다.")

    def _on_restart(self) -> None:
        if self._can_restart():
            self.accept()
            self.restart_requested.emit()
