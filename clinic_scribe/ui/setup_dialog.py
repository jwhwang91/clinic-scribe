"""'설치' 창: 새 컴퓨터에서 처음 열 때(또는 설정 → '설치 · 요약 방식…') 필요한 프로그램·모델을 받고 요약 방식을 고른다.

1) 전사에 필요한 것: whisper.cpp(전사·음성 구간 탐지 프로그램), Whisper 모델, 음성 구간 모델, 화자 인식 모델
2) 요약 방식: Claude Code (Anthropic · 요약할 때 전사문 전송, 로그인 필요) / 로컬 모델 (이 컴퓨터에서만, 처음 한 번 받기)
   로컬 모델은 이 컴퓨터의 메모리·CPU·GPU 로 돌아가는지(원활 / 느림 / 메모리 부족) 보여 주고 알맞은 것을 추천한다.
'설치하고 시작' 을 눌러야 받기 시작한다(사용자가 확인). 받은 파일은 SHA-256 을 확인하고, 끊기면 다음에 이어받는다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from clinic_scribe import config, installer
from clinic_scribe.config import Settings
from clinic_scribe.ui import theme

_FIT_MARK = {"good": "✓", "slow": "△", "no": "✗"}
_FIT_COLOR = {"good": "#4ade80", "slow": "#fbbf24", "no": "#f87171"}


@dataclass
class Part:
    """전사에 필요한 것 하나."""

    key: str
    title: str
    detail: str
    ready: bool
    required: bool = True


def transcription_parts(settings: Settings) -> list[Part]:
    whisper_cli = settings.whisper_cli_path()
    if installer.os_key() == "darwin":
        how = "Homebrew 로 설치 (brew install whisper-cpp)"
    else:
        build = installer.WHISPER_BUILDS.get((installer.os_key(), installer.machine()))
        how = f"공식 빌드 {installer.human_size(build.size)}" if build else "이 OS·CPU 용 공식 빌드 없음 (직접 빌드)"
    whisper_model, vad_model, speaker_model = installer.MODELS
    return [
        Part("whisper", "전사 프로그램 (whisper.cpp)", f"전사·음성 구간 탐지 · {how}", installer.whisper_ready(whisper_cli)),
        Part("whisper_model", "Whisper 전사 모델 (large-v3-turbo)", installer.human_size(whisper_model.size),
             _model_ready(whisper_model, settings.whisper_model, settings.whisper_model_path())),
        Part("vad_model", "음성 구간 모델 (Silero VAD)", installer.human_size(vad_model.size),
             _model_ready(vad_model, settings.vad_model, settings.vad_model_path())),
        Part("speaker_model", "화자 인식 모델 (말한 사람 표시)", installer.human_size(speaker_model.size),
             _model_ready(speaker_model, settings.speaker_model, settings.speaker_model_path()), required=False),
    ]


def _model_ready(item: installer.Download, configured: str, path: Path) -> bool:
    """설정에서 경로를 직접 지정했으면 파일만 있으면 되고, 기본 위치면 크기까지 맞아야 한다."""
    return path.is_file() if configured else installer.is_valid(item, path)


def engine_ready(settings: Settings) -> bool:
    if settings.summary_engine == "local":
        model = installer.local_model(settings.local_model)
        return (model is not None and installer.llama_server_path().is_file()
                and installer.is_valid(model.file, installer.local_model_path(model)))
    return settings.summary_engine == "claude" and bool(settings.claude_cli_path())


class _Signals(QObject):
    progress = Signal(object, object)  # 받은 바이트, 전체 (Qt int 는 32비트라 2GB 넘는 모델 크기를 담지 못한다)
    step = Signal(str)
    done = Signal(object)  # None(성공) 또는 오류 문구
    login = Signal(object)  # Claude 로그인 상태 (True/False/None)


class SetupDialog(QDialog):
    @staticmethod
    def needed(settings: Settings) -> bool:
        """빠진 것이 있거나 요약 방식을 아직 고르지 않았으면 띄운다."""
        parts_missing = any(p.required and not p.ready for p in transcription_parts(settings))
        return parts_missing or not settings.summary_engine or not engine_ready(settings)

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("clinic-scribe 설치 · 요약 방식")
        self.setMinimumWidth(760)
        self.settings = settings
        self.specs = installer.system_specs()
        self._cancel: threading.Event | None = None  # 받는 중이면 취소 신호
        self._login: bool | None = None  # Claude Code 로그인 상태 (모르면 None)
        self._close_when_logged_in = False
        self._current_step = ""
        self._engine_choice: tuple[str, installer.LocalModel | None] = ("claude", None)
        self._signals = _Signals(self)
        self._signals.progress.connect(self._on_progress)
        self._signals.step.connect(self._on_step)
        self._signals.done.connect(self._on_done)
        self._signals.login.connect(self._on_login_checked)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(12)
        title = QLabel("필요한 프로그램·모델을 받고 요약 방식을 고릅니다")
        title.setObjectName("cardTitle")
        outer.addWidget(title)
        intro = QLabel("받는 파일은 모두 공식 배포본이며 SHA-256 으로 확인합니다. '설치하고 시작' 을 누르기 전에는 아무것도 받지 않습니다.")
        intro.setObjectName("subtle")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # 1) 전사에 필요한 것
        parts_card, parts_layout = self._card("전사에 필요한 것 (이 컴퓨터에서 처리)")
        self.parts = transcription_parts(settings)
        self.part_labels: dict[str, QLabel] = {}
        for part in self.parts:
            row = QHBoxLayout()
            name = QLabel(part.title)
            detail = QLabel(part.detail)
            detail.setObjectName("subtle")
            state = QLabel()
            self.part_labels[part.key] = state
            row.addWidget(name)
            row.addWidget(detail, 1)
            row.addWidget(state)
            parts_layout.addLayout(row)
        outer.addWidget(parts_card)

        # 2) 요약 방식
        engine_card, engine_layout = self._card("요약 방식")
        self.engine_group = QButtonGroup(self)
        self.claude_radio = QRadioButton("Claude Code — 요약 품질이 가장 좋음 · 요약할 때 전사문이 Anthropic 으로 전송됨 · 로그인 필요")
        self.local_radio = QRadioButton("로컬 모델 — 이 컴퓨터에서만 요약 (밖으로 보내지 않음) · 처음 한 번 모델을 받음")
        for button in (self.claude_radio, self.local_radio):
            self.engine_group.addButton(button)
            engine_layout.addWidget(button)
            button.toggled.connect(self._refresh)
        claude_row = QHBoxLayout()
        claude_row.setContentsMargins(26, 0, 0, 0)
        self.claude_state = QLabel()
        self.claude_state.setWordWrap(True)
        claude_row.addWidget(self.claude_state, 1)
        self.login_btn = QPushButton("로그인 창 열기")
        self.login_btn.setToolTip("터미널에서 'claude auth login' 을 실행합니다. 브라우저에서 로그인한 뒤 '다시 확인' 을 누르세요.")
        self.login_btn.clicked.connect(self._open_login)
        self.recheck_btn = QPushButton("다시 확인")
        self.recheck_btn.clicked.connect(self._check_login)
        claude_row.addWidget(self.login_btn)
        claude_row.addWidget(self.recheck_btn)
        engine_layout.addLayout(claude_row)

        self.specs_label = QLabel(f"이 컴퓨터: {self.specs.describe()}")
        self.specs_label.setObjectName("subtle")
        self.specs_label.setContentsMargins(26, 4, 0, 0)
        engine_layout.addWidget(self.specs_label)
        self.models = QTableWidget(len(installer.LOCAL_MODELS), 4)
        self.models.setHorizontalHeaderLabels(["모델", "크기", "이 컴퓨터에서", "설명"])
        self.models.verticalHeader().hide()
        self.models.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.models.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.models.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.models.horizontalHeader().setStretchLastSection(True)
        recommended = installer.recommended_model(self.specs)
        self.fits = {m.id: installer.model_fit(m, self.specs) for m in installer.LOCAL_MODELS}
        for row, model in enumerate(installer.LOCAL_MODELS):
            fit = self.fits[model.id]
            have = installer.is_valid(model.file, installer.local_model_path(model))
            label = model.label + ("  · 추천" if model is recommended else "") + ("  · 받음" if have else "")
            cells = (label, installer.human_size(model.file.size), f"{_FIT_MARK[fit.level]} {fit.text}", model.note)
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, model.id)
                if col == 2:
                    item.setForeground(QColor(_FIT_COLOR[fit.level]))
                if fit.level == "no":
                    item.setFlags(Qt.ItemFlag.NoItemFlags)  # 메모리가 모자라는 모델은 고를 수 없다
                self.models.setItem(row, col, item)
        self.models.setWordWrap(True)
        for col, width in enumerate((178, 68, 262)):
            self.models.setColumnWidth(col, width)
        self.models.resizeRowsToContents()
        self.models.setFixedHeight(self.models.verticalHeader().length() + self.models.horizontalHeader().height() + 6)
        self.models.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.models.itemSelectionChanged.connect(self._refresh)
        engine_layout.addWidget(self.models)
        outer.addWidget(engine_card)

        # 진행·버튼
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.hide()
        outer.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.hide()
        outer.addWidget(self.bar)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.later_btn = QPushButton("나중에")
        self.later_btn.clicked.connect(self.reject)
        self.cancel_btn = QPushButton("받기 취소")
        self.cancel_btn.clicked.connect(self._cancel_install)
        self.cancel_btn.hide()
        self.install_btn = QPushButton("설치하고 시작")
        self.install_btn.setObjectName("primary")
        self.install_btn.clicked.connect(self.install)
        for button in (self.later_btn, self.cancel_btn, self.install_btn):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        outer.addLayout(buttons)

        # 처음 고른 값: 저장된 방식 → 없으면 Claude 가 이미 있으면 Claude, 아니면 이 컴퓨터에 알맞은 로컬 모델
        chosen = installer.local_model(settings.local_model)
        if chosen is None or self.fits[chosen.id].level == "no" or not settings.summary_engine:
            chosen = recommended
        self._select_model(chosen.id)
        if settings.summary_engine == "local" or (not settings.summary_engine and not settings.claude_cli_path()):
            self.local_radio.setChecked(True)
        else:
            self.claude_radio.setChecked(True)
        self._refresh()
        self._check_login()

    # ---- 화면 ---------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 — 창이 실제 너비가 된 뒤 줄 높이를 다시 맞춘다 (설명이 두 줄일 때)
        super().showEvent(event)
        self.models.resizeRowsToContents()
        self.models.setFixedHeight(self.models.verticalHeader().length() + self.models.horizontalHeader().height() + 6)
        self.adjustSize()

    @staticmethod
    def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        return card, layout

    def _select_model(self, model_id: str) -> None:
        for row in range(self.models.rowCount()):
            if self.models.item(row, 0).data(Qt.ItemDataRole.UserRole) == model_id:
                self.models.selectRow(row)
                return

    def selected_model(self) -> installer.LocalModel | None:
        rows = self.models.selectionModel().selectedRows()
        if not rows:
            return None
        return installer.local_model(self.models.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole))

    def choose(self, engine: str, model_id: str | None = None) -> None:
        """요약 방식을 고른다 (자동 검증에서도 쓴다)."""
        (self.local_radio if engine == "local" else self.claude_radio).setChecked(True)
        if model_id:
            self._select_model(model_id)

    def _refresh(self) -> None:
        for part in self.parts:
            label = self.part_labels[part.key]
            label.setText("✓ 있음" if part.ready else ("받기 필요" if part.required else "받기 필요 (선택)"))
            label.setStyleSheet(f"color: {'#4ade80' if part.ready else '#fbbf24'};")
        local = self.local_radio.isChecked()
        # 고른 방식에 해당하는 것만 보여 준다 (Claude: 설치·로그인 상태 / 로컬: 이 컴퓨터 사양과 모델 표)
        self.models.setVisible(local)
        self.specs_label.setVisible(local)
        self.claude_state.setVisible(not local)
        cli = self.settings.claude_cli_path()
        if not cli:
            state, color = "설치 필요 — '설치하고 시작' 을 누르면 공식 설치 프로그램으로 설치합니다 (로그인은 직접 합니다)", "#fbbf24"
        elif self._login is True:
            state, color = f"✓ 설치됨 · 로그인됨 ({cli})", "#4ade80"
        elif self._login is False:
            state, color = "설치됨 · 로그인 필요 — '로그인 창 열기' 로 로그인한 뒤 '다시 확인' 을 누르세요", "#fbbf24"
        else:
            state, color = f"설치됨 ({cli}) · 로그인 상태 확인 중…", "#a3aab8"
        self.claude_state.setText(state)
        self.claude_state.setStyleSheet(f"color: {color};")
        self.login_btn.setVisible(bool(cli) and self._login is not True and not local)
        self.recheck_btn.setVisible(bool(cli) and self._login is not True and not local)
        nothing_to_get = not self._plan()
        self.install_btn.setText("시작" if nothing_to_get else "설치하고 시작")
        self.install_btn.setEnabled(self._cancel is None and (not local or self.selected_model() is not None))
        self.adjustSize()

    # ---- 할 일 -------------------------------------------------------------
    def _plan(self) -> list[str]:
        """받아야 할 것 (화면 순서)."""
        todo = [p.key for p in self.parts if not p.ready and not self._custom_path(p.key)]
        if self.local_radio.isChecked():
            model = self.selected_model()
            if not installer.llama_server_path().is_file():
                todo.append("llama")
            if model is not None and not installer.is_valid(model.file, installer.local_model_path(model)):
                todo.append("local_model")
        elif not self.settings.claude_cli_path():
            todo.append("claude")
        return todo

    def _custom_path(self, key: str) -> bool:
        """설정에서 경로를 직접 지정한 모델은 받지 않는다 (사용자가 고른 파일을 그대로 쓴다)."""
        s = self.settings
        return {"whisper_model": s.whisper_model, "vad_model": s.vad_model, "speaker_model": s.speaker_model}.get(key, "") != ""

    def install(self) -> None:
        """빠진 것을 받는다 ('설치하고 시작'). 받을 것이 없으면 고른 방식을 저장하고 닫는다."""
        todo = self._plan()
        model = self.selected_model()
        engine = "local" if self.local_radio.isChecked() else "claude"
        if not todo:
            self._save(engine, model)
            if engine == "claude" and self._login is False:
                self._say("요약 방식을 저장했습니다. Claude Code 에 로그인해야 요약할 수 있습니다.", "warn")
                return
            self.accept()
            return
        self._cancel = threading.Event()
        cancel = self._cancel
        for button in (self.install_btn, self.later_btn):
            button.setEnabled(False)
        self.cancel_btn.show()
        self.bar.show()
        self.bar.setValue(0)
        self.models.setEnabled(False)
        signals = self._signals

        def progress(done: int, total: int) -> None:
            signals.progress.emit(done, total)

        def run() -> None:
            try:
                for i, key in enumerate(todo, 1):
                    signals.step.emit(f"{i}/{len(todo)} · {self._step_title(key, model)}")
                    self._install_one(key, model, progress, cancel)
                signals.done.emit(None)
            except installer.InstallCancelled:
                signals.done.emit("받기를 취소했습니다. 받은 부분은 남겨 두었다가 다음에 이어받습니다.")
            except installer.InstallError as exc:
                signals.done.emit(str(exc))
            except Exception as exc:  # noqa: BLE001 — 창에 이유를 보여 준다
                signals.done.emit(f"설치 중 오류: {type(exc).__name__}: {exc}")

        self._engine_choice = (engine, model)
        threading.Thread(target=run, name="clinic-scribe-setup", daemon=True).start()

    @staticmethod
    def _step_title(key: str, model: installer.LocalModel | None) -> str:
        titles = {
            "whisper": "전사 프로그램(whisper.cpp) 설치",
            "whisper_model": "Whisper 전사 모델 받는 중",
            "vad_model": "음성 구간 모델 받는 중",
            "speaker_model": "화자 인식 모델 받는 중",
            "llama": "로컬 모델 실행 프로그램(llama.cpp) 받는 중",
            "local_model": f"로컬 모델 {model.label if model else ''} 받는 중",
            "claude": "Claude Code 설치 중 (공식 설치 프로그램)",
        }
        return titles[key]

    def _install_one(self, key: str, model: installer.LocalModel | None, progress, cancel: threading.Event) -> None:
        folder = config.default_model_dir()
        if key == "whisper":
            installer.install_whisper(progress, cancel)
            self.settings.whisper_cli = ""  # 자동으로 찾게 (방금 설치한 곳 포함)
        elif key in ("whisper_model", "vad_model", "speaker_model"):
            item = installer.MODELS[("whisper_model", "vad_model", "speaker_model").index(key)]
            installer.fetch(item, folder, progress, cancel)
        elif key == "llama":
            installer.install_llama(progress, cancel)
        elif key == "local_model" and model is not None:
            installer.fetch(model.file, folder, progress, cancel)
        elif key == "claude":
            installer.install_claude(cancel)
            self.settings.claude_cli = ""

    def _cancel_install(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, done: int, total: int) -> None:
        self.bar.setValue(int(1000 * done / total) if total else 0)
        self.status.setText(self._current_step + f" · {installer.human_size(done)} / {installer.human_size(total)}")

    def _on_step(self, text: str) -> None:
        self._current_step = text
        self._say(text, "note")
        self.bar.setValue(0)

    def _on_done(self, error: str | None) -> None:
        self._cancel = None
        self.cancel_btn.hide()
        self.cancel_btn.setEnabled(True)
        self.later_btn.setEnabled(True)
        self.parts = transcription_parts(self.settings)
        if error:
            self.bar.hide()
            self._say(error, "error")
            self._refresh()
            return
        engine, model = self._engine_choice
        self._save(engine, model)
        self.bar.setValue(1000)
        self._refresh()
        if engine == "claude":
            self._say("설치했습니다. Claude Code 에 로그인했는지 확인합니다…", "ok")
            self._check_login(close_when_ready=True)
            return
        self._say("모두 준비했습니다.", "ok")
        self.accept()

    def _save(self, engine: str, model: installer.LocalModel | None) -> None:
        self.settings.summary_engine = engine
        if model is not None:
            self.settings.local_model = model.id

    # ---- Claude 로그인 -------------------------------------------------------
    def _check_login(self, close_when_ready: bool = False) -> None:
        cli = self.settings.claude_cli_path()
        self._close_when_logged_in = close_when_ready
        if not cli:
            return
        self._login = None
        self._refresh()
        signals = self._signals
        threading.Thread(target=lambda: signals.login.emit(installer.claude_logged_in(cli)), daemon=True).start()

    def _on_login_checked(self, logged_in: bool | None) -> None:
        self._login = logged_in
        self._refresh()
        if getattr(self, "_close_when_logged_in", False):
            if logged_in:
                self.accept()
            else:
                self._say("Claude Code 를 설치했습니다. '로그인 창 열기' 로 로그인한 뒤 '다시 확인' 을 누르세요. "
                          "로그인하지 않으면 요약만 안 되고 녹음·전사는 그대로 됩니다.", "warn")

    def _open_login(self) -> None:
        cli = self.settings.claude_cli_path()
        if cli and not installer.open_claude_login(cli):
            self._say(f"터미널을 열지 못했습니다. 터미널에서 직접 실행하세요: \"{cli}\" auth login", "warn")

    def _say(self, text: str, kind: str) -> None:
        self.status.setStyleSheet(theme.callout(kind))
        self.status.setText(text)
        self.status.show()
