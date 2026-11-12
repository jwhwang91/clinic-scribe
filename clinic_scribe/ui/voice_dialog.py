"""의사 목소리 등록·삭제 화면 (+ '딕션 사전' 탭: 의사별 용어·교정).

등록은 '따라 읽기 학습'(약 1분)으로 한다: 화면의 짧은 문장을 하나씩 따라 읽으면 같은 녹음으로 목소리를 등록하고,
받아 적은 글을 읽은 문장과 비교해 그 의사의 딕션 사전(교정·용어)을 만든다. 녹음해 둔 파일로 목소리만 등록할 수도 있다.
등록한 목소리와 비교해 전사문의 발화마다 '의사 이름' 또는 '환자'를 붙인다(이 컴퓨터 에서 처리).
목소리 특징값은 생체인식정보라 본인 동의를 받은 의사만 등록하고, 이 컴퓨터의 설정 폴더에만 저장한다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from clinic_scribe import mic_permission
from clinic_scribe.audio import AudioFileError, load_audio_file, to_whisper_pcm
from clinic_scribe.diction import DictionBook, DictionError
from clinic_scribe.recorder import InputDevice
from clinic_scribe.speakers import (
    MIN_ENROLL_S,
    SAMPLE_RATE,
    SpeakerEncoder,
    SpeakerError,
    VoiceBook,
    clean_name,
    enrollment_embeddings,
    speech_only,
)
from clinic_scribe.training import GENERAL, SPECIALTIES, build_prompts
from clinic_scribe.ui import theme
from clinic_scribe.ui.diction_dialog import DictionEditor
from clinic_scribe.ui.training_panel import TrainingPanel
from clinic_scribe.ui.widgets import commit_typing


class VoiceDialog(QDialog):
    def __init__(self, book: VoiceBook, encoder: Callable[[], SpeakerEncoder], device: InputDevice | None,
                 parent=None, diction: DictionBook | None = None, transcribe: Callable | None = None,
                 base_prompt: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("의사 목소리 · 딕션 사전" if diction is not None else "의사 목소리 등록")
        self.setMinimumWidth(680)
        self.book = book
        self.diction = diction
        self.changed = False
        self.renamed: list[tuple[str, str]] = []  # 이 창에서 바꾼 의사 이름 (예전 이름, 새 이름)
        self.interactive = True  # False 면 확인 대화상자를 건너뛴다 (자동 검증용)
        self._encoder = encoder
        self._device = device
        self._transcribe = transcribe
        self._pcm: np.ndarray | None = None  # 파일로 등록할 소리 (16 kHz)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(10)
        voice_page = QWidget()
        page_layout = QVBoxLayout(voice_page)
        page_layout.setContentsMargins(0, 8, 0, 0)
        self.voice_stack = QStackedWidget()
        page_layout.addWidget(self.voice_stack)
        self.voice_stack.addWidget(self._build_form())
        self.training: TrainingPanel | None = None
        if transcribe is not None:
            self.training = TrainingPanel(book, diction if diction is not None else DictionBook(Path()), encoder,
                                          transcribe, base_prompt)
            self.training.finished.connect(self._on_training_finished)
            self.voice_stack.addWidget(self.training)

        # 딕션 사전 탭: 의사별 자주 쓰는 용어와 '잘못 들린 말 → 바른 말'
        self.diction_editor: DictionEditor | None = None
        if diction is not None:
            self.diction_editor = DictionEditor(diction, self.book.names)
            self.tabs = QTabWidget()
            self.tabs.addTab(voice_page, "목소리")
            self.tabs.addTab(self.diction_editor, "딕션 사전")
            self.tabs.currentChanged.connect(self._on_tab_changed)
            outer.addWidget(self.tabs, 1)
        else:
            outer.addWidget(voice_page, 1)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.accept)
        bottom.addWidget(self.close_btn)
        outer.addLayout(bottom)
        for button in (self.rename_btn, self.remove_btn, self.train_btn, self.file_btn, self.enroll_btn, self.close_btn):
            button.setAutoDefault(False)  # 이름 칸에서 Enter 를 눌러도 등록·닫기가 되지 않게
        self._refresh()

    def _build_form(self) -> QWidget:
        form = QWidget()
        layout = QVBoxLayout(form)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        title = QLabel("등록된 의사")
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        intro = QLabel("등록한 목소리와 비교해 전사문의 발화마다 의사 이름이나 '환자'를 붙입니다. 비교는 이 컴퓨터 에서 합니다.")
        intro.setObjectName("subtle")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.list = QListWidget()
        self.list.setMaximumHeight(120)
        layout.addWidget(self.list)
        remove_row = QHBoxLayout()
        remove_row.addStretch(1)
        self.rename_btn = QPushButton("이름 바꾸기…")
        self.rename_btn.setToolTip("고른 의사의 이름을 고칩니다. 딕션 사전과 지난 진료 기록의 이름도 함께 바뀝니다.")
        self.rename_btn.clicked.connect(self._on_rename)
        remove_row.addWidget(self.rename_btn)
        self.remove_btn = QPushButton("선택한 목소리 삭제")
        self.remove_btn.clicked.connect(self._on_remove)
        remove_row.addWidget(self.remove_btn)
        layout.addLayout(remove_row)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        new_title = QLabel("새로 등록 · 따라 읽기 학습")
        new_title.setObjectName("cardTitle")
        layout.addWidget(new_title)
        name_row = QHBoxLayout()
        name_label = QLabel("이름")
        name_label.setObjectName("muted")
        name_row.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("예: 김원장")
        self.name_edit.textChanged.connect(self._on_name_changed)
        name_row.addWidget(self.name_edit, 1)
        specialty_label = QLabel("진료과")
        specialty_label.setObjectName("muted")
        name_row.addWidget(specialty_label)
        self.specialty_combo = QComboBox()
        self.specialty_combo.addItems(SPECIALTIES)
        self.specialty_combo.setToolTip("고른 과 외래에서 자주 쓰는 약·검사 이름이 든 문장을 따라 읽습니다. 의사마다 기억합니다.")
        name_row.addWidget(self.specialty_combo)
        layout.addLayout(name_row)
        self.consent = QCheckBox("본인에게 목소리 등록 동의를 받았습니다 (목소리 특징값은 생체정보로, 이 컴퓨터 에만 저장됩니다)")
        layout.addWidget(self.consent)
        guide = QLabel(
            "의사가 컴퓨터 앞에서 화면에 나오는 짧은 문장 12개를 하나씩 따라 읽습니다(약 1분). 문장은 고른 진료과에서 자주 쓰는 "
            "말과 약·검사 이름으로 나옵니다. 같은 녹음으로 목소리를 등록하고, "
            "받아 적은 글을 읽은 문장과 비교해 이 의사의 말이 어떻게 잘못 들리는지 찾아 딕션 사전을 만듭니다. "
            "실제 진료실에서 진료 때 쓰는 마이크로 하는 것이 가장 정확합니다."
        )
        guide.setObjectName("muted")
        guide.setWordWrap(True)
        layout.addWidget(guide)
        buttons = QHBoxLayout()
        self.train_btn = QPushButton("따라 읽기 시작 (약 1분)")
        self.train_btn.setObjectName("primary")
        self.train_btn.clicked.connect(self._on_train)
        buttons.addWidget(self.train_btn)
        buttons.addStretch(1)
        self.file_btn = QPushButton("녹음 파일로 목소리만 등록…")
        self.file_btn.setToolTip("녹음해 둔 파일(WAV 등, 말소리 10초 이상)로 목소리만 등록합니다. 딕션 학습은 하지 않습니다.")
        self.file_btn.clicked.connect(self._on_file)
        buttons.addWidget(self.file_btn)
        self.enroll_btn = QPushButton("파일로 등록")
        self.enroll_btn.clicked.connect(self._on_enroll)
        buttons.addWidget(self.enroll_btn)
        layout.addLayout(buttons)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.hide()
        layout.addWidget(self.message)
        layout.addStretch(1)
        return form

    # ---- 화면 상태 ----------------------------------------------------------
    def _refresh(self) -> None:
        self.list.clear()
        for voice in self.book.voices:
            field = f"   ·   {voice.specialty}" if voice.specialty else ""
            item = QListWidgetItem(f"{voice.name}{field}   ·   등록 말소리 {voice.seconds:.0f}초   ·   {voice.created_at}")
            item.setData(Qt.ItemDataRole.UserRole, voice.name)
            self.list.addItem(item)
        if not self.book.voices:
            empty = QListWidgetItem("아직 등록된 의사가 없습니다")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(empty)
        self.train_btn.setEnabled(self.training is not None and self._device is not None)
        if self.training is None:
            self.train_btn.setToolTip("전사 설정을 찾지 못해 따라 읽기를 할 수 없습니다.")
        elif self._device is None:
            self.train_btn.setToolTip("입력 장치가 없습니다. 마이크를 연결하세요.")
        self.remove_btn.setEnabled(bool(self.book.voices))
        self.rename_btn.setEnabled(bool(self.book.voices))
        self.enroll_btn.setEnabled(self._pcm is not None)

    def _say(self, text: str, kind: str = "note") -> None:
        self.message.setStyleSheet(theme.callout(kind))
        self.message.setText(text)
        self.message.show()

    def _checked_name(self) -> str | None:
        """이름과 동의를 확인한다. 문제가 있으면 안내하고 None."""
        commit_typing()  # 마지막 글자가 조합 중이면 확정한 뒤 읽는다 ('황재웅' 이 '황재' 로 저장되던 문제)
        try:
            name = clean_name(self.name_edit.text())
        except SpeakerError as exc:
            self._say(str(exc), "warn")
            return None
        if not self.consent.isChecked():
            self._say("본인에게 목소리 등록 동의를 받았는지 확인해 주세요.", "warn")
            return None
        if name in self.book.names() and not self._confirm("다시 등록", f"'{name}' 목소리가 이미 있습니다. 새로 등록할까요?"):
            return None
        return name

    def _on_name_changed(self, text: str) -> None:
        """이미 등록한 의사 이름이면 그 의사의 진료과를 고른다 (다시 학습할 때)."""
        known = self.book.specialty(text.strip())
        if known:
            self.specialty_combo.setCurrentText(known)

    # ---- 따라 읽기 학습 ------------------------------------------------------
    def _on_train(self) -> None:
        if self.training is None or self._device is None:
            return
        name = self._checked_name()
        if name is None:
            return
        status = mic_permission.authorization_status()
        if not self._device.demo_file and status in ("denied", "restricted"):
            self._say(mic_permission.denied_message(status), "error")
            return
        if mic_permission.is_builtin_mic(self._device.name) and mic_permission.lid_closed():
            self._say(mic_permission.lid_closed_message([]), "error")  # 덮개를 닫으면 내장 마이크는 0 만 들어온다
            return
        vocabulary = self.diction.entry(name).vocabulary if self.diction is not None else []
        specialty = self.specialty_combo.currentText()
        problem = self.training.start(name, self._device, build_prompts(vocabulary, specialty=specialty), specialty)
        if problem:
            self._say(problem, "error")
            return
        self.voice_stack.setCurrentWidget(self.training)

    def _on_training_finished(self, saved: bool) -> None:
        self.voice_stack.setCurrentIndex(0)
        if saved:
            self.changed = True
            name = self.training.trainee
            learned = len(self.diction.entry(name).corrections) if self.diction is not None else 0
            self.name_edit.clear()
            self.consent.setChecked(False)
            self._say(f"'{name}' 따라 읽기 학습을 저장했습니다 (이 의사 사전의 교정 {learned}개). "
                      "이제부터 전사하는 진료에 적용됩니다.", "ok")
            if self.diction_editor is not None:
                self.diction_editor.refresh_scopes()
        else:
            self._say("따라 읽기를 저장하지 않았습니다. 녹음은 버렸습니다.", "note")
        self._refresh()

    # ---- 녹음 파일로 목소리만 등록 ---------------------------------------------
    def _on_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "등록할 녹음 파일", str(Path.home()), "오디오 (*.wav *.m4a *.mp3 *.aiff *.flac *.caf);;모든 파일 (*)"
        )
        if path:
            self.load_file(Path(path))

    def load_file(self, path: Path) -> None:
        try:
            data, rate = load_audio_file(path)
        except AudioFileError as exc:
            self._say(str(exc), "error")
            return
        pcm16 = to_whisper_pcm(data, rate)
        seconds = len(speech_only(pcm16)) / SAMPLE_RATE
        self._pcm = pcm16
        if seconds < MIN_ENROLL_S:
            self._say(f"{path.name}: 말소리가 {seconds:.0f}초뿐입니다. {MIN_ENROLL_S:.0f}초 이상 필요합니다.", "warn")
        else:
            self._say(f"{path.name}: 말소리 {seconds:.0f}초. 이름과 동의를 확인하고 '파일로 등록'을 누르세요.", "ok")
        self._refresh()

    def _on_enroll(self) -> None:
        if self._pcm is None:
            self._say("먼저 녹음 파일을 고르세요.", "warn")
            return
        name = self._checked_name()
        if name is None:
            return
        try:
            speech = speech_only(self._pcm)
            seconds = len(speech) / SAMPLE_RATE
            if seconds < MIN_ENROLL_S:
                raise SpeakerError(f"말소리가 {seconds:.0f}초뿐입니다. {MIN_ENROLL_S:.0f}초 이상 필요합니다.")
            self.book.enroll(name, enrollment_embeddings(self._encoder(), speech), seconds, self.specialty_combo.currentText())
            self.book.save()
        except SpeakerError as exc:
            self._say(str(exc), "error")
            return
        except OSError as exc:
            self._say(f"저장하지 못했습니다: {exc}", "error")
            return
        self.changed = True
        self._pcm = None
        self.name_edit.clear()
        self.consent.setChecked(False)
        self._say(f"'{name}' 목소리를 등록했습니다. 이제부터 전사하는 진료에 이름이 붙습니다.", "ok")
        self._refresh()

    def _on_rename(self) -> None:
        item = self.list.currentItem()
        old = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not old:
            self._say("이름을 바꿀 목소리를 고르세요.", "warn")
            return
        new, ok = QInputDialog.getText(self, "이름 바꾸기", f"'{old}' 의 새 이름", QLineEdit.EchoMode.Normal, old)
        if ok:
            self.rename(old, new)

    def rename(self, old: str, new: str) -> bool:
        """등록한 의사 이름을 바꾼다. 그 의사의 딕션 사전도 새 이름으로 옮긴다 (지난 진료 기록은 창을 닫은 뒤 바꾼다)."""
        if self.diction_editor is not None:
            self.diction_editor.save(quiet=True)  # 사전 탭에서 고치던 내용을 먼저 반영
        try:
            new = self.book.rename(old, new)
        except SpeakerError as exc:
            self._say(str(exc), "warn")
            return False
        if new == old:
            return False
        if self.diction is not None:
            self.diction.rename_scope(old, new)
        try:
            self.book.save()
            if self.diction is not None:
                self.diction.save()
        except (OSError, DictionError) as exc:
            self._say(f"저장하지 못했습니다: {exc}", "error")
            return False
        if self.diction_editor is not None:
            self.diction_editor.scope_renamed(old, new)
        self.renamed.append((old, new))
        self.changed = True
        self._say(f"'{old}' 을(를) '{new}' 로 바꿨습니다. 딕션 사전과 지난 진료 기록의 이름도 함께 바뀝니다.", "ok")
        self._refresh()
        return True

    def _on_remove(self) -> None:
        item = self.list.currentItem()
        name = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not name:
            self._say("삭제할 목소리를 고르세요.", "warn")
            return
        if not self._confirm("삭제", f"'{name}' 목소리를 삭제할까요? 저장된 목소리 특징값이 지워집니다."):
            return
        self.book.remove(name)
        try:
            self.book.save()
        except OSError as exc:
            self._say(f"저장하지 못했습니다: {exc}", "error")
            return
        self.changed = True
        self._say(f"'{name}' 목소리를 삭제했습니다.", "ok")
        self._refresh()

    def _on_tab_changed(self, index: int) -> None:
        if self.diction_editor is None:
            return
        if self.tabs.widget(index) is self.diction_editor:
            self.diction_editor.refresh_scopes()  # 방금 등록한 의사도 사전 목록에 나오게
        else:
            self.diction_editor.save(quiet=True)

    def _confirm(self, title: str, text: str) -> bool:
        if not self.interactive:
            return True
        answer = QMessageBox.question(self, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def done(self, result: int) -> None:  # noqa: D401 — 닫을 때 마이크를 놓고, 고친 사전을 저장한다
        if self.training is not None and self.training.recording:
            self.training.cancel()
        if self.diction_editor is not None:
            self.diction_editor.save(quiet=True)
        super().done(result)
