"""의사 목소리 등록·삭제 화면 (+ '딕션 사전' 탭: 의사별 용어·교정).

등록한 목소리와 비교해 전사문의 발화마다 '의사 이름' 또는 '환자'를 붙인다(이 Mac 에서 처리).
목소리 특징값은 생체인식정보라 본인 동의를 받은 의사만 등록하고, 이 Mac 의 설정 폴더에만 저장한다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from clinic_scribe import mic_permission
from clinic_scribe.audio import AudioFileError, load_audio_file, rms_dbfs, to_mono, to_whisper_pcm
from clinic_scribe.diction import DictionBook
from clinic_scribe.recorder import InputDevice, Recorder, RecorderError
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
from clinic_scribe.ui import theme
from clinic_scribe.ui.diction_dialog import DictionEditor
from clinic_scribe.ui.widgets import LevelMeter

# 등록할 때 읽는 문장: 진료 때 말투가 나오도록 진료 대화 문장으로 한다.
READING_SCRIPT = (
    "안녕하세요, 어디가 불편해서 오셨어요? 언제부터 그러셨는지 말씀해 주세요. "
    "식사하고 나서 더 심해지는지, 열이나 기침은 없는지도 알려 주세요. 배를 한번 눌러 볼게요. "
    "오늘은 약을 드리고 경과를 보다가, 계속되면 검사를 해 보겠습니다. "
    "약은 하루 세 번 식후에 드시고, 일주일 뒤에 다시 오세요."
)


class VoiceDialog(QDialog):
    def __init__(self, book: VoiceBook, encoder: Callable[[], SpeakerEncoder], device: InputDevice | None,
                 parent=None, diction: DictionBook | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("의사 목소리 · 딕션 사전" if diction is not None else "의사 목소리 등록")
        self.setMinimumWidth(640)
        self.book = book
        self.changed = False
        self.interactive = True  # False 면 확인 대화상자를 건너뛴다 (자동 검증용)
        self._encoder = encoder
        self._device = device
        self._recorder = Recorder()
        self._pcm: np.ndarray | None = None  # 등록할 소리 (16 kHz)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_tick)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(10)
        voice_page = QWidget()
        layout = QVBoxLayout(voice_page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)
        title = QLabel("등록된 의사")
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        intro = QLabel("등록한 목소리와 비교해 전사문의 발화마다 의사 이름이나 '환자'를 붙입니다. 비교는 이 Mac 에서 합니다.")
        intro.setObjectName("subtle")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.list = QListWidget()
        self.list.setMaximumHeight(120)
        layout.addWidget(self.list)
        remove_row = QHBoxLayout()
        remove_row.addStretch(1)
        self.remove_btn = QPushButton("선택한 목소리 삭제")
        self.remove_btn.clicked.connect(self._on_remove)
        remove_row.addWidget(self.remove_btn)
        layout.addLayout(remove_row)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        new_title = QLabel("새로 등록")
        new_title.setObjectName("cardTitle")
        layout.addWidget(new_title)
        name_row = QHBoxLayout()
        name_label = QLabel("이름")
        name_label.setObjectName("muted")
        name_row.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("예: 김원장")
        name_row.addWidget(self.name_edit, 1)
        layout.addLayout(name_row)
        self.consent = QCheckBox("본인에게 목소리 등록 동의를 받았습니다 (목소리 특징값은 생체정보로, 이 Mac 에만 저장됩니다)")
        layout.addWidget(self.consent)
        guide = QLabel(f"아래 문장을 평소 진료하실 때처럼 읽어 주세요. 말소리 {MIN_ENROLL_S:.0f}초 이상, 20~30초를 권합니다.")
        guide.setObjectName("muted")
        guide.setWordWrap(True)
        layout.addWidget(guide)
        script = QLabel(f"“{READING_SCRIPT}”")
        script.setWordWrap(True)
        script.setStyleSheet(theme.callout("note") + " font-size: 14px;")
        layout.addWidget(script)

        record_row = QHBoxLayout()
        record_row.setSpacing(10)
        self.record_btn = QPushButton()
        self.record_btn.clicked.connect(self._on_record)
        record_row.addWidget(self.record_btn)
        self.elapsed = QLabel("00:00")
        self.elapsed.setFont(theme.timer_font(22))
        self.elapsed.setStyleSheet("color: #f1f3f7;")
        record_row.addWidget(self.elapsed)
        self.level = LevelMeter()
        record_row.addWidget(self.level, 1)
        self.file_btn = QPushButton("파일에서…")
        self.file_btn.setToolTip("녹음해 둔 파일(WAV 등)로 등록합니다.")
        self.file_btn.clicked.connect(self._on_file)
        record_row.addWidget(self.file_btn)
        layout.addLayout(record_row)

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.hide()
        layout.addWidget(self.message)

        enroll_row = QHBoxLayout()
        enroll_row.addStretch(1)
        self.enroll_btn = QPushButton("등록")
        self.enroll_btn.setObjectName("primary")
        self.enroll_btn.clicked.connect(self._on_enroll)
        enroll_row.addWidget(self.enroll_btn)
        layout.addLayout(enroll_row)

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
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        outer.addLayout(bottom)
        for button in (self.remove_btn, self.record_btn, self.file_btn, self.enroll_btn, close_btn):
            button.setAutoDefault(False)  # 이름 칸에서 Enter 를 눌러도 등록·닫기가 되지 않게
        self._refresh()

    def _on_tab_changed(self, index: int) -> None:
        if self.diction_editor is None:
            return
        if self.tabs.widget(index) is self.diction_editor:
            self.diction_editor.refresh_scopes()  # 방금 등록한 의사도 사전 목록에 나오게
        else:
            self.diction_editor.save(quiet=True)

    # ---- 화면 상태 ----------------------------------------------------------
    def _refresh(self) -> None:
        self.list.clear()
        for voice in self.book.voices:
            item = QListWidgetItem(f"{voice.name}   ·   등록 말소리 {voice.seconds:.0f}초   ·   {voice.created_at}")
            item.setData(Qt.ItemDataRole.UserRole, voice.name)
            self.list.addItem(item)
        if not self.book.voices:
            empty = QListWidgetItem("아직 등록된 의사가 없습니다")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(empty)
        recording = self._recorder.active
        self.record_btn.setText("■  녹음 종료" if recording else "●  녹음 시작")
        self.record_btn.setEnabled(recording or self._device is not None)
        self.remove_btn.setEnabled(bool(self.book.voices) and not recording)
        self.file_btn.setEnabled(not recording)
        self.enroll_btn.setEnabled(self._pcm is not None and not recording)

    def _say(self, text: str, kind: str = "note") -> None:
        self.message.setStyleSheet(theme.callout(kind))
        self.message.setText(text)
        self.message.show()

    # ---- 등록할 소리 --------------------------------------------------------
    def _on_record(self) -> None:
        if self._recorder.active:
            self._timer.stop()
            audio, rate = self._recorder.stop(keep_audio=True)
            self.level.set_level(-120.0)
            self._set_audio(to_whisper_pcm(audio, rate) if len(audio) else np.zeros(0, np.int16), "녹음")
            return
        if self._device is None:
            self._say("입력 장치가 없습니다. 파일에서 등록하세요.", "error")
            return
        status = mic_permission.authorization_status()
        if not self._device.demo_file and status in ("denied", "restricted"):
            self._say(mic_permission.denied_message(status), "error")
            return
        try:
            self._recorder.start(self._device)
        except RecorderError as exc:
            self._say(str(exc), "error")
            return
        self._pcm = None
        self.elapsed.setText("00:00")
        self._timer.start()
        self._say("위 문장을 읽어 주세요. 다 읽으면 '녹음 종료'를 누르세요.", "note")
        self._refresh()

    def _on_tick(self) -> None:
        blocks = self._recorder.drain()
        if blocks:
            self.level.set_level(rms_dbfs(to_mono(np.concatenate(blocks))))
        seconds = int(self._recorder.elapsed)
        self.elapsed.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")

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
        self._set_audio(to_whisper_pcm(data, rate), path.name)

    def _set_audio(self, pcm16: np.ndarray, source: str) -> None:
        seconds = len(speech_only(pcm16)) / SAMPLE_RATE
        self._pcm = pcm16
        if seconds < MIN_ENROLL_S:
            self._say(f"{source}: 말소리가 {seconds:.0f}초뿐입니다. {MIN_ENROLL_S:.0f}초 이상 읽어 주세요.", "warn")
        else:
            self._say(f"{source}: 말소리 {seconds:.0f}초. 이름과 동의를 확인하고 '등록'을 누르세요.", "ok")
        self._refresh()

    # ---- 등록·삭제 ----------------------------------------------------------
    def _on_enroll(self) -> None:
        if self._pcm is None:
            self._say("먼저 녹음하거나 파일을 고르세요.", "warn")
            return
        if not self.consent.isChecked():
            self._say("본인에게 목소리 등록 동의를 받았는지 확인해 주세요.", "warn")
            return
        try:
            name = clean_name(self.name_edit.text())
            speech = speech_only(self._pcm)
            seconds = len(speech) / SAMPLE_RATE
            if seconds < MIN_ENROLL_S:
                raise SpeakerError(f"말소리가 {seconds:.0f}초뿐입니다. {MIN_ENROLL_S:.0f}초 이상 필요합니다.")
            if name in self.book.names() and not self._confirm("다시 등록", f"'{name}' 목소리가 이미 있습니다. 새 녹음으로 바꿀까요?"):
                return
            self.book.enroll(name, enrollment_embeddings(self._encoder(), speech), seconds)
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
        self.elapsed.setText("00:00")
        self._say(f"'{name}' 목소리를 등록했습니다. 이제부터 전사하는 진료에 이름이 붙습니다.", "ok")
        self._refresh()

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

    def _confirm(self, title: str, text: str) -> bool:
        if not self.interactive:
            return True
        answer = QMessageBox.question(self, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def done(self, result: int) -> None:  # noqa: D401 — 닫을 때 마이크를 놓고, 고친 사전을 저장한다
        self._timer.stop()
        if self._recorder.active:
            self._recorder.abort()
        if self.diction_editor is not None:
            self.diction_editor.save(quiet=True)
        super().done(result)
