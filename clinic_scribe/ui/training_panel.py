"""따라 읽기 학습 화면 ('의사 목소리' 창 안): 문장을 하나씩 보여 주고 읽으면 다음 문장으로 넘어간다.

다 읽으면 이 Mac 에서 분석해 목소리 등록과 딕션 사전(교정·용어)을 만들고, 의사가 고른 것만 저장한다.
녹음은 분석이 끝나면 버린다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from clinic_scribe.audio import rms_dbfs, to_mono, to_whisper_pcm
from clinic_scribe.diction import DictionBook, DictionError
from clinic_scribe.live_monitor import LiveActivityMonitor
from clinic_scribe.recorder import InputDevice, Recorder, RecorderError
from clinic_scribe.speakers import MIN_ENROLL_S, SpeakerEncoder, SpeakerError, VoiceBook
from clinic_scribe.training import TrainingResult, analyze
from clinic_scribe.ui import theme
from clinic_scribe.ui.tasks import Job, TaskRunner
from clinic_scribe.ui.widgets import LevelMeter


class TrainingPanel(QWidget):
    finished = Signal(bool)  # 저장했으면 True

    def __init__(self, voices: VoiceBook, diction: DictionBook, encoder: Callable[[], SpeakerEncoder],
                 transcribe: Callable, base_prompt: str, parent=None) -> None:
        super().__init__(parent)
        self.voices = voices
        self.diction = diction
        self._encoder = encoder
        self._transcribe = transcribe
        self._base_prompt = base_prompt
        self._recorder = Recorder()
        # 읽다가 쉼표에서 잠깐 쉬어도 넘어가지 않게 0.9초 쉬어야 '다 읽음'으로 본다
        self._monitor = LiveActivityMonitor(min_silence_ms=900, max_clip_s=0)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._on_tick)
        self._tasks = TaskRunner(self)
        self._tasks.finished.connect(self._on_analyzed)
        self._name = ""
        self._prompts: list[str] = []
        self._marks: list[float] = []
        self._index = 0
        self._heard = False
        self.result: TrainingResult | None = None
        self.error: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)

        # 1) 읽기
        reading = QWidget()
        rl = QVBoxLayout(reading)
        rl.setSpacing(12)
        self.progress = QLabel()
        self.progress.setObjectName("subtle")
        rl.addWidget(self.progress)
        self.prompt_label = QLabel()
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setMinimumHeight(120)
        self.prompt_label.setStyleSheet(
            "background: #1b1f29; color: #f1f3f7; border: 1px solid #2a303c; border-radius: 12px;"
            " padding: 16px; font-size: 24px; font-weight: 600;"
        )
        rl.addWidget(self.prompt_label)
        self.state = QLabel()
        self.state.setObjectName("muted")
        self.state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(self.state)
        self.level = LevelMeter()
        rl.addWidget(self.level)
        reading_buttons = QHBoxLayout()
        self.redo_btn = QPushButton("다시 읽기")
        self.redo_btn.clicked.connect(self.redo)
        self.next_btn = QPushButton("다음 ▶")
        self.next_btn.clicked.connect(self.next_prompt)
        self.stop_btn = QPushButton("그만두기")
        self.stop_btn.setObjectName("ghost")
        self.stop_btn.clicked.connect(self.cancel)
        reading_buttons.addWidget(self.stop_btn)
        reading_buttons.addStretch(1)
        reading_buttons.addWidget(self.redo_btn)
        reading_buttons.addWidget(self.next_btn)
        rl.addLayout(reading_buttons)
        rl.addStretch(1)
        self.pages.addWidget(reading)

        # 2) 분석 중
        busy = QWidget()
        bl = QVBoxLayout(busy)
        bl.addStretch(1)
        busy_label = QLabel("분석 중…\n같은 녹음으로 목소리를 등록하고, 받아 적은 글을 읽은 문장과 비교합니다 (이 Mac).")
        busy_label.setObjectName("muted")
        busy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(busy_label)
        bl.addStretch(1)
        self.pages.addWidget(busy)

        # 3) 결과
        review = QWidget()
        vl = QVBoxLayout(review)
        vl.setSpacing(10)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        vl.addWidget(self.summary)
        findings_title = QLabel("이렇게 잘못 들렸습니다 — 맞는 것만 남기고 저장하면 다음 진료부터 자동으로 고쳐집니다.")
        findings_title.setObjectName("muted")
        findings_title.setWordWrap(True)
        vl.addWidget(findings_title)
        self.findings = QListWidget()
        vl.addWidget(self.findings, 1)
        review_buttons = QHBoxLayout()
        self.discard_btn = QPushButton("저장하지 않기")
        self.discard_btn.clicked.connect(self.cancel)
        self.save_btn = QPushButton("목소리와 딕션 사전 저장")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self.save)
        review_buttons.addStretch(1)
        review_buttons.addWidget(self.discard_btn)
        review_buttons.addWidget(self.save_btn)
        vl.addLayout(review_buttons)
        self.pages.addWidget(review)
        for button in (self.redo_btn, self.next_btn, self.stop_btn, self.discard_btn, self.save_btn):
            button.setAutoDefault(False)

    @property
    def recording(self) -> bool:
        return self._recorder.active

    @property
    def trainee(self) -> str:
        """따라 읽는 의사 이름."""
        return self._name

    # ---- 읽기 ---------------------------------------------------------------
    def start(self, name: str, device: InputDevice, prompts: list[str]) -> str | None:
        """녹음을 시작한다. 시작하지 못하면 그 이유를 돌려준다."""
        try:
            self._recorder.start(device)
        except RecorderError as exc:
            return str(exc)
        self._name, self._prompts, self._index, self._marks = name, list(prompts), 0, [0.0]
        self.result, self.error, self._heard = None, None, False
        self._monitor.reset()
        self.pages.setCurrentIndex(0)
        self._show_prompt()
        self._timer.start()
        return None

    def _show_prompt(self) -> None:
        self.progress.setText(f"{self._index + 1} / {len(self._prompts)} · 다 읽고 잠깐 쉬면 다음 문장으로 넘어갑니다")
        self.prompt_label.setText(self._prompts[self._index])
        self.state.setText("평소 진료하실 때처럼 읽어 주세요")

    def _on_tick(self) -> None:
        blocks = self._recorder.drain()
        if not blocks:
            return
        mono = to_mono(np.concatenate(blocks))
        self.level.set_level(rms_dbfs(mono))
        for event in self._monitor.feed(mono, self._recorder.samplerate):
            if event in ("start", "split"):
                self._heard = True
                self.state.setText("듣는 중…")
            elif event == "end" and self._heard:
                self.next_prompt()
                return

    def next_prompt(self) -> None:
        if not self._recorder.active:
            return
        self._marks.append(self._recorder.elapsed)
        self._index += 1
        self._heard = False
        if self._index >= len(self._prompts):
            self._finish_recording()
        else:
            self._show_prompt()

    def redo(self) -> None:
        """이 문장을 처음부터 다시 읽는다 (앞서 읽은 소리는 버린다)."""
        if self._recorder.active:
            self._marks[-1] = self._recorder.elapsed
            self._heard = False
            self.state.setText("다시 읽어 주세요")

    def cancel(self) -> None:
        self._timer.stop()
        if self._recorder.active:
            self._recorder.abort()
        self.finished.emit(False)

    def _finish_recording(self) -> None:
        self._timer.stop()
        audio, rate = self._recorder.stop(keep_audio=True)
        self.level.set_level(-120.0)
        clips = []
        for start, end in zip(self._marks, self._marks[1:]):
            part = audio[int(start * rate) : int(end * rate)]
            clips.append(to_whisper_pcm(part, rate) if len(part) else np.zeros(0, np.int16))
        self.analyze_clips(clips)

    # ---- 분석·저장 ------------------------------------------------------------
    def analyze_clips(self, clips: list[np.ndarray], name: str | None = None, prompts: list[str] | None = None) -> None:
        """문장별 녹음을 분석한다 (녹음이 끝나면 부르고, 시험할 때는 직접 부른다)."""
        if name is not None:
            self._name = name
        if prompts is not None:
            self._prompts = list(prompts)
        self.result, self.error = None, None
        self.pages.setCurrentIndex(1)
        texts, transcribe, encoder, base = list(self._prompts), self._transcribe, self._encoder, self._base_prompt
        self._tasks.start(Job("training", 0, 0), lambda cancel: analyze(texts, clips, transcribe, encoder, base, cancel))

    def _on_analyzed(self, _task, result: TrainingResult | None, error) -> None:
        self.findings.clear()
        if error is not None or result is None:
            self.error = f"분석하지 못했습니다: {error}"
            self.summary.setStyleSheet(theme.callout("error"))
            self.summary.setText(self.error)
            self.save_btn.setEnabled(False)
            self.pages.setCurrentIndex(2)
            return
        self.result = result
        total = len(result.prompts)
        if result.embeddings and result.speech_seconds >= MIN_ENROLL_S:
            voice = f"목소리: 말소리 {result.speech_seconds:.0f}초로 등록합니다."
        else:
            voice = f"목소리는 등록하지 않습니다 ({result.voice_issue or f'말소리 {result.speech_seconds:.0f}초뿐'})."
        self.summary.setStyleSheet(theme.callout("ok"))
        self.summary.setText(
            f"정확히 받아 적은 문장: 학습 전 {result.correct_before}/{total} → 학습 후 {result.correct_after}/{total}\n{voice}"
        )
        for finding in result.findings:
            item = QListWidgetItem(f"들림: {finding.heard}   →   읽은 말: {finding.expected}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, finding)
            self.findings.addItem(item)
        if not result.findings:
            empty = QListWidgetItem("잘못 들린 말이 없습니다 — 이 의사의 말은 지금도 잘 받아 적습니다.")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.findings.addItem(empty)
        self.save_btn.setEnabled(True)
        self.pages.setCurrentIndex(2)

    def save(self) -> None:
        result = self.result
        if result is None:
            return
        try:
            if result.embeddings and result.speech_seconds >= MIN_ENROLL_S:
                self.voices.enroll(self._name, result.embeddings, result.speech_seconds)
                self.voices.save()
            for row in range(self.findings.count()):
                item = self.findings.item(row)
                finding = item.data(Qt.ItemDataRole.UserRole)
                if finding is not None and item.checkState() == Qt.CheckState.Checked:
                    try:
                        self.diction.add_correction(self._name, finding.heard, finding.expected)
                    except DictionError:
                        continue  # 한 글자짜리 등은 교정으로 쓸 수 없다
            self.diction.save()
        except (SpeakerError, OSError) as exc:
            self.summary.setStyleSheet(theme.callout("error"))
            self.summary.setText(f"저장하지 못했습니다: {exc}")
            return
        self.finished.emit(True)
