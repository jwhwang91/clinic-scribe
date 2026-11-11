"""딕션 사전 화면: 전사문에서 고쳐 쓰기(CorrectionDialog)와 사전 편집(DictionEditor, '의사 목소리' 창의 탭)."""

from __future__ import annotations

import re
from collections.abc import Callable

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from clinic_scribe.diction import COMMON, Correction, DictionBook, DictionError, check_correction
from clinic_scribe.ui import theme

COMMON_LABEL = "공통 (모든 발화)"


def _fill_scopes(combo: QComboBox, scopes: list[str], current: str) -> None:
    combo.blockSignals(True)
    combo.clear()
    for scope in [COMMON] + [s for s in scopes if s != COMMON]:
        combo.addItem(COMMON_LABEL if scope == COMMON else scope, scope)
    combo.setCurrentIndex(max(0, combo.findData(current)))
    combo.blockSignals(False)


class CorrectionDialog(QDialog):
    """전사문에서 고른 말을 고쳐 쓰는 작은 창. 저장하면 같은 말이 다시 나올 때 자동으로 바뀐다."""

    def __init__(self, wrong: str, scopes: list[str], scope: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("선택한 말 고쳐 쓰기")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)
        form = QFormLayout()
        self.wrong_edit = QLineEdit(wrong)
        self.right_edit = QLineEdit()
        self.right_edit.setPlaceholderText("예: 트리메부틴")
        self.scope_combo = QComboBox()
        _fill_scopes(self.scope_combo, scopes, scope)
        self.scope_combo.setToolTip("의사를 고르면 그 의사가 말한 부분에만, 공통은 모든 발화에 적용합니다.")
        form.addRow("잘못 들린 말", self.wrong_edit)
        form.addRow("바른 말", self.right_edit)
        form.addRow("적용할 사전", self.scope_combo)
        layout.addLayout(form)
        self.add_term = QCheckBox("용어 목록에도 넣기 (다음 녹음부터 처음부터 바르게 받아 적도록 힌트로 씀)")
        self.add_term.setChecked(True)
        layout.addWidget(self.add_term)
        note = QLabel("같은 말이 다시 나오면 자동으로 바뀝니다. 띄어쓰기는 무시하고 찾습니다. 원래 들린 말은 전사문에 함께 남습니다.")
        note.setObjectName("subtle")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.hide()
        layout.addWidget(self.message)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        save = QPushButton("저장")
        save.setObjectName("primary")
        save.clicked.connect(self.accept)
        cancel = QPushButton("취소")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(save)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self.right_edit.setFocus()

    def values(self) -> tuple[str, str, str, bool]:
        return self.wrong_edit.text(), self.right_edit.text(), self.scope_combo.currentData(), self.add_term.isChecked()

    def accept(self) -> None:
        wrong, right, _scope, _term = self.values()
        try:
            check_correction(wrong, right)
        except DictionError as exc:
            self.message.setStyleSheet(theme.callout("warn"))
            self.message.setText(str(exc))
            self.message.show()
            return
        super().accept()


class DictionEditor(QWidget):
    """딕션 사전 편집: 공통 또는 의사별 용어 목록과 교정 목록."""

    def __init__(self, diction: DictionBook, scopes: Callable[[], list[str]], parent=None) -> None:
        super().__init__(parent)
        self.diction = diction
        self.changed = False
        self._scopes = scopes
        self._scope = COMMON
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        intro = QLabel(
            "의사 목소리를 등록해 두면 그 의사가 말한 부분에는 그 의사의 사전을, 모든 발화에는 공통 사전을 적용합니다. "
            "전사문에서 틀린 말을 드래그해 오른쪽 클릭 → '선택한 말 고쳐 쓰기'로 고쳐도 여기에 쌓입니다."
        )
        intro.setObjectName("subtle")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        scope_row = QHBoxLayout()
        scope_label = QLabel("사전")
        scope_label.setObjectName("muted")
        scope_row.addWidget(scope_label)
        self.scope_combo = QComboBox()
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        scope_row.addWidget(self.scope_combo, 1)
        layout.addLayout(scope_row)

        vocab_label = QLabel("자주 쓰는 용어 — 쉼표나 줄바꿈으로 구분. 전사할 때 힌트로 넣어 처음부터 바르게 받아 적게 합니다.")
        vocab_label.setObjectName("muted")
        vocab_label.setWordWrap(True)
        layout.addWidget(vocab_label)
        self.vocab_edit = QPlainTextEdit()
        self.vocab_edit.setPlaceholderText("예: 트리메부틴, 레바미피드, 세티리진")
        self.vocab_edit.setFixedHeight(72)
        layout.addWidget(self.vocab_edit)

        rules_label = QLabel("고쳐 쓸 말 — 전사 뒤 자동으로 바꿉니다 (띄어쓰기는 무시).")
        rules_label.setObjectName("muted")
        layout.addWidget(rules_label)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["잘못 들린 말", "바른 말"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for column in (0, 1):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(150)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.add_row_btn = QPushButton("줄 추가")
        self.add_row_btn.clicked.connect(lambda: self._add_row("", ""))
        self.remove_row_btn = QPushButton("선택한 줄 삭제")
        self.remove_row_btn.clicked.connect(self._remove_rows)
        self.save_btn = QPushButton("사전 저장")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(lambda: self.save())
        buttons.addWidget(self.add_row_btn)
        buttons.addWidget(self.remove_row_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.save_btn)
        for button in (self.add_row_btn, self.remove_row_btn, self.save_btn):
            button.setAutoDefault(False)
        layout.addLayout(buttons)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.hide()
        layout.addWidget(self.message)
        self.refresh_scopes()
        self._load(COMMON)

    def refresh_scopes(self) -> None:
        """등록한 의사가 바뀌었을 수 있으니 사전 목록을 다시 채운다."""
        names = self._scopes() + [d for d in self.diction.doctors() if d not in self._scopes()]
        _fill_scopes(self.scope_combo, names, self._scope)

    def select_scope(self, scope: str) -> None:
        self.scope_combo.setCurrentIndex(max(0, self.scope_combo.findData(scope)))

    def _on_scope_changed(self, _index: int) -> None:
        self.save(quiet=True)  # 다른 사전으로 넘어가기 전에 지금 사전을 저장한다
        self._scope = self.scope_combo.currentData() or COMMON
        self._load(self._scope)

    def _load(self, scope: str) -> None:
        entry = self.diction.entry(scope)
        self.vocab_edit.setPlainText(", ".join(entry.vocabulary))
        self.table.setRowCount(0)
        for rule in entry.corrections:
            self._add_row(rule.wrong, rule.right)

    def _add_row(self, wrong: str, right: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(wrong))
        self.table.setItem(row, 1, QTableWidgetItem(right))
        if not wrong:
            self.table.editItem(self.table.item(row, 0))

    def _remove_rows(self) -> None:
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def _cell(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return item.text() if item is not None else ""

    def save(self, quiet: bool = False) -> bool:
        vocabulary = [t for t in re.split(r"[,\n]", self.vocab_edit.toPlainText())]
        rules = [Correction(self._cell(r, 0), self._cell(r, 1)) for r in range(self.table.rowCount())]
        rules = [r for r in rules if r.wrong.strip() or r.right.strip()]
        before = self.diction.entry(self._scope)
        dropped = self.diction.set_entry(self._scope, vocabulary, rules)
        if quiet and self.diction.entry(self._scope) == before:
            return True
        try:
            self.diction.save()
        except OSError as exc:
            self._say(f"저장하지 못했습니다: {exc}", "error")
            return False
        self.changed = True
        self._load(self._scope)  # 정리된 값으로 다시 보여 준다
        if not quiet:
            extra = f" (두 글자 미만·빈 칸·같은 말인 교정 {dropped}개는 뺐습니다)" if dropped else ""
            self._say(f"저장했습니다. 다음 전사부터 적용합니다.{extra}", "warn" if dropped else "ok")
        return True

    def _say(self, text: str, kind: str) -> None:
        self.message.setStyleSheet(theme.callout(kind))
        self.message.setText(text)
        self.message.show()
