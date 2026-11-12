"""진료 기록 저장소: 앱을 다시 열어도 진료 기록(무음을 줄인 녹음 · 전사문 · 요약)이 남는다.

설정 파일과 같은 폴더의 records/ 에 둔다 (~/Library/Application Support/clinic-scribe/records, Git 밖, 이 컴퓨터 에만).
  records.db          SQLite. 진료 하나가 한 줄: 번호·시각·길이·상태 + 전사문·요약(JSON)
  audio/<번호>.flac   무음을 줄인 녹음 (16 kHz mono, 무손실 FLAC)
진료 녹음·전사문은 민감 정보라 폴더는 700, 파일은 600 으로 만들고, 지운 기록은 DB 안에서도 0 으로 덮는다
(secure_delete). 기록을 지우면 녹음 파일도 함께 지운다.

녹음은 전사를 시작할 때 받은 그대로 먼저 저장하고(전사 도중 앱이 꺼져도 남도록), 전사가 끝나면 Silero VAD
음성 구간으로 긴 무음을 줄인 녹음으로 바꾼다. 그때 전사문 시각도 줄인 녹음 기준으로 옮겨 녹음과 맞춘다.
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import soundfile as sf

from clinic_scribe.audio import WHISPER_SR
from clinic_scribe.chart import ChartDraft, ChartItem
from clinic_scribe.config import VadParams
from clinic_scribe.conversation_summary import ConversationSummary, KeyPointGroup
from clinic_scribe.session import Session, TranscriptRun
from clinic_scribe.speakers import main_doctor
from clinic_scribe.whisper import SpeechRegion, TranscriptionResult, TranscriptSegment

SCHEMA_VERSION = 1
# 이보다 긴 무음은 이 길이로 줄인다. 말 사이의 짧은 쉼은 그대로 두어 자연스럽게 들리고,
# 줄인 녹음을 다시 전사해도 줄 나누기(1초 이상 쉬면 새 줄)가 같게 나온다.
MAX_PAUSE_S = 1.2
FADE_S = 0.01  # 잘라 이은 곳의 '딸깍' 소리를 없애는 짧은 페이드 (무음 한가운데라 들리지 않는다)
# 처리 도중 앱이 꺼진 기록. 다음에 열면 이 안내와 함께 '다시 시도' 버튼이 보인다.
INTERRUPTED = {
    "transcribe": "전사가 끝나기 전에 앱이 종료되었습니다. 녹음은 저장되어 있으니 '전사 다시 시도'를 누르세요.",
    "summary": "요약이 끝나기 전에 앱이 종료되었습니다. 전사문은 저장되어 있으니 '요약 다시 시도'를 누르세요.",
}
_INTERRUPTED_STATUS = {"transcribe": "전사 실패", "summary": "요약 실패"}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source TEXT NOT NULL,
    duration_s REAL NOT NULL DEFAULT 0,
    end_reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    doctor TEXT,
    pending TEXT NOT NULL DEFAULT '',
    audio TEXT,
    audio_s REAL NOT NULL DEFAULT 0,
    audio_trimmed INTEGER NOT NULL DEFAULT 0,
    data TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""
_COLUMNS = ("id", "created_at", "updated_at", "source", "duration_s", "end_reason", "status", "doctor", "pending",
            "audio", "audio_s", "audio_trimmed", "data")


class RecordError(Exception):
    pass


# ---------------------------------------------------------------- 무음 줄이기
@dataclass
class Trimmed:
    pcm16: np.ndarray
    spans: list[tuple[float, float]]  # 남긴 원본 구간(초). 차례로 이어 붙인 것이 pcm16


def trim_silence(pcm16: np.ndarray, regions: list[SpeechRegion], max_pause_s: float = MAX_PAUSE_S,
                 rate: int = WHISPER_SR) -> Trimmed:
    """음성 구간은 그대로 두고 그 사이의 긴 무음을 max_pause_s 로 줄인다 (처음·끝 무음은 절반만 남긴다)."""
    half = int(max_pause_s / 2 * rate)
    spans: list[list[int]] = []
    for region in sorted(regions, key=lambda r: r.start):
        a = max(0, int(region.start * rate) - half)
        b = min(len(pcm16), int(np.ceil(region.end * rate)) + half)
        if b <= a:
            continue
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    fade = int(FADE_S * rate)
    pieces = []
    for i, (a, b) in enumerate(spans):
        piece = pcm16[a:b].astype(np.float32)
        if len(piece) > 2 * fade > 0:
            if i > 0:
                piece[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            if i < len(spans) - 1:
                piece[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        pieces.append(piece)
    out = np.round(np.concatenate(pieces)).astype(np.int16) if pieces else np.zeros(0, np.int16)
    return Trimmed(out, [(a / rate, b / rate) for a, b in spans])


def remap_time(t: float, spans: list[tuple[float, float]]) -> float:
    """원본 녹음 시각 → 무음을 줄인 녹음 시각. 잘라 낸 무음 안의 시각은 이어 붙인 자리로 옮긴다."""
    offset = 0.0
    for a, b in spans:
        if t <= b:
            return offset + max(0.0, t - a)
        offset += b - a
    return offset


def remap_result(result: TranscriptionResult, spans: list[tuple[float, float]]) -> TranscriptionResult:
    """전사 결과의 시각(문장·음성 구간)을 무음을 줄인 녹음 기준으로 옮긴다."""

    def at(t: float) -> float:
        return round(remap_time(t, spans), 3)

    segments = [replace(s, start=at(s.start), end=at(s.end)) for s in result.segments]
    regions = None if result.regions is None else [SpeechRegion(at(r.start), at(r.end)) for r in result.regions]
    return replace(result, segments=segments, regions=regions, phrases=None,
                   audio_duration=round(sum(b - a for a, b in spans), 3))


# ---------------------------------------------------------------- 저장 형식
def _build(cls, data: Any):
    """저장한 dict → 데이터클래스 (모르는 키는 버린다: 나중 버전이 칸을 늘려도 읽힌다)."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


def result_to_dict(result: TranscriptionResult) -> dict:
    data = asdict(result)
    data.pop("phrases", None)  # 줄로 잇기 전 말 덩어리와 낱말은 전사할 때만 쓴다
    for seg in data.get("segments", []):
        seg.pop("words", None)
    return data


def result_from_dict(data: dict) -> TranscriptionResult:
    regions = data.get("regions")
    return TranscriptionResult(
        segments=[_build(TranscriptSegment, s) for s in data.get("segments", [])],
        regions=None if regions is None else [_build(SpeechRegion, r) for r in regions],
        vad_enabled=bool(data.get("vad_enabled")),
        vad_params=_build(VadParams, data.get("vad_params")),
        audio_duration=float(data.get("audio_duration", 0.0)),
        elapsed=float(data.get("elapsed", 0.0)),
        doctor=data.get("doctor"),
    )


def draft_to_dict(draft: ChartDraft | ConversationSummary | None) -> dict | None:
    if draft is None:
        return None
    return {"kind": "general" if isinstance(draft, ConversationSummary) else "chart", **asdict(draft)}


def draft_from_dict(data: dict | None) -> ChartDraft | ConversationSummary | None:
    if not data:
        return None
    meta = {k: data[k] for k in ("model", "based_on", "created_at") if k in data}
    if data.get("kind") == "general":
        return ConversationSummary(
            one_line=data.get("one_line", ""),
            key_points=[_build(KeyPointGroup, g) for g in data.get("key_points", [])],
            action_items=list(data.get("action_items", [])),
            needs_review=list(data.get("needs_review", [])),
            **meta,
        )
    sections = {key: [_build(ChartItem, item) for item in items] for key, items in (data.get("sections") or {}).items()}
    return ChartDraft(sections=sections, **meta)


# 진료에서 그대로 저장하는 값 (전사·요약·녹음과 목록용 정보는 따로)
_SESSION_FIELDS = ("draft_run_id", "summary_error", "summary_note", "summary_short", "summary_stale",
                   "transcribe_error", "transcribe_seconds", "summary_seconds", "original_rate", "original_channels",
                   "offset_s", "auto_continued", "exported")


def session_data(session: Session) -> dict:
    data: dict[str, Any] = {name: getattr(session, name) for name in _SESSION_FIELDS}
    data["runs"] = [{"run_id": r.run_id, "result": result_to_dict(r.result)} for r in session.runs]
    data["draft"] = draft_to_dict(session.draft)
    data["next_run"] = session._next_run
    return data


def apply_data(session: Session, data: dict) -> None:
    """저장한 전사문·요약을 진료에 채운다. 처리 도중 앱이 꺼진 기록은 다시 시도할 수 있게 오류로 표시한다."""
    try:
        runs = [TranscriptRun(int(r["run_id"]), result_from_dict(r["result"])) for r in data.get("runs", [])]
        draft = draft_from_dict(data.get("draft"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordError(f"기록 내용을 읽지 못했습니다: {exc}") from exc
    for name in _SESSION_FIELDS:
        if name in data:
            setattr(session, name, data[name])
    session.runs, session.draft = runs, draft
    session._next_run = max(int(data.get("next_run", 1)), max((r.run_id for r in runs), default=0) + 1)
    if session.stored_pending == "transcribe" and not runs:
        session.transcribe_error = INTERRUPTED["transcribe"]
    elif session.stored_pending == "summary":
        session.summary_error = INTERRUPTED["summary"]
    session.loaded = True


@dataclass
class RecordRow:
    """목록에 보일 정보. 앱을 열 때는 이것만 읽고, 진료를 고르면 전사문·요약(data)을 읽는다."""

    id: int
    created_at: datetime
    source: str
    duration_s: float
    end_reason: str
    status: str
    doctor: str | None
    pending: str  # "transcribe" / "summary": 그 처리 도중 앱이 꺼졌다
    audio: str | None
    audio_s: float
    audio_trimmed: bool


def record_row(session: Session, status: str, pending: str = "") -> dict:
    """저장할 목록용 정보. 녹음 파일 이름·길이는 저장소가 녹음을 쓸 때 채운다."""
    return {
        "id": session.session_id,
        "created_at": session.created_at.isoformat(timespec="seconds"),
        "source": session.source,
        "duration_s": session.duration_s,
        "end_reason": session.end_reason,
        "status": status,
        "doctor": main_doctor(session.runs[-1].result.segments) if session.runs else None,
        "pending": pending,
        "audio_trimmed": int(session.audio_trimmed),
    }


def session_from_row(row: RecordRow) -> Session:
    """목록용 정보만 채운 진료 (고르면 apply_data 로 나머지를 채운다)."""
    session = Session(session_id=row.id, source=row.source, created_at=row.created_at,
                      duration_s=row.duration_s, end_reason=row.end_reason)
    session.loaded = False
    session.stored_audio = bool(row.audio)
    session.audio_trimmed = row.audio_trimmed
    session.stored_status = _INTERRUPTED_STATUS.get(row.pending, row.status)
    session.stored_doctor = row.doctor
    session.stored_pending = row.pending
    return session


# ---------------------------------------------------------------- 저장소
class RecordStore:
    """SQLite 파일 하나 + 녹음 폴더. 부를 때마다 연결을 새로 열어 어느 스레드에서 불러도 된다."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.audio_dir = root / "audio"
        self.db_path = root / "records.db"
        self._lock = threading.Lock()
        try:
            for folder in (root, self.audio_dir):
                folder.mkdir(parents=True, exist_ok=True)
                os.chmod(folder, 0o700)
            with self._db() as db:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise RecordError("더 새로운 버전의 clinic-scribe 가 만든 기록입니다. 앱을 업데이트하세요.")
                db.executescript(_SCHEMA)
                db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            os.chmod(self.db_path, 0o600)
        except (OSError, sqlite3.Error) as exc:
            raise RecordError(f"{root}: {exc}") from exc

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path, timeout=10)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA secure_delete = ON")  # 지운 기록을 DB 파일 안에서도 0 으로 덮는다
            with db:  # 끝나면 commit, 예외면 rollback
                yield db
        finally:
            db.close()

    def allocate_id(self) -> int:
        """새 진료 번호. 지운 번호는 다시 쓰지 않는다."""
        try:
            with self._lock, self._db() as db:
                db.execute("INSERT OR IGNORE INTO counters (name, value) SELECT 'session', IFNULL(MAX(id), 0) FROM records")
                return int(db.execute("UPDATE counters SET value = value + 1 WHERE name = 'session' RETURNING value").fetchone()[0])
        except sqlite3.Error as exc:
            raise RecordError(str(exc)) from exc

    def load_index(self) -> list[RecordRow]:
        """모든 기록의 목록용 정보 (오래된 것부터)."""
        try:
            with self._db() as db:
                rows = db.execute(
                    "SELECT id, created_at, source, duration_s, end_reason, status, doctor, pending, audio, audio_s,"
                    " audio_trimmed FROM records ORDER BY created_at, id"
                ).fetchall()
        except sqlite3.Error as exc:
            raise RecordError(str(exc)) from exc
        out = []
        for r in rows:
            try:
                created = datetime.fromisoformat(r["created_at"])
            except ValueError:
                continue
            out.append(RecordRow(id=r["id"], created_at=created, source=r["source"], duration_s=r["duration_s"],
                                 end_reason=r["end_reason"], status=r["status"], doctor=r["doctor"],
                                 pending=r["pending"], audio=r["audio"], audio_s=r["audio_s"],
                                 audio_trimmed=bool(r["audio_trimmed"])))
        return out

    def load_data(self, record_id: int) -> dict:
        try:
            with self._db() as db:
                row = db.execute("SELECT data FROM records WHERE id = ?", (record_id,)).fetchone()
            return json.loads(row["data"]) if row else {}
        except (sqlite3.Error, ValueError) as exc:
            raise RecordError(f"기록 #{record_id} 를 읽지 못했습니다: {exc}") from exc

    def save(self, row: dict, data: dict, audio: np.ndarray | None = None) -> None:
        """기록을 넣거나 고친다. audio 를 주면 녹음 파일도 새로 쓴다 (주지 않으면 있던 녹음은 그대로)."""
        values = dict(row, updated_at=datetime.now().isoformat(timespec="seconds"),
                      data=json.dumps(data, ensure_ascii=False))
        try:
            if audio is not None and len(audio):
                values["audio"] = self._write_audio(int(row["id"]), audio)
                values["audio_s"] = round(len(audio) / WHISPER_SR, 3)
            names = [n for n in values if n in _COLUMNS]
            updates = ", ".join(f"{n} = excluded.{n}" for n in names if n != "id")
            with self._lock, self._db() as db:
                db.execute(
                    f"INSERT INTO records ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})"
                    f" ON CONFLICT(id) DO UPDATE SET {updates}",
                    [values[n] for n in names],
                )
        except (OSError, sqlite3.Error, RuntimeError) as exc:  # soundfile 오류는 RuntimeError 계열
            raise RecordError(f"기록 #{row['id']} 를 저장하지 못했습니다: {exc}") from exc

    def rename_speaker(self, old: str, new: str) -> int:
        """의사 이름을 고쳤을 때 지난 기록의 말한 사람·의사 표시를 바꾼다 (요약 본문 글자는 그대로). 바꾼 기록 수."""
        changed = 0
        try:
            with self._lock, self._db() as db:
                for row in db.execute("SELECT id, doctor, data FROM records").fetchall():
                    data = json.loads(row["data"])
                    touched = False
                    for run in data.get("runs", []):
                        result = run.get("result", {})
                        if result.get("doctor") == old:
                            result["doctor"], touched = new, True
                        for seg in result.get("segments", []):
                            if seg.get("speaker") == old:
                                seg["speaker"], touched = new, True
                    if touched or row["doctor"] == old:
                        db.execute("UPDATE records SET data = ?, doctor = ? WHERE id = ?",
                                   (json.dumps(data, ensure_ascii=False), new if row["doctor"] == old else row["doctor"],
                                    row["id"]))
                        changed += 1
        except (sqlite3.Error, ValueError) as exc:
            raise RecordError(f"지난 기록의 이름을 바꾸지 못했습니다: {exc}") from exc
        return changed

    def delete(self, record_id: int) -> None:
        try:
            with self._lock, self._db() as db:
                db.execute("DELETE FROM records WHERE id = ?", (record_id,))
            for path in (self.audio_path(record_id), self.audio_dir / f".{record_id}.flac.tmp"):
                path.unlink(missing_ok=True)
        except (OSError, sqlite3.Error) as exc:
            raise RecordError(f"기록 #{record_id} 를 지우지 못했습니다: {exc}") from exc

    def audio_path(self, record_id: int) -> Path:
        return self.audio_dir / f"{record_id}.flac"

    def read_audio(self, record_id: int) -> np.ndarray:
        try:
            data, _rate = sf.read(str(self.audio_path(record_id)), dtype="int16")
        except Exception as exc:  # soundfile 은 형식별로 다른 예외를 던진다
            raise RecordError(f"녹음 파일을 읽지 못했습니다: {exc}") from exc
        return np.ascontiguousarray(data if data.ndim == 1 else data[:, 0])

    def _write_audio(self, record_id: int, pcm16: np.ndarray) -> str:
        """임시 파일에 쓴 뒤 바꿔 넣는다 (쓰는 도중 앱이 꺼져도 이전 녹음은 온전하다)."""
        path = self.audio_path(record_id)
        tmp = self.audio_dir / f".{path.name}.tmp"
        sf.write(str(tmp), np.asarray(pcm16, dtype=np.int16), WHISPER_SR, format="FLAC", subtype="PCM_16")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        return path.name


class RecordSaver:
    """저장·삭제를 백그라운드 스레드 하나에서 차례로 한다 (화면이 멈추지 않고, 저장과 삭제 순서가 뒤바뀌지 않는다).

    지운 기록은 번호를 기억해 두었다가, 그 뒤에 늦게 도착한 저장(처리 중이던 전사 등)을 버린다.
    """

    def __init__(self, store: RecordStore | None, on_error: Callable[[str], None] | None = None) -> None:
        self.store = store
        self.on_error = on_error
        self._queue: queue.Queue = queue.Queue()
        self._deleted: set[int] = set()
        self._lock = threading.Lock()
        self._closed = False
        if store is not None:
            threading.Thread(target=self._run, name="clinic-scribe-records", daemon=True).start()

    def save(self, row: dict, data: dict, audio: np.ndarray | None = None) -> None:
        with self._lock:
            if self.store is not None and not self._closed and row["id"] not in self._deleted:
                self._queue.put(("save", row, data, audio))

    def delete(self, record_id: int) -> None:
        with self._lock:
            if self.store is not None and not self._closed:
                self._deleted.add(record_id)
                self._queue.put(("delete", record_id))

    def rename_speaker(self, old: str, new: str) -> None:
        with self._lock:
            if self.store is not None and not self._closed:
                self._queue.put(("rename", old, new))

    def flush(self, timeout: float = 10.0) -> bool:
        """앞서 맡긴 저장·삭제가 끝날 때까지 기다린다."""
        done = threading.Event()
        with self._lock:
            if self.store is None or self._closed:
                return True
            self._queue.put(("flush", done))
        return done.wait(timeout)

    def close(self, timeout: float = 10.0) -> None:
        self.flush(timeout)
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            try:
                if item[0] == "flush":
                    item[1].set()
                elif item[0] == "delete":
                    self.store.delete(item[1])
                elif item[0] == "rename":
                    self.store.rename_speaker(item[1], item[2])
                else:
                    _kind, row, data, audio = item
                    with self._lock:
                        deleted = row["id"] in self._deleted
                    if not deleted:
                        self.store.save(row, data, audio)
            except RecordError as exc:
                if self.on_error is not None:
                    try:
                        self.on_error(str(exc))
                    except RuntimeError:
                        pass  # 앱 종료 중 신호를 받을 창이 이미 사라짐
