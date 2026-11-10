"""내보내기: 사용자가 고른 경로에만 결과를 저장한다 (자동 저장 없음)."""

from __future__ import annotations

from pathlib import Path

from clinic_scribe.audio import write_wav
from clinic_scribe.chart import REVIEW_BANNER, ChartDraft
from clinic_scribe.session import Session, TranscriptRun
from clinic_scribe.whisper import format_time


def build_markdown(session: Session, run: TranscriptRun | None, draft: ChartDraft | None,
                   include_evidence: bool = True) -> str:
    lines = [
        "# clinic-scribe 기록 (검토용 초안)",
        "",
        f"> {REVIEW_BANNER}",
        "",
        f"- 세션: {session.created_at:%Y-%m-%d %H:%M} · {session.source}",
    ]
    if run is not None:
        r = run.result
        lines.append(f"- 전사: {run.label} · 녹음 길이 {format_time(r.audio_duration)}")
    lines.append("")
    lines.append("## 차트 초안")
    lines.append("")
    if draft is not None:
        lines.append("```text")
        lines.append(draft.to_text(include_evidence=include_evidence))
        lines.append("```")
    else:
        lines.append("(요약 없음)")
    lines.append("")
    if run is not None:
        r = run.result
        lines.append("## 전사문 (시각은 원본 녹음 기준)")
        lines.append("")
        if not r.segments:
            lines.append("(전사된 발화 없음)")
        for seg in r.segments:
            flag = " ⚠ 음성 미감지 구간의 전사(환각 의심)" if not seg.speech_detected else ""
            lines.append(f"- [{format_time(seg.start)} → {format_time(seg.end)}]{flag} {seg.text}")
        lines.append("")
        if r.regions is not None:
            lines.append(f"## 음성 구간 (Silero VAD · {r.vad_params.describe()})")
            lines.append("")
            if not r.regions:
                lines.append("(음성이 감지되지 않음)")
            for i, region in enumerate(r.regions):
                lines.append(f"{i + 1}. {format_time(region.start)} – {format_time(region.end)} ({region.duration:.1f}초)")
            lines.append("")
    return "\n".join(lines)


def export_session(path: Path, session: Session, run: TranscriptRun | None, draft: ChartDraft | None,
                   include_audio: bool, include_evidence: bool = True) -> list[Path]:
    path = path if path.suffix.lower() == ".md" else path.with_suffix(".md")
    path.write_text(build_markdown(session, run, draft, include_evidence), encoding="utf-8")
    written = [path]
    if include_audio and session.pcm16 is not None:
        wav = path.with_suffix(".wav")
        write_wav(wav, session.pcm16)
        written.append(wav)
    return written
