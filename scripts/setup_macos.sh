#!/bin/bash
# clinic-scribe 설치 (macOS). 실제 설치는 OS 공통 scripts/setup.py 가 한다:
#   Homebrew whisper.cpp, repo 안 .venv (전역 Python 은 건드리지 않음), 모델(repo 밖 사용자 캐시, SHA-256 검증), 점검.
# 사용: ./scripts/setup_macos.sh            (Linux: ./scripts/setup_linux.sh, Windows: scripts\setup_windows.bat)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$(uname -s)" != "Darwin" ]; then
    echo "macOS 용입니다. Linux 는 ./scripts/setup_linux.sh, Windows 는 scripts\\setup_windows.bat 을 쓰세요." >&2
    exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
    echo "  · Apple Silicon 이 아닙니다. whisper.cpp Metal 가속 없이 느리게 동작할 수 있습니다."
fi
for c in "${CLINIC_SCRIBE_PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3; do
    [ -z "$c" ] && continue
    p="$(command -v "$c" 2>/dev/null)" || continue
    if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        exec "$p" "$REPO_ROOT/scripts/setup.py" "$@"
    fi
done
echo "Python 3.10 이상을 찾지 못했습니다. 아래를 실행한 뒤 다시 실행하세요." >&2
echo "  \$ brew install python@3.12 && ./scripts/setup_macos.sh" >&2
exit 1
