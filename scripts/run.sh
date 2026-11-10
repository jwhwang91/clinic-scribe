#!/bin/bash
# clinic-scribe 실행. 추가 인자는 앱으로 전달된다 (예: ./scripts/run.sh --open samples/mock_consult.wav)
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "가상환경이 없습니다. 먼저 ./scripts/setup_macos.sh 를 실행하세요." >&2
    exit 1
fi
# 현재 폴더를 바꾸지 않으므로 --open 의 상대 경로는 실행한 위치 기준이다.
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m clinic_scribe "$@"
