#!/bin/bash
# clinic-scribe 설치 (Linux: Ubuntu·Debian·Fedora·Arch 등, x64·ARM64). 실제 설치는 OS 공통 scripts/setup.py 가 한다:
#   whisper.cpp 공식 빌드(SHA-256 검증), repo 안 .venv, 모델(repo 밖 ~/.cache/clinic-scribe), 시스템 꾸러미 확인, 앱 목록 바로 가기.
# sudo 가 필요한 시스템 꾸러미는 직접 설치하지 않고 실행할 명령을 알려 준다.
# 사용: ./scripts/setup_linux.sh
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for c in "${CLINIC_SCRIBE_PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3; do
    [ -z "$c" ] && continue
    p="$(command -v "$c" 2>/dev/null)" || continue
    if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        exec "$p" "$REPO_ROOT/scripts/setup.py" "$@"
    fi
done
echo "Python 3.10 이상을 찾지 못했습니다. 예: Ubuntu·Debian 은 아래를 실행한 뒤 다시 실행하세요." >&2
echo "  \$ sudo apt install -y python3 python3-venv && ./scripts/setup_linux.sh" >&2
exit 1
