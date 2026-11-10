#!/bin/bash
# clinic-scribe 설치 스크립트 (macOS, Apple Silicon)
#   1) whisper.cpp 의 whisper-cli: 있으면 재사용, 없으면 Homebrew 로 설치
#   2) repo 안 .venv 가상환경과 Python 패키지 (전역 Python 은 건드리지 않음)
#   3) Whisper 모델 + Silero VAD 모델: repo 밖 사용자 캐시에 다운로드, SHA-256 검증
#   4) Claude Code CLI(요약용) 존재·로그인 확인 (전사문은 보내지 않음)
#   5) 가상 음성으로 whisper-cli 자체 점검
# 여러 번 실행해도 된다. 받은 모델은 재사용하고, 끊긴 다운로드는 이어받는다.
# 막힌 단계는 건너뛰고 마지막에 직접 실행할 명령을 모아서 보여준다.
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
MODEL_DIR="${CLINIC_SCRIBE_MODEL_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/clinic-scribe/models}"

WHISPER_MODEL="ggml-large-v3-turbo-q5_0.bin"
WHISPER_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$WHISPER_MODEL"
WHISPER_SHA256="394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"
WHISPER_SIZE=574041195

# whisper.cpp v1.9.x 의 models/download-vad-model.sh 가 받는 공식 Silero 변환본
VAD_MODEL="ggml-silero-v6.2.0.bin"
VAD_URL="https://huggingface.co/ggml-org/whisper-vad/resolve/main/$VAD_MODEL"
VAD_SHA256="2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"
VAD_SIZE=885098

BLOCKED_STEPS=()
BLOCKED_CMDS=()

section() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok() { printf '  ✓ %s\n' "$1"; }
info() { printf '  · %s\n' "$1"; }
blocked() {
    BLOCKED_STEPS+=("$1")
    BLOCKED_CMDS+=("$2")
    printf '  ✗ %s\n' "$1"
}

find_whisper_cli() {
    local c
    for c in "${CLINIC_SCRIBE_WHISPER_CLI:-}" "$(command -v whisper-cli 2>/dev/null)" \
        /opt/homebrew/bin/whisper-cli /usr/local/bin/whisper-cli; do
        if [ -n "$c" ] && [ -x "$c" ]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

find_claude_cli() {
    local c
    for c in "${CLINIC_SCRIBE_CLAUDE_CLI:-}" "$(command -v claude 2>/dev/null)" \
        "$HOME/.local/bin/claude" "$HOME/.claude/local/claude" \
        /opt/homebrew/bin/claude /usr/local/bin/claude; do
        if [ -n "$c" ] && [ -x "$c" ]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

file_size() {
    if [ -f "$1" ]; then stat -f %z "$1"; else echo 0; fi
}

sha256_of() {
    shasum -a 256 "$1" | awk '{print $1}'
}

# fetch_model <파일명> <URL> <SHA-256> <바이트 크기>
# 완성본은 크기와 SHA-256 이 모두 맞을 때만 <파일명>으로 옮긴다. 그 전까지는 <파일명>.part 로 둔다.
fetch_model() {
    local name="$1" url="$2" sha="$3" size="$4"
    local dest="$MODEL_DIR/$name" part="$MODEL_DIR/$name.part"

    if [ -f "$dest" ]; then
        if [ "$(file_size "$dest")" = "$size" ] && [ "$(sha256_of "$dest")" = "$sha" ]; then
            ok "이미 받은 모델 재사용: $dest"
            return 0
        fi
        blocked "$name 이 손상되었거나 다른 파일입니다 (크기 또는 SHA-256 불일치)" \
            "rm \"$dest\" && ./scripts/setup_macos.sh"
        return 1
    fi

    local have
    have="$(file_size "$part")"
    if [ "$have" -gt "$size" ]; then
        rm -f "$part"
        have=0
    fi
    if [ "$have" -lt "$size" ]; then
        if [ "$have" -gt 0 ]; then
            info "이어받기: $have / $size bytes"
        fi
        curl -fL --retry 3 --retry-delay 3 --connect-timeout 30 --progress-bar -C - -o "$part" "$url"
        local rc=$?
        if [ "$rc" -eq 33 ]; then
            # 서버가 이어받기를 지원하지 않으면 처음부터 다시 받는다.
            rm -f "$part"
            curl -fL --retry 3 --retry-delay 3 --connect-timeout 30 --progress-bar -o "$part" "$url"
            rc=$?
        fi
        if [ "$rc" -ne 0 ]; then
            blocked "$name 다운로드 실패 (curl 종료 코드 $rc, 받은 부분은 다음 실행 때 이어받음)" \
                "curl -fL -C - -o \"$part\" \"$url\" && ./scripts/setup_macos.sh"
            return 1
        fi
    fi

    if [ "$(file_size "$part")" = "$size" ] && [ "$(sha256_of "$part")" = "$sha" ]; then
        mv -f "$part" "$dest"
        ok "다운로드·검증 완료: $dest"
        return 0
    fi
    rm -f "$part"
    blocked "$name 검증 실패 (받은 파일의 크기 또는 SHA-256 불일치, 부분 파일 삭제함)" \
        "./scripts/setup_macos.sh   # 처음부터 다시 받는다"
    return 1
}

echo "clinic-scribe 설치"
echo "  repo:   $REPO_ROOT"
echo "  모델:   $MODEL_DIR"
if [ "$(uname -s)" != "Darwin" ]; then
    echo "이 스크립트는 macOS 전용입니다." >&2
    exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
    info "Apple Silicon 이 아닙니다. whisper.cpp Metal 가속 없이 느리게 동작할 수 있습니다."
fi

# ---------------------------------------------------------------------------
section "1/5 whisper.cpp (whisper-cli)"
WHISPER_CLI=""
if WHISPER_CLI="$(find_whisper_cli)"; then
    ok "기존 whisper-cli 재사용: $WHISPER_CLI"
elif command -v brew >/dev/null 2>&1; then
    info "whisper-cli 가 없어 Homebrew 로 설치합니다: brew install whisper-cpp"
    if brew install whisper-cpp && WHISPER_CLI="$(find_whisper_cli)"; then
        ok "설치 완료: $WHISPER_CLI"
    else
        WHISPER_CLI=""
        blocked "whisper.cpp 설치 실패" "brew install whisper-cpp"
    fi
else
    blocked "Homebrew 가 없어 whisper.cpp 를 설치하지 못함" \
        '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" && brew install whisper-cpp'
fi
if [ -n "$WHISPER_CLI" ] && command -v brew >/dev/null 2>&1; then
    WHISPER_VERSION="$(brew list --versions whisper-cpp 2>/dev/null)"
    [ -n "$WHISPER_VERSION" ] && info "Homebrew 버전: $WHISPER_VERSION"
fi

# ---------------------------------------------------------------------------
section "2/5 Python 가상환경 ($VENV_DIR)"
PYTHON_BIN=""
for c in "${CLINIC_SCRIBE_PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3; do
    [ -z "$c" ] && continue
    p="$(command -v "$c" 2>/dev/null)" || continue
    if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON_BIN="$p"
        break
    fi
done
VENV_OK=0
if [ -z "$PYTHON_BIN" ]; then
    blocked "Python 3.10 이상을 찾지 못함" "brew install python@3.12 && ./scripts/setup_macos.sh"
else
    info "기준 Python: $PYTHON_BIN ($("$PYTHON_BIN" -c 'import platform; print(platform.python_version())'))"
    if [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -c 'import sys' 2>/dev/null; then
        ok "기존 가상환경 재사용"
        VENV_OK=1
    elif "$PYTHON_BIN" -m venv --clear "$VENV_DIR"; then
        ok "가상환경 생성"
        VENV_OK=1
    else
        blocked "가상환경 생성 실패" "\"$PYTHON_BIN\" -m venv \"$VENV_DIR\""
    fi
fi
if [ "$VENV_OK" = 1 ]; then
    PIP=("$VENV_DIR/bin/python" -m pip --disable-pip-version-check)
    if "${PIP[@]}" install -q -r "$REPO_ROOT/requirements.txt" -r "$REPO_ROOT/requirements-dev.txt"; then
        ok "Python 패키지 설치 완료"
    else
        blocked "Python 패키지 설치 실패" \
            "\"$VENV_DIR/bin/python\" -m pip install -r requirements.txt -r requirements-dev.txt"
    fi
fi

# ---------------------------------------------------------------------------
section "3/5 Whisper / VAD 모델 ($MODEL_DIR)"
WHISPER_MODEL_OK=0
VAD_MODEL_OK=0
if mkdir -p "$MODEL_DIR"; then
    fetch_model "$WHISPER_MODEL" "$WHISPER_URL" "$WHISPER_SHA256" "$WHISPER_SIZE" && WHISPER_MODEL_OK=1
    fetch_model "$VAD_MODEL" "$VAD_URL" "$VAD_SHA256" "$VAD_SIZE" && VAD_MODEL_OK=1
else
    blocked "모델 폴더를 만들 수 없음" "mkdir -p \"$MODEL_DIR\" && ./scripts/setup_macos.sh"
fi

# ---------------------------------------------------------------------------
section "4/5 Claude Code CLI (차트 초안 요약용)"
if CLAUDE_CLI="$(find_claude_cli)"; then
    ok "claude 실행 파일: $CLAUDE_CLI ($("$CLAUDE_CLI" --version 2>/dev/null | head -1))"
    # 로그인 상태만 확인한다. 이 단계에서는 어떤 전사문도 전송하지 않는다.
    AUTH_JSON="$(env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT "$CLAUDE_CLI" auth status --json 2>/dev/null)"
    if printf '%s' "$AUTH_JSON" | grep -q '"loggedIn": *true'; then
        ok "로그인됨"
    elif printf '%s' "$AUTH_JSON" | grep -q '"loggedIn": *false'; then
        blocked "Claude Code 에 로그인되어 있지 않음" "\"$CLAUDE_CLI\" auth login"
    else
        info "로그인 상태를 확인하지 못했습니다. 요약이 실패하면 터미널에서 'claude' 를 실행해 /login 하세요."
    fi
else
    blocked "claude 실행 파일을 찾지 못함 (요약 기능만 영향)" \
        "curl -fsSL https://claude.ai/install.sh | bash   # 설치 후 'claude' 실행해 로그인"
fi

# ---------------------------------------------------------------------------
section "5/5 whisper-cli 자체 점검 (가상 음성)"
if [ -n "$WHISPER_CLI" ] && [ "$WHISPER_MODEL_OK" = 1 ] && [ "$VAD_MODEL_OK" = 1 ]; then
    TMP_DIR="$(mktemp -d -t clinic-scribe-selftest)"
    if say -v Yuna -o "$TMP_DIR/hello.aiff" "안녕하세요. 설치 점검 중입니다." 2>/dev/null &&
        afconvert -f WAVE -d LEI16@16000 -c 1 "$TMP_DIR/hello.aiff" "$TMP_DIR/hello.wav" 2>/dev/null; then
        if "$WHISPER_CLI" -m "$MODEL_DIR/$WHISPER_MODEL" -f "$TMP_DIR/hello.wav" -l ko -np -oj \
            -of "$TMP_DIR/out" --vad -vm "$MODEL_DIR/$VAD_MODEL" >"$TMP_DIR/log.txt" 2>&1 &&
            [ -s "$TMP_DIR/out.json" ]; then
            ok "whisper-cli + Whisper 모델 + Silero VAD 동작 확인"
        else
            blocked "whisper-cli 자체 점검 실패 (로그: $TMP_DIR/log.txt)" \
                "\"$WHISPER_CLI\" -m \"$MODEL_DIR/$WHISPER_MODEL\" -f \"$TMP_DIR/hello.wav\" -l ko --vad -vm \"$MODEL_DIR/$VAD_MODEL\""
            TMP_DIR=""
        fi
    else
        info "say/afconvert 로 점검 음성을 만들지 못해 자체 점검을 건너뜁니다."
    fi
    [ -n "$TMP_DIR" ] && rm -rf "$TMP_DIR"
else
    info "whisper-cli 또는 모델이 준비되지 않아 건너뜁니다."
fi

# ---------------------------------------------------------------------------
section "결과"
echo "  whisper-cli : ${WHISPER_CLI:-(없음)}"
echo "  Whisper 모델: $MODEL_DIR/$WHISPER_MODEL $([ "$WHISPER_MODEL_OK" = 1 ] && echo '(검증됨)' || echo '(없음)')"
echo "  VAD 모델    : $MODEL_DIR/$VAD_MODEL $([ "$VAD_MODEL_OK" = 1 ] && echo '(검증됨)' || echo '(없음)')"
if [ "${#BLOCKED_STEPS[@]}" -eq 0 ]; then
    echo
    echo "설치 완료. 앱 실행: ./scripts/run.sh"
    exit 0
fi
echo
echo "막힌 단계가 있습니다. 아래 명령을 직접 실행한 뒤 이 스크립트를 다시 실행하세요."
i=0
while [ "$i" -lt "${#BLOCKED_STEPS[@]}" ]; do
    printf '  %d) %s\n     $ %s\n' "$((i + 1))" "${BLOCKED_STEPS[$i]}" "${BLOCKED_CMDS[$i]}"
    i=$((i + 1))
done
exit 1
