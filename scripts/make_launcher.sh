#!/bin/bash
# 클릭해서 여는 앱을 만든다: ~/Applications/clinic-scribe.app
# Spotlight(⌘Space → clinic) · Launchpad 에서 열거나 Dock 에 끌어다 두면 된다. 지우려면 앱을 휴지통으로 옮긴다.
# 앱은 이 repo 의 가상환경으로 python -m clinic_scribe 를 실행한다 (scripts/launcher.c).
# repo 를 옮겼거나 가상환경을 다시 만들었으면 이 스크립트를 다시 실행하세요.
# 처음 녹음할 때 macOS 가 'clinic-scribe' 의 마이크 접근을 묻습니다 (앱을 다시 만들면 한 번 더 묻습니다).
#
# 사용: ./scripts/make_launcher.sh [만들 위치.app]
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$HOME/Applications/clinic-scribe.app}"
BUNDLE_ID="local.clinic-scribe"  # clinic_scribe/mic_permission.py 의 APP_BUNDLE_ID 와 같게
PY="$REPO_ROOT/.venv/bin/python"
case "$APP" in
    *.app) ;;
    *) echo "만들 위치는 .app 으로 끝나야 합니다: $APP" >&2; exit 1 ;;
esac
if [ ! -x "$PY" ]; then
    echo "가상환경이 없습니다. 먼저 ./scripts/setup_macos.sh 를 실행하세요." >&2
    exit 1
fi
if ! xcrun --find clang >/dev/null 2>&1; then
    echo "C 컴파일러가 없습니다. xcode-select --install 로 명령행 도구를 설치한 뒤 다시 실행하세요." >&2
    exit 1
fi

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
STAGE="$BUILD/$(basename "$APP")"
mkdir -p "$STAGE/Contents/MacOS" "$STAGE/Contents/Resources"

# 가상환경 Python 의 헤더와 공유 라이브러리 (Homebrew·python.org 는 framework, Anaconda·uv 는 libpython*.dylib)
"$PY" - "$REPO_ROOT" "$BUILD" <<'PYCODE'
import json, os, sys, sysconfig

repo, build = sys.argv[1:]
v = sysconfig.get_config_var
ver = v("VERSION")
if v("PYTHONFRAMEWORK"):
    lib = os.path.join(v("PYTHONFRAMEWORKPREFIX"), v("PYTHONFRAMEWORK") + ".framework", "Versions", ver, v("PYTHONFRAMEWORK"))
else:
    lib = os.path.join(v("LIBDIR"), f"libpython{ver}.dylib")
with open(os.path.join(build, "launch_config.h"), "w", encoding="utf-8") as f:
    f.write(f"#define REPO_ROOT {json.dumps(repo, ensure_ascii=False)}\n")
with open(os.path.join(build, "python.txt"), "w", encoding="utf-8") as f:
    f.write(sysconfig.get_paths()["include"] + "\n" + (lib if os.path.isfile(lib) else "") + "\n")
PYCODE
{ read -r PY_INCLUDE; read -r PY_LIB; } < "$BUILD/python.txt"
if [ -z "$PY_LIB" ]; then
    echo "이 가상환경의 Python 에는 앱에 넣을 공유 라이브러리(libpython)가 없습니다." >&2
    echo "Homebrew python 등으로 가상환경을 다시 만들거나, 터미널에서 ./scripts/run.sh 로 실행하세요." >&2
    exit 1
fi

# 실행 파일. 메인 스레드 스택은 python 실행 파일과 같은 16MB.
xcrun clang -O2 -Wall -I"$PY_INCLUDE" -I"$BUILD" \
    -Wl,-stack_size,0x1000000 -Wl,-rpath,"$(dirname "$PY_LIB")" \
    -o "$STAGE/Contents/MacOS/clinic-scribe" "$REPO_ROOT/scripts/launcher.c" "$PY_LIB" -framework CoreFoundation

cat > "$STAGE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>clinic-scribe</string>
    <key>CFBundleDisplayName</key><string>clinic-scribe</string>
    <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
    <key>CFBundleExecutable</key><string>clinic-scribe</string>
    <key>CFBundleIconFile</key><string>clinic-scribe</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>진료 대화를 녹음해 이 Mac 에서 전사합니다.</string>
</dict>
</plist>
PLIST

# 아이콘: 앱 위쪽의 로고와 같은 그림 (Qt 로 그려 .icns 로 묶는다)
ICONSET="$BUILD/clinic-scribe.iconset"
mkdir -p "$ICONSET"
if QT_QPA_PLATFORM=offscreen PYTHONPATH="$REPO_ROOT" "$PY" - "$ICONSET" <<'PYCODE'
import sys
from PySide6.QtGui import QGuiApplication
from clinic_scribe.ui.widgets import brand_image

app = QGuiApplication([])
out = sys.argv[1]
for size in (16, 32, 128, 256, 512):
    brand_image(size).save(f"{out}/icon_{size}x{size}.png")
    brand_image(size * 2).save(f"{out}/icon_{size}x{size}@2x.png")
PYCODE
then
    iconutil -c icns "$ICONSET" -o "$STAGE/Contents/Resources/clinic-scribe.icns" || echo "아이콘을 만들지 못했습니다 (기본 아이콘 사용)."
else
    echo "아이콘을 만들지 못했습니다 (기본 아이콘 사용)."
fi

# 로컬 서명: 마이크 권한이 이 앱에 붙는다
codesign --force --sign - "$STAGE"

rm -rf "$APP"
mkdir -p "$(dirname "$APP")"
mv "$STAGE" "$APP"
case "$APP" in
    */Applications/*)  # Spotlight·Launchpad 에 바로 보이게 등록
        /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" 2>/dev/null || true ;;
esac
echo "만들었습니다: $APP"
echo "Spotlight(⌘Space → clinic) · Launchpad 에서 열거나, Finder 에서 Dock 으로 끌어다 두세요."
