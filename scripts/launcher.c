/* clinic-scribe.app 의 실행 파일. scripts/make_launcher.sh 가 컴파일한다.
 *
 * repo 의 가상환경 Python 을 이 프로세스 안에서 돌려 `python -m clinic_scribe` 와 똑같이 실행한다.
 * python 을 따로 띄우지 않으므로 Dock·메뉴 막대에 'clinic-scribe' 하나만 보이고,
 * 마이크 권한도 이 앱 이름(clinic-scribe)으로 묻는다.
 */
#include <Python.h>

#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "launch_config.h" /* REPO_ROOT (make_launcher.sh 가 만든다) */

#define STR(x) #x
#define XSTR(x) STR(x)
#define PY_LIB_DIR "/.venv/lib/python" XSTR(PY_MAJOR_VERSION) "." XSTR(PY_MINOR_VERSION)

/* 터미널 없이(Finder·Dock) 열었으면 stderr 가 보이지 않으므로 경고창으로도 알린다. */
static int fail(const char *message) {
    fprintf(stderr, "clinic-scribe: %s\n", message);
    if (getppid() == 1) {
        CFStringRef text = CFStringCreateWithCString(NULL, message, kCFStringEncodingUTF8);
        CFUserNotificationDisplayAlert(0, kCFUserNotificationStopAlertLevel, NULL, NULL, NULL,
                                       CFSTR("clinic-scribe"), text, NULL, NULL, NULL, NULL);
        if (text) CFRelease(text);
    }
    return 1;
}

int main(int argc, char *argv[]) {
    char python[4096], path[16384];
    snprintf(python, sizeof python, "%s/.venv/bin/python", REPO_ROOT);
    if (access(python, X_OK) != 0 || chdir(REPO_ROOT) != 0)
        return fail("가상환경을 찾지 못했습니다. repo 를 옮겼다면 ./scripts/make_launcher.sh 를 다시 실행하고, "
                    "처음이면 ./scripts/setup_macos.sh 부터 실행하세요.");
    if (access(REPO_ROOT PY_LIB_DIR, F_OK) != 0)
        return fail("가상환경의 Python 버전이 바뀌었습니다. ./scripts/make_launcher.sh 를 다시 실행하세요.");

    /* Finder 에서 열면 PATH 가 /usr/bin:/bin:/usr/sbin:/sbin 뿐이다. 터미널처럼 흔한 설치 위치를 앞에 둔다. */
    const char *home = getenv("HOME"), *old = getenv("PATH");
    snprintf(path, sizeof path, "%s/.local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:%s",
             home ? home : "", old ? old : "/usr/bin:/bin:/usr/sbin:/sbin");
    setenv("PATH", path, 1);

    /* python -m clinic_scribe [인자…] (옛 macOS 가 붙이는 -psn_ 인자는 버린다) */
    char **args = calloc((size_t)argc + 3, sizeof *args);
    int n = 0;
    args[n++] = python;
    args[n++] = "-m";
    args[n++] = "clinic_scribe";
    for (int i = 1; i < argc; i++)
        if (strncmp(argv[i], "-psn_", 5) != 0) args[n++] = argv[i];

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    /* sys.executable = 가상환경 python: 그 위의 pyvenv.cfg 로 site-packages 를 찾고, 하위 프로세스도 그 python 으로 돈다 */
    PyStatus status = PyConfig_SetBytesString(&config, &config.executable, python);
    if (!PyStatus_Exception(status)) status = PyConfig_SetBytesArgv(&config, n, args);
    if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        fail("Python 을 시작하지 못했습니다. 터미널에서 ./scripts/run.sh 로 실행해 오류를 확인하세요.");
        Py_ExitStatusException(status);
    }
    int code = Py_RunMain();
    if (code != 0)
        fail("clinic-scribe 가 오류로 종료되었습니다. 터미널에서 ./scripts/run.sh 로 실행하면 자세한 내용이 보입니다.");
    return code;
}
