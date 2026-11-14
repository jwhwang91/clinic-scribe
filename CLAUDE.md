# clinic-scribe — Claude Code 작업 지침

진료 대화를 녹음해 이 컴퓨터에서 whisper.cpp 로 전사하고, 말한 사람(의사·환자)을 표시하고, 요약(Claude Code CLI 또는 로컬 llama.cpp)해
진료 기록으로 남기는 데스크톱 앱 (macOS·Windows·Linux, PySide6). 병원 EMR 은 씨차트(Windows)이고, 녹음 버튼과 씨차트 연결로 이어진다.

- 사용법·설치·파일 구조: `README.md`
- **지금까지 한 일, 결정과 이유, 현재 상태, 다음 할 일: `docs/WORKLOG.md`. 작업을 시작하기 전에 먼저 읽는다.**
- 답과 보고는 한국어로 한다.

## 꼭 지킬 것

- Python 가상환경은 repo 안 `.venv` 에 만들고 기존 전역 환경은 건드리지 않는다.
- Whisper·VAD·화자 인식 모델과 받은 whisper.cpp 는 repo 밖 사용자 캐시에 둔다 (`config.cache_dir()`).
- 모델, 녹음, 전사문, 요약은 Git 에 넣지 않는다. 이 저장소는 **공개(public)** 다: 환자 정보·개인 정보를 커밋·문서·이슈에 쓰지 않는다.
- 요약에 대화에 없는 진단, 정상 소견, 처방, 용량을 추가하지 않는다. 잘못 들린 말은 몰래 고치지 않고 `확인 필요` 에 남긴다.
  화면과 복사 결과에 `검토용 초안` 임을 표시한다.
- 전사는 이 컴퓨터에서만 한다. 요약은 사용자가 고른 방식(Claude Code CLI 또는 로컬 llama.cpp)으로만 하고, 다른 외부 API 를 더하지 않는다.
- 진료 내용, 전사문·요약 원문, 환자 이름을 콘솔·로그·오류 메시지에 출력하지 않는다.
- 외부 프로그램은 `shell=False` 로 실행하고 전사문은 stdin 으로 넘긴다 (`procutil.run`).
- 검증에는 가상 진료 대화만 쓴다: `scripts/make_test_audio.py` 의 가상 음성, `tests/` 의 가짜 claude·llama-server·진료 차트 창.
  실제 Claude 요약 테스트는 `CLINIC_SCRIBE_TEST_CLAUDE=1` 일 때만 돈다 (가상 전사문이 전송된다).
- 의사 목소리 특징값은 생체정보다: 동의를 받고, 이 컴퓨터에만 두고, 파일 권한은 600.
- 환자 이름은 이 컴퓨터에만 두고 요약 요청에 넣지 않는다. 씨차트 점검 결과는 이름을 `[환자 이름]` 으로 가리고,
  설정에는 자리 구조와 가린 모양만 저장한다.
- 씨차트: 숨은 다운로드 주소를 추측하지 않고 보호 장치를 우회하지 않는다 (후킹·메모리 읽기 없이 화면 읽기(UI Automation)와 창 글자만).
  요약은 같은 환자일 때만 진료 기록 칸 끝에 덧붙이고, 저장 단추는 누르지 않는다.
- 인터넷에서 받은 실행 파일(llama-server 등)을 직접 돌려 확인하려면 먼저 사용자에게 묻는다. 평소에는 가짜로 확인한다.
- 사용자가 쓰고 있는 앱을 강제로 끄지 않는다 (대화상자가 떠 있을 수 있다). 앱이 켜져 있을 때 사용자 데이터를 고치지 않는다.

## 병원 PC (씨차트가 있는 Windows) 에서 작업할 때

이 PC 에는 실제 환자 정보가 있고, Claude 가 읽은 내용은 Anthropic 으로 전송된다.

- 열지 않는다: `%LOCALAPPDATA%\clinic-scribe\` 의 `records\`(진료 기록·녹음), `voices.json`(목소리 특징값), `diction.json`.
  같은 폴더의 `settings.json` 은 봐도 된다 (씨차트 자리는 가린 모양만 들어 있다).
- 씨차트 화면을 스크린샷, UI Automation 덤프, 창 글자 출력으로 직접 읽지 않는다. 씨차트 구조는 앱의
  `설정 → 씨차트 → 씨차트 연결 점검… → 결과 저장…` 파일(이름을 가린 것)로만 본다.
- `tests/test_emr_probe.py`·`tests/test_emr_link.py` 는 가짜 창으로 돌지만 화면의 모든 창을 훑는다. 씨차트에 실제 환자를 열어 두지 않았을 때 돌린다.
- 개발은 앱이 쓰는 폴더 말고 **따로 받은 폴더**에서 한다. 앱 폴더는 테스트(CI)를 통과한 main 만 자동 업데이트로 받게 둔다.

  ```bat
  git clone https://github.com/jwhwang91/clinic-scribe.git clinic-scribe-dev
  cd clinic-scribe-dev
  py -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
  ```

  개발 폴더에서는 `scripts\setup_windows.bat` 을 돌리지 않는다 (시작 메뉴·바탕 화면 바로 가기가 개발 폴더를 가리키게 바뀐다).
  whisper.cpp 와 모델은 앱을 설치할 때 받은 `%LOCALAPPDATA%\clinic-scribe` 것을 같이 쓴다. push 하려면 이 PC 에서 GitHub 로그인이 필요하다.
- 개발 폴더의 앱을 띄워 볼 때는 원래 앱을 닫고(설정·진료 기록·녹음 버튼 단축키를 같이 쓴다) `CLINIC_SCRIBE_NO_UPDATE=1` 로 연다.
  진료 기록 형식을 바꾸는 작업이면 `CLINIC_SCRIBE_SETTINGS` 를 임시 폴더의 settings.json 으로 줘서 실제 기록과 떼어 놓는다.

## 개발 명령

| | macOS | Windows |
| --- | --- | --- |
| 빠른 테스트 (모델 없이) | `.venv/bin/python -m pytest -m "not models"` | `.venv\Scripts\python.exe -m pytest -m "not models"` |
| 전체 테스트 (whisper.cpp 실제 전사) | `.venv/bin/python scripts/make_test_audio.py` 다음 `.venv/bin/python -m pytest` | `.venv\Scripts\python.exe -m pytest` (가상 음성 `samples/` 는 Mac 에서 만들어 복사, 없으면 그 테스트만 건너뜀) |
| 씨차트 연결·녹음 버튼 | (Windows 전용) | `.venv\Scripts\python.exe -m pytest tests\test_emr_probe.py tests\test_emr_link.py tests\test_hotkey.py` |
| 앱 열기 | `open ~/Applications/clinic-scribe.app` (run.sh 말고) | `scripts\run.bat` |

- 테스트는 설정·진료 기록을 임시 폴더로 돌리고 자동 업데이트와 전역 단축키를 끈다 (`tests/conftest.py`).
- macOS 에는 `timeout` 명령이 없다.

## Git · CI · 배포

- 설치한 컴퓨터(병원 PC)는 `main` 가운데 GitHub Actions `tests` 가 성공한 커밋만 자동 업데이트로 받는다 (`updater.py`).
- 순서: `feat/…` 브랜치에서 커밋 → 사용자가 올리라고 하면 main 에 fast-forward 병합 → push → main 의 `tests`(Windows·Linux 는 실제 전사까지 약 40분)와
  `windows-desktop`(씨차트 연결·녹음 버튼, 몇 분) 결과를 확인해 알린다 → 병합한 브랜치는 지운다 (같은 커밋의 브랜치 CI 가 돌고 있으면 취소).
- 커밋과 push 는 사용자가 요청할 때 한다. 커밋 메시지는 한국어로 `영역: 무엇을 (왜)`.
- **작업을 마칠 때마다 `docs/WORKLOG.md` 의 현재 상태·다음 할 일·작업 기록을 고쳐 같은 커밋에 넣는다.**
  맥과 병원 PC 를 오가며 이어서 작업하므로 이 파일이 인수인계 문서다 (Claude 의 메모리는 컴퓨터마다 따로라 넘어가지 않는다).

## 일하는 방식 (사용자가 바라는 것)

- 계획이나 코드 예시에서 멈추지 않고 파일 생성, 설치, 다운로드, 실행, 검증까지 한다.
  보고할 때는 받은 것 / 확인한 것 / 확인하지 못한 것(과 사용자가 직접 누를 단추·명령)을 나눠 쓴다.
- 사용자가 보는 화면은 처음부터 다듬는다: 어두운 테마, 빈 공간이 적은 촘촘한 배치, 분명한 위계와 카드·상태 표시.
  밋밋하거나 개발자 도구처럼 보이면 덜 된 것으로 본다. 넘기기 전에 실제 화면을 스크린샷으로 확인한다.
- 차트(요약) 형식은 사용자가 실제 차트에 필요한 칸을 알려 주기로 했다. 그 전에는 형식을 크게 바꾸지 않는다.
