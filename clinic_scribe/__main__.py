from clinic_scribe import updater

# 받아 둔 새 버전이 있으면 적용한다 (화면 라이브러리와 앱 코드를 불러오기 전에. 실패하면 지금 버전으로 연다)
updater.apply_on_start()

from clinic_scribe.app import main  # noqa: E402

raise SystemExit(main())
