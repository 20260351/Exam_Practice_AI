examforge-ai/
├── app.py                          # 메인 진입점
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml                 # 다크모드 / 테마
├── data/
│   └── examforge.db                # SQLite (자동 생성)
└── src/
    ├── __init__.py
    ├── config.py                   # 환경설정/상수
    ├── database.py                 # SQLite 모델
    ├── pdf_processor.py            # PDF 텍스트 추출 + 전처리
    ├── ai_engine.py                # OpenAI 호출 (요약/문제생성/리포트)
    ├── quiz_manager.py             # 시험 세션/채점 로직
    ├── analytics.py                # 학습 분석/대시보드
    ├── exporter.py                 # 오답노트 PDF/CSV
    ├── ui_components.py            # 재사용 UI (카드, 진행바, 애니메이션)
    └── pages/
        ├── __init__.py
        ├── home.py
        ├── upload.py
        ├── summary.py
        ├── quiz.py
        ├── result.py
        ├── wrong_notes.py
        └── dashboard.py
