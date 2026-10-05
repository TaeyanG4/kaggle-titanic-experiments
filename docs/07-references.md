# 07. 참고 자료와 출처

[프로젝트 홈](../README.md) · [NOTICE](../NOTICE.md)

## 1. 프로젝트 1차 근거

| 자료 | 역할 |
|---|---|
| [Kaggle submission receipts](evidence/kaggle-submissions.csv) | 실제 제출 ID·시각·표시 점수; 종료 때 read-only 조회 |
| [Submission manifest](evidence/submission-manifest.json) | 보존된 예측 CSV의 SHA-256·행 수·양성 예측 수 |
| [Source inventory](evidence/source-inventory.json) | 당시 코드와 가벼운 파생 자료의 지문 |
| [Data fingerprints](evidence/data-fingerprints.json) | 원본 파일 확인용; 데이터 자체는 미포함 |
| [Environment snapshot](evidence/environment-snapshot.json) | 종료 시점 패키지 버전; 과거 모든 실행의 lockfile 아님 |
| [Archived session notes](../archive/session-notes/) | 당시 plan·설명·실패 해석; 과도한 주장은 최종 감사에서 정정 |

## 2. 공개 아이디어·원저자

**Gunes Evitan — Titanic: Advanced Feature Engineering Tutorial.** 분위수 구간, Deck 통합, 가족·티켓 통계 등의 아이디어와 역사적 재현의 출처입니다. 이 프로젝트는 그 방식의 성능을 새로운 독립 결과로 재귀속하지 않습니다.

https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial

**Chris Deotte — Titanic WCG+XGBoost.** 정교한 WCG와 역할별 규칙의 출처입니다. Python 근사와 공개 실행 출력에서 복구한 test 예측 목록이 함께 존재합니다. 공개 목록의 재사용은 자체 학습과 분리하여 설명합니다.

https://www.kaggle.com/code/cdeotte/titanic-wcg-xgboost-0-84688

**Chris Deotte — Titanic Mega Model.** 관련 그룹·도메인 규칙을 조사한 공개 자료입니다. 모든 원본 R 실행을 동일 환경에서 재현했다고 주장하지 않습니다.

https://www.kaggle.com/code/cdeotte/titantic-mega-model-0-84210

**L. D. Freeman — A Data Science Framework to Achieve 99% Accuracy.** 사용자의 초기 토론에서 검토한 자료입니다. 제목의 99%를 이 프로젝트의 점수나 무누출 test 점수로 인용하지 않습니다.

https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

## 3. 공식 문서

- [Kaggle Titanic](https://www.kaggle.com/competitions/titanic): 데이터·평가·규칙의 원 출처.
- [Kaggle CLI](https://github.com/Kaggle/kaggle-cli): 데이터 다운로드와 제출 기록 조회 도구.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): 전처리 불일치, 데이터 누출, randomness를 해석할 때의 기준.
- [scikit-learn nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): 선택과 평가를 분리하는 원리.
- [GitHub repository rename](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository): 저장소 이름 변경과 기존 주소 처리.

## 4. 출처 미확정 부분

공개 노트북 미러의 정확한 commit·사용 라이선스·R 실행 환경은 이 스냅샷에서 모두 고정되지 않았습니다. 따라서 다운로드한 제3자 노트북 전체를 다시 배포하지 않으며, 원저자 링크와 역사적 코드의 출처 주석을 남깁니다. 프로젝트 전체에 일괄 MIT 라이선스를 새로 부여하지 않습니다.

Antigravity 초기 설정과 이후 GPT 웹 세션 중심 작업이라는 설명은 소유자의 진술과 이 대화의 실행 기록을 근거로 합니다. 모든 스킬 버전과 프롬프트 원문을 시간순·불변 형태로 보존한 증거는 없으므로 완전한 재현 패키지라고 부르지 않습니다.
