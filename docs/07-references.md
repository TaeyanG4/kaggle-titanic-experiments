# 07. 참고 자료

[프로젝트 홈](../README.md) / [재사용 안내](../NOTICE.md)

## 실험 기록

| 자료 | 내용 |
|---|---|
| [Kaggle 제출 기록](evidence/kaggle-submissions.csv) | 실제 제출 ID, 시각과 Public 점수 |
| [제출 파일 목록](evidence/submission-manifest.json) | 예측 CSV의 해시, 행 수와 양성 예측 수 |
| [소스 목록](evidence/source-inventory.json) | 보존한 코드와 파생 자료의 지문 |
| [데이터 지문](evidence/data-fingerprints.json) | 원본 파일 확인용 해시 |
| [환경 스냅샷](evidence/environment-snapshot.json) | 종료 시점 라이브러리 버전 |
| [작업 노트](../archive/session-notes/) | 당시 계획과 결과 해석 |

## 공개 방법

### Gunes Evitan의 Advanced Feature Engineering Tutorial

분위수 구간화, Deck 통합, 가족과 티켓 통계의 출처다. v10은 이 방법의 역사적 동작을 재현했고, 다른 분기에서는 피처를 fold별로 재작성해 비교했다.

[원본 노트북](https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial)

### Chris Deotte의 WCG + XGBoost

정교한 여성과 아동 그룹, 역할별 예측 규칙을 참고했다. Python으로 근사한 코드와 공개 실행 출력에서 가져온 test 예측이 함께 존재하며 두 출처를 구분해 기록했다.

[원본 노트북](https://www.kaggle.com/code/cdeotte/titanic-wcg-xgboost-0-84688)

### Chris Deotte의 Mega Model

관련 그룹과 도메인 규칙을 조사할 때 참고했다. 원본 R 환경을 동일하게 재실행한 완전 재현이라고 설명하지 않는다.

[원본 노트북](https://www.kaggle.com/code/cdeotte/titantic-mega-model-0-84210)

### L. D. Freeman의 Data Science Framework

초기 대화에서 고득점 방식과 과적합 문제를 검토할 때 참고한 자료다. 제목의 99%를 이 프로젝트나 무누출 test 성능의 수치로 사용하지 않았다.

[원본 노트북](https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)

## 도구와 검증 문서

| 자료 | 용도 |
|---|---|
| [Kaggle Titanic](https://www.kaggle.com/competitions/titanic) | 공식 데이터, 평가와 대회 규칙 |
| [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) | 다운로드, 제출과 결과 확인 |
| [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | 전처리 불일치, 누출과 randomness |
| [scikit-learn nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) | 모델 선택과 평가 분리 |
| [Viz.js API](https://viz-js.com/api/) | DOT 소스를 SVG로 렌더링 |
| [sharp](https://sharp.pixelplumbing.com/) | 흐름도 PNG 생성 |

## 기록하지 못한 부분

공개 노트북 미러의 정확한 commit, 실행 환경과 재사용 조건을 모든 자료에서 고정하지 못했다. 다운로드한 제3자 노트북 전체를 다시 배포하지 않고 원저자 링크와 코드의 출처 주석을 남긴 이유다. 프로젝트 전체에 일괄적인 MIT 라이선스를 새로 부여하지 않았다.

초기 설정은 Antigravity에서, 이후 작업은 GPT 웹 세션에서 진행했다. 모든 스킬 버전과 프롬프트를 시간순으로 완전히 보존한 것은 아니므로 같은 대화 과정을 처음부터 그대로 재현하는 패키지는 아니다.
