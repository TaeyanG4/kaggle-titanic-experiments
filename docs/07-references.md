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

## 외부 공개 노트북 비교군

아래 표는 2026-10-05에 Kaggle 공개 페이지에서 확인한 표시 점수를 기준으로 만든 비교 메모다. 이 저장소에서 해당 노트북을 동일 환경으로 재실행하거나 숨겨진 test 정답으로 검증한 결과가 아니다. `clean 우선 후보`라는 표현도 저자의 제목과 공개된 입력 범위, 현재까지 확인한 구현 정보에 따른 연구 우선순위이며 독립 인증이 아니다.

| 노트북 | Kaggle 표시 점수 | 현재 판정 | 확인한 범위 |
|---|---:|---|---|
| [Yoni Krichevsky - Top 3% with only 4 features - no data leakage](https://www.kaggle.com/code/yoni2k/top-3-with-only-4-features-no-data-leakage) | Public/Best 0.81818 | clean 우선 후보 | 페이지 제목이 `no data leakage`를 명시하고 input 1 file로 표시된다. strict fold-safe 기준의 전체 코드 독립 감사 전이므로 인증 표현은 쓰지 않는다. |
| [Jonathan Oheix - Titanic survivors prediction - TOP 5%](https://www.kaggle.com/code/jonathanoheix/titanic-survivors-prediction-top-5) | Public/Best 0.82296 | 코드 감사 대기 | 페이지에서 input 1 file과 점수는 확인했다. family/ticket target statistic, train/test 결합 전처리 등 strict 기준의 전체 구현 검토는 아직 끝내지 않았다. |
| [Chris Deotte - Titanic Deep Net [0.82296]](https://www.kaggle.com/code/cdeotte/titanic-deep-net-0-82296) | Best 0.82296 | 코드 감사 대기 | R competition notebook이며 점수는 확인했다. 전체 feature engineering과 평가 경계를 검사하기 전에는 clean으로 분류하지 않는다. |
| [Titanic competition w/ TensorFlow Decision Forests](https://www.kaggle.com/code/gusthema/titanic-competition-w-tensorflow-decision-forests) | Public/Best 0.80143 | 보수적 외부 baseline | Kaggle competition의 `train.csv`, `test.csv`, `gender_submission.csv`가 입력으로 표시되는 pinned notebook이다. 고득점 후보보다 외부 재현 baseline 역할로 기록한다. |

이 표에서 0.81818은 현재 가장 강하게 clean 비교 대상으로 둘 수 있는 공개 점수이고, 0.82296 두 사례는 감사가 끝나면 상한이 바뀔 수 있는 후보로 남긴다. 반대로 이 프로젝트의 v47 0.83014는 공개 예측 재사용과 Public 피드백 기반 선택이 포함되어 있으므로 이 clean 비교선과 같은 열에 놓고 무누출 성능으로 해석하지 않는다.

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
