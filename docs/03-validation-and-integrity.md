# 03. 검증과 한계

[프로젝트 홈](../README.md) / [단계별 기록](02-experiment-journey.md)

처음에는 정답 조회나 데이터 누출 없이 성능을 개선하는 것을 목표로 했다. 하지만 구현과 제출 과정을 확인하면 그 목표와 구분해야 할 부분이 있다. 이 문서는 점수를 숨기거나 과거 코드를 지우기 위한 것이 아니라, 결과를 어디까지 해석할 수 있는지 남기기 위한 기록이다.

| 확인한 항목 | 실제 기록 | 해석 |
|---|---|---|
| v10 관계 통계 | 전체 학습 타깃으로 통계를 만든 뒤 CV 수행 | 원본 OOF를 무누출 비교 점수로 사용하지 않음 |
| 전처리 | train과 test의 입력을 함께 사용하거나 scaler를 따로 fit | 입력 공유와 좌표 불일치 문제를 구분 |
| Deotte 공개 출력 | PassengerId별 예측 목록 재사용 | 자체 학습 예측과 다른 출처로 표시 |
| 최종 선택 | 제출 점수를 본 뒤 후속 후보와 최종 파일 선택 | 독립적인 한 번의 최종 평가가 아님 |
| 반복 검증 | 같은 승객을 여러 split에서 재사용 | 결과 수를 독립 표본 수로 합산하지 않음 |

대회 규정 위반 여부를 판정하는 문서는 아니다. 코드가 사용한 정보와 평가 절차를 설명하는 데 범위를 두었다.

## 타깃 누출과 test 입력 사용은 다른 문제다

숨겨진 test 정답을 학습에 넣는 것, validation 라벨을 피처 계산에 섞는 것, test의 입력 X를 미리 보는 것, 제출 점수로 후보를 고르는 것은 서로 다르다. 한 가지를 하지 않았다고 다른 문제까지 없어지는 것은 아니다.

검토한 스크립트에서는 별도의 숨겨진 test 정답 파일을 읽어 학습하는 흐름을 확인하지 못했다. 그렇다고 세션과 외부 자료의 모든 출처를 완전히 감사했다는 뜻은 아니다. 공개 예측 목록을 사용한 분기는 실제로 존재하므로 최종 artifact를 공식 train의 X와 y만으로 만든 결과라고 설명할 수는 없다.

전처리를 학습 fold에서 fit하고 validation에는 transform만 적용하는 것은 일반적인 inductive 평가의 기본 경계다. test 입력을 함께 사용하는 transductive 실험은 다른 가정을 둔 것이다. 입력 X를 본 사실만으로 숨겨진 타깃을 사용했다고 볼 수는 없지만, 학습 fold만 사용한 평가라고 부를 수도 없다. [scikit-learn의 관련 설명](https://scikit-learn.org/stable/common_pitfalls.html)을 참고할 수 있다.

## v10의 CV와 전처리

[reproduce_gunes_original_v10.py](../scripts/reproduce_gunes_original_v10.py)는 공개 방법의 역사적 재현이다. `preprocess_original`에서 Family와 Ticket 생존 통계를 전체 학습 라벨로 계산하고, 그 결과를 만든 뒤 5-fold 모델을 학습한다. validation 라벨이 관계 피처에 반영될 수 있는 구조다.

train과 test에 `LabelEncoder`, `OneHotEncoder`, `StandardScaler`를 각각 fit하는 동작도 있다. 두 데이터에서 범주나 값의 좌표가 다르게 정의될 수 있으므로 그대로 권장할 구현은 아니다. 당시 원본 동작을 재현하기 위해 남긴 코드이며, 뒤의 감사 분기와 구분한다.

Public 점수가 높아졌다고 이 평가 문제가 사라지는 것은 아니다. 반대로 이 구현 문제가 Public 개선의 유일한 원인이라고 증명한 것도 아니다. 코드에서 확인한 사실과 점수 차이를 설명하는 가설을 나누어 기록했다.

## Deotte 포트와 공개 예측 목록

[deotte_wcg_xgb_v21.py](../scripts/deotte_wcg_xgb_v21.py)에는 `DEOTTE_EXACT_MALE_LIVE`, `DEOTTE_EXACT_FEMALE_PERISH` 상수가 있다. 공개 R 노트북 실행 출력에서 복구한 test 예측 목록이다. R imputation은 같은 환경에서 실행하지 못해 Python 트리로 근사했다.

따라서 Python 포트가 fold별로 학습해 만든 OOF와, 공개 예측 목록을 실제 test에 덮어쓴 `exact_public` 파일을 같은 출처의 결과로 취급하지 않는다. 후자의 파일에 전자의 OOF를 붙여 동일 모델의 검증 성능이라고 설명하면 두 경로를 혼합하게 된다. 특정 승객의 친족 관계를 수동 연결한 규칙도 자동으로 추론한 관계와 구분한다.

최종 v47은 v10, v38 텍스트 보정과 더 넓은 Deotte 여성 예측을 결합한다. 이 파일에는 공개 예측 재사용과 Public 결과를 반영한 선택이 포함되어 있으며, 별도로 평가한 독립 OOF 점수는 없다.

## 같은 데이터를 반복해서 평가한 한계

같은 승객이 여섯 seed에서 모두 맞았다고 새로운 승객 여섯 명을 맞힌 것은 아니다. split 안정성에 대한 정보는 얻을 수 있지만, rescue와 harm을 모두 독립 Bernoulli 시행처럼 합쳐 좁은 신뢰구간을 만들면 과도한 확신이 생긴다. pseudo-test도 기존 train의 겹치는 부분집합이다.

Group-aware 검증은 새로운 관계 집단에 대한 성능을 확인한다. 실제 test에 학습 데이터와 연결된 그룹이 남아 있다면 이 검증은 실제 상황의 일부만 나타낸다. Group 점수가 낮으면 test도 반드시 낮다거나, 높으면 특정 test 승객을 잘 맞힌다고 결론낼 수 없다.

## Nested CV와 stacking

[v29](../scripts/nested_selection_audit_v29.py)는 outer-train 안에서 inner 후보 선택을 수행했다. 하지만 그 후보군은 이미 같은 전체 데이터에서 여러 실험을 거쳐 발전했다. 새 outer seed를 사용해도 이전 연구 선택의 영향까지 없어지지는 않는다.

oracle 진단과 선택된 후보에 서로 다른 model seed를 주는 부분도 있다. 그 점수 차이를 순수한 선택 regret으로만 볼 수 없는 이유다. 기존 OOF 위에서 메타 모델만 cross-fit한 stacking도 base model의 학습 범위까지 분리한 완전한 nested stack과는 다를 수 있다. [nested CV의 기본 설명](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)과 실제 코드의 범위를 함께 봐야 한다.

## EB, 분포 가중치와 피처 비교

EB에서 peer 합의 자기 행을 제외해도 prior와 alpha는 자기 라벨을 포함한 reference로 추정될 수 있다. outer validation 라벨을 직접 사용한 것과 같지는 않지만, 학습 피처의 self-influence와 학습 및 검증 피처의 차이가 남는다. `exclude_self=True` 하나로 전체 전처리가 엄격히 cross-fit되었다고 보장하지 않는다.

v26 domain classifier는 balanced class weight를 사용했다. 그 확률을 density-ratio weight로 바꾸려면 calibration과 class prior를 고려해야 한다. AUC 약 0.555도 분포 변화가 없다는 증명이 아니다. 당시 결과는 탐색 진단으로 보존했다.

v40~v42 등의 parent와 변형은 model seed가 다르기도 하다. 관찰된 차이에는 피처 효과와 무작위성이 함께 포함된다. 같은 model seed를 맞춘 반복 비교 없이 특정 피처가 실패의 원인 또는 개선의 원인이라고 확정하지 않았다.

## Headroom 계산의 범위

같은 N개의 이진 분류 평가 행에서 기준 예측과 후보가 d개 다르고, 그중 후보가 맞힌 수가 w라면 순정답 변화는 `2w - d`다. 목표 제출이 기준보다 g개 더 맞았다면 이를 엄격히 넘으려면 `2w - d > g`여야 한다.

`d <= g`이면 엄격히 넘기는 것은 수학적으로 불가능하다. 반면 `d > g`이지만 필요한 변경 정확도가 높을 때는 가능성은 남아 있다. 당시 사용한 local precision이 낮다는 이유만으로 실패가 증명되는 것은 아니다.

v43의 24개 변경 중 17개를 맞혀야 한다는 계산은 필요한 조건이다. OOF에서 얻은 평균 precision을 실제 test의 다른 불일치 집합에 그대로 넣을 수는 없다. 과거 기록의 headroom 기각은 이 둘을 구분해서 읽어야 한다.

## Public 피드백도 선택에 사용한 정보였다

v38 한 행, v45 추가 한 행, v46 네 행처럼 작은 변경의 점수를 확인했다. 이진 Accuracy에서는 이런 점수 차이로 변경 집합의 성공 여부에 관한 정보를 얻을 수 있다. 별도의 정답 파일을 내려받지 않았더라도 제출 피드백 자체가 정보라는 점은 남는다.

이후 후보를 고르고 유지한 과정은 leaderboard에 적응한 결과다. 최종 최고점을 처음부터 봉인된 단일 test 결과처럼 설명하지 않는다. 후속 실험에서는 제출 횟수와 후보군을 사전에 정하고, 점수에서 추론한 라벨을 학습이나 행별 규칙 선택에 다시 사용하지 않아야 한다.

## 공개 범위

원본 Kaggle CSV, 인증 파일, 캐시, 체크포인트와 내려받은 제3자 노트북 전체는 새 스냅샷에서 제외했다. 파생 OOF에는 공식 train 라벨이 포함될 수 있다. submissions의 Survived 열은 예측이며 숨겨진 정답 데이터가 아니다. [제외 목록](evidence/excluded-files.json)과 [소스 목록](evidence/source-inventory.json)을 남겼다.

LLM의 사전학습에 어떤 Titanic 자료가 포함됐는지는 이 실험에서 확인할 수 없다. 파일 검증 도구도 해시, 스키마, 문서 링크와 그림을 확인할 뿐, 모델의 무누출성이나 통계적 유의성을 인증하지 않는다.
