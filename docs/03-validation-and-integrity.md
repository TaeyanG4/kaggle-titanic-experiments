# 03. 검증·데이터 출처·무결성: 최종 감사와 정정

[프로젝트 홈](../README.md) · [실험 여정](02-experiment-journey.md)

## 요약

**“정답 조회와 의도적인 누출 없이 개선한다”는 사용자의 목표와 “모든 실험이 실제로 무누출이었다”는 인증은 다릅니다.** 이 저장소는 전자를 프로젝트 의도로 기록하고, 후자는 주장하지 않습니다. 실제 코드와 제출 계보에 다음 예외가 확인됩니다.

| 항목 | 확인된 사실 | 보고서에서의 처리 |
|---|---|---|
| v10 관계 통계 | 전체 학습 타깃으로 통계를 만든 뒤 CV 수행 | 원본 OOF는 누출 없는 비교 점수에서 제외 |
| train/test 전처리 | 여러 분기에서 X를 합쳐 통계 생성; 일부는 scaler를 따로 fit | 전이적 처리와 불일치 문제를 명시 |
| Deotte 공개 출력 | PassengerId별 예측 상수를 외부 노트북 출력에서 재사용 | train-only 자체 학습 결과와 분리 |
| 최종 제출 선택 | 점수를 본 뒤 후속 규칙 조합과 champion 선택 | LB-adaptive 결과로 표시 |
| 반복 검증 | 동일 승객을 많은 split·모델에서 재사용 | 독립 증거 수로 합산하지 않음 |
| LLM 사전 지식 | Titanic 공개 자료가 사전학습에 포함됐는지 확인 불가 | 오염이 전혀 없다는 보증을 하지 않음 |

이 감사는 경기 운영자의 규정 위반 판정을 대신하지 않습니다. **데이터 출처와 과학적 주장의 범위를 명확히 하는 문서**입니다.

## 1. 서로 다른 문제를 같은 말로 묶지 않기

숨겨진 test 정답을 직접 사용하는 것, validation 라벨을 피처 계산에 섞는 것, test의 입력 X를 미리 보는 것, leaderboard 점수로 후보를 고르는 것은 서로 다른 문제입니다. 허용 여부와 일반화 해석을 각각 구분해야 합니다.

이번 최종 감사에서 검토한 스크립트는 별도의 숨겨진 test 정답 파일을 읽어 학습하는 흐름을 확인하지 못했습니다. 하지만 이는 전체 세션과 외부 자료의 모든 출처를 완전히 감사했다는 뜻은 아닙니다. 특히 공개 예측 목록 재사용은 명시적으로 존재하므로 **“공식 train의 X·y만 사용했다”는 설명은 최종 artifact에 맞지 않습니다.**

표준적인 inductive 평가에서는 전처리까지 train fold에서 fit하고 validation에서는 transform하는 것이 원칙입니다. 시험 입력 X를 함께 사용하는 transductive 실험은 별도의 가정을 둔 프로토콜이며, 자동으로 숨겨진 타깃 누수와 같은 것은 아니지만 inductive 결과라고 부를 수 없습니다. 관련 기본 원칙은 [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)에 설명되어 있습니다.

## 2. v10: 평가 누출과 전처리 불일치

[reproduce_gunes_original_v10.py](../scripts/reproduce_gunes_original_v10.py)는 스스로 역사적 재현이며 선호하는 누출 방지 파이프라인이 아니라고 명시합니다. `preprocess_original`에서 Family/Ticket 생존 통계를 전체 학습 라벨로 계산하고, 그 결과를 이용해 나중에 5-fold 모델을 학습합니다. 그러므로 validation 라벨이 통계에 반영될 수 있습니다.

같은 파일에는 train과 test에 별도로 `LabelEncoder`, `OneHotEncoder`, `StandardScaler`를 fit하는 동작도 있습니다. 같은 숫자·범주가 서로 다른 좌표로 해석될 수 있는 전처리 불일치이며, 역사적 재현을 이유로 남겨둔 동작입니다. 이것을 권장 구현으로 가져가면 안 됩니다.

v10 Public 점수가 높다는 사실로 이 구현의 평가 문제가 사라지는 것은 아닙니다. 반대로 이 문제가 Public 개선의 유일한 원인이라고도 입증하지 못했습니다. 코드 수준 사실과 성능 원인에 대한 가설을 구분합니다.

## 3. v21과 v47: 공개 예측 재사용의 범위

[deotte_wcg_xgb_v21.py](../scripts/deotte_wcg_xgb_v21.py)에는 `DEOTTE_EXACT_MALE_LIVE`, `DEOTTE_EXACT_FEMALE_PERISH`라는 PassengerId 목록이 있습니다. 코드 주석은 공개 R 노트북 실행 출력에서 복구한 예측이라고 설명합니다. R `rpart`는 로컬에서 동일하게 재실행하지 못해 Python 트리로 근사했습니다.

따라서 두 종류의 결과를 분리해야 합니다.

1. Python 포트가 학습 fold에서 라벨을 읽고 validation에 예측한 OOF.
2. 실제 test에 대해 공개 출력의 예측 목록을 덮어쓴 `exact_public` 파일.

두 번째 파일에 첫 번째 OOF를 그대로 붙여 “같은 모델의 검증 성능”이라고 설명하면 출처가 혼합됩니다. 또한 특정 승객의 친족 관계를 수동 연결한 규칙도 공식 입력만으로 자동 추론한 관계와 구별해야 합니다.

최종 v47은 v10 예측, v38 text 보정, v24 계열의 더 넓은 Deotte 여성 예측을 결합합니다. 외부 예측 재사용과 Public 피드백에 따른 선택을 포함한 역사적 결과로 보존하며, 독립 OOF를 새로 만들어 부여하지 않습니다.

## 4. 반복 CV가 독립 증거를 만드는 것은 아니다

같은 승객이 6개 seed에서 모두 맞았다면 실행 안정성에 대한 정보는 늘지만 새로운 승객 6명을 맞힌 것은 아닙니다. pseudo-test들도 기존 train의 중첩된 부분집합입니다. 따라서 pooled rescue/harm을 독립 Bernoulli 시행처럼 취급해 지나치게 좁은 신뢰구간을 만들면 안 됩니다.

Group-aware 검증 역시 실제 test에 동일 그룹이 어느 정도 남아 있는지에 따라 보수적인 별도 질문을 평가합니다. Group 점수가 낮다는 이유만으로 실제 test에 반드시 약하다고 결론내릴 수 없습니다. 반대로 group 점수가 높다는 이유로 구체적인 418행의 정답을 보장할 수도 없습니다.

## 5. Nested CV와 stacking에 대한 정정

[v29](../scripts/nested_selection_audit_v29.py)는 outer-train 내부에서 inner 후보 선택을 수행합니다. 하지만 후보군 자체는 그 전에 전체 학습 데이터를 보며 발전시킨 것입니다. 새 outer seed가 과거의 연구 선택을 되돌리지는 못합니다. 이 결과로 meta-overfitting이 “확정됐다”거나 그 크기를 정확히 측정했다고 말하는 것은 과합니다.

또한 oracle 진단과 선택된 후보에 서로 다른 model seed를 사용하는 부분이 있으므로, 두 점수의 차이를 순수한 선택 regret으로만 해석해서는 안 됩니다. 저장된 OOF를 입력으로 메타 모델만 cross-fit한 stacking도 base-model 학습 범위까지 분리하는 완전한 nested stack과 다를 수 있습니다. 후속 연구에서는 전체 pipeline 단위 평가가 필요합니다. [공식 nested CV 설명](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)을 참고하세요.

## 6. EB·shift·feature ablation의 추가 한계

EB peer 합에서 자기 행을 제외하더라도 prior와 shrinkage 강도 alpha가 그 행을 포함한 reference 라벨에서 추정되는지 별도로 확인해야 합니다. 이것은 outer validation 라벨의 직접 사용과 같은 문제는 아니지만, training feature의 self-influence와 train/validation 생성 분포 차이를 만듭니다. “exclude_self=True” 하나만으로 전체 처리가 엄격히 cross-fit되었다고 인증하지 않습니다.

v26 domain classifier는 balanced class weight를 사용합니다. 그 raw 확률을 실제 domain prior가 반영된 확률처럼 density-ratio로 바꾸려면 보정·calibration이 필요합니다. AUC 약 0.555도 shift 부재의 증명이 아닙니다. v26은 탐색 진단으로만 보존합니다.

v40~v42 등의 parent와 변형에 서로 다른 모델 seed를 사용하는 코드도 있습니다. 관찰된 차이는 피처 변경과 모델 무작위성의 결합입니다. 동일한 model seed를 맞춘 반복 ablation과 신뢰구간 없이 “P2가 실패의 원인”, “P3가 검증된 원인”으로 단정하면 안 됩니다.

## 7. Headroom gate: 수학적 상한과 성공 확률은 다르다

동일한 N개 이진 분류 평가 행에서 기준 예측과 후보가 d개 다르고, 그중 후보가 맞히는 수가 w라면 순정답 변화는 `2w-d`입니다. 기준보다 목표 제출이 g개 더 맞았다면 목표를 엄격히 넘기 위한 조건은 `2w-d > g`입니다. 이는 전제가 맞을 때의 정확한 산술입니다.

그러나 **OOF disagreement의 평균 precision을 실제 test의 다른 disagreement 집합에 그대로 대입할 수는 없습니다.** v43처럼 수학적으로 초과 가능하지만 관찰된 local precision이 요구값보다 낮은 경우, 실패가 증명된 것은 아닙니다. 당시 제출 보류를 위한 휴리스틱이었을 뿐입니다. 이 보고서는 과거의 “headroom이 없다”를 다음 둘로 나누어 읽습니다.

- `d <= g`: 엄격한 초과가 수학적으로 불가능한 경우.
- `d > g`이지만 요구 precision 근거가 약함: 가능성은 있으나 당시 선택 근거가 부족한 경우.

숫자 24개 변경·17개 성공 필요라는 계산만으로 후보의 실패 확률을 알 수 없습니다. 과거 문서의 이 부분은 이 정정이 우선합니다.

## 8. Leaderboard는 최종 봉인 평가가 아니었다

v38 한 행 변경, v45 추가 한 행 변경, v46 네 행 변경 등의 점수 변화를 확인했습니다. 이런 작은 차이를 이용하면 이진 Accuracy의 산술을 통해 특정 행이나 변경 집합의 성공 여부를 추론할 수 있습니다. **숨겨진 정답 파일을 다운로드하지 않았더라도, 점수 피드백은 정보입니다.**

이후 champion을 유지·변경한 결과는 leaderboard에 적응한 결과입니다. 최종 최고점을 객관적인 단일 독립 test 결과처럼 제시하면 안 됩니다. 후속 실험에서는 제출 횟수·규칙군을 사전에 고정하고, 외부 점수에 따른 행별 수정과 그 결과로 얻은 라벨을 학습·선택에 재투입하지 않아야 합니다.

## 9. 공개 범위와 인증의 한계

원시 Kaggle CSV, 인증 파일, 캐시, 체크포인트, 전체 제3자 다운로드 노트북은 이번 스냅샷에서 제외했습니다. 파생 OOF에는 공식 train 라벨이 포함될 수 있습니다. submissions의 `Survived`는 예측값이며 숨겨진 정답이 아닙니다. [제외 목록](evidence/excluded-files.json)과 [소스 목록](evidence/source-inventory.json)을 공개합니다.

최종 검증 도구는 파일 해시, 스키마, 내부 링크, notebook 출력 제거 등을 검사합니다. 이는 **재현 가능한 자료 관리의 검사**이지 모델의 무누출성이나 통계적 유의성에 대한 자동 인증이 아닙니다.

이 문서의 정정은 이전 세션 기록을 지우기 위한 것이 아닙니다. 시행착오와 과도한 해석을 함께 남기는 것이 이 반자동 실험을 평가하는 데 더 유용하기 때문입니다.
