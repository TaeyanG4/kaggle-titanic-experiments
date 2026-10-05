> HISTORICAL SESSION NOTES. Claims, scores and stopping decisions below reflect that stage, not a leakage-free certification. See ../../docs/03-validation-and-integrity.md for the final retrospective. Local-machine links may no longer resolve.

# 💡 Discoveries & Lessons Learned

> **Kaggle Titanic - Machine Learning from Disaster**  
> 이 문서는 버전별 실험(v1 ~ v6)을 거치며 발견한 핵심 인사이트와 머신러닝/데이터 사이언스 원리를 기록합니다.

---

## 🔍 1. 실패 분석 및 근본 원인 (Root Cause Analysis)

### ① 타깃 결측치(NaN Target) 누수로 인한 0점(채점 실패) 발생
* **원인**:
  기존 `Kaggle_Titanic_2023_01_09_0_0_2.ipynb`에서는 `df = pd.concat([data, test_data])`로 전체 1,309건을 합친 후 `train_test_split(df)`을 호출했습니다.
  * 테스트 데이터 418건의 `Survived`는 `NaN`입니다.
  * 이로 인해 훈련 세트에 330개, 검증 세트에 88개의 `NaN` 타깃이 유입되어 모델이 비정상적으로 학습/예측되었습니다.
  * 결과적으로 캐글 제출 시 **`0.00000` 점수를 4회나 기록**하는 원인이 되었습니다.
* **해결책**:
  피처 생성 단계에서만 전체 데이터를 활용하고, 모델링 직전에는 반드시 `Survived.notnull()`인 891건만 훈련 세트로 엄격히 분리하여 **5-Fold Stratified K-Fold CV**를 수행해야 합니다.

---

### ② WCG (Woman-Child-Group) 생존율의 정수 캐스팅 함정
* **원인**:
  기존 코드에 `df.WomanOrBoySurvived = df.WomanOrBoySurvived.astype(int)`라는 코드가 있었습니다.
  * 가족 중 1명이 생존하고 1명이 테스트셋에 있는 등 생존율이 0.5인 경우, `int`로 변환되면 소수점이 버려져 **`0`으로 강제 변환**되었습니다.
  * 살 수 있었던 가족의 생존 신호가 전부 사망으로 둔갑하여 가장 중요한 특성이 왜곡되었습니다.
* **해결책**:
  * **Leave-One-Out (자기 자신 제외)**: 그룹 내에서 나를 뺀 나머지 동행자들의 생존 여부만 집계.
  * 동행자 전원 생존 시 `1.0`, 전원 사망 시 `0.0`, 정보가 없거나 1인 승객은 `0.5`(중립)로 보존.

### ③ WCG `GroupSurvival`의 CV Fold 경계 누수
* **발견 경로**:
  * v1 `scripts/generate_notebook.py`와 v2 `scripts/generate_notebook_v2.py`를 감사한 결과, 두 버전 모두 5-Fold 분할 **이전**에 전체 891개 train의 `Survived`를 사용해 `GroupSurvival`을 계산합니다.
* **왜 문제인가?**:
  * Leave-One-Out은 자기 자신의 라벨만 제외합니다. 따라서 validation fold의 승객이 같은 validation fold 안의 가족/티켓 동행자 라벨을 피처로 참조할 수 있습니다.
  * 즉, self leakage는 막았지만 **fold-to-fold target leakage**는 남아 있습니다.
  * 이 때문에 v1 OOF `0.8507 / AUC 0.8715`, v2 OOF `0.8440 / AUC 0.8980`은 재검증 전까지 낙관 편향 가능성이 있는 참고치로 취급해야 합니다.
* **대응**:
  * `scripts/audit_group_survival.py`를 추가해 동일 fold에서 `no_wcg / global_loo / fold_safe`를 직접 비교하도록 했습니다.
  * `fold_safe`에서는 validation/test의 그룹 생존 신호가 **해당 fold-train의 라벨만** 참조합니다.
  * v2 audit 실행 결과:
    * `no_wcg` Ensemble: Accuracy `0.83389`, ROC-AUC `0.88378`
    * `fold_safe` Ensemble: Accuracy **`0.84175`**, ROC-AUC **`0.89730`**
    * `global_loo` Ensemble: Accuracy `0.84287`, ROC-AUC `0.89870`
  * **해석**: fold-safe WCG는 WCG 제거 대비 Accuracy `+0.00786`, AUC `+0.01352`로 실제 신호가 분명합니다. 반면 global LOO의 낙관 편향은 fold-safe 대비 Accuracy `+0.00112`, AUC `+0.00140` 수준으로 작았습니다.
  * 결론적으로 WCG는 유지하되, 이후 모든 OOF/모델 비교는 **fold-safe WCG**를 기준으로 합니다.

---

## 🚀 2. 점수 향상을 이끈 성공 요인과 머신러닝 원리 (What Worked, and Why)

### 1. WCG + Ticket 동행 그룹 피처 (도메인 특화 피처)
* **원리**:
  타이타닉 침몰 당시의 대원칙은 **"여성과 어린이 먼저(Women and Children First)"**였습니다.
  그러나 3등석 대가족(Rice, Palsson, Goodwin 등)은 가족이 함께 움직이다가 전원 탈출하지 못해 여성과 아동임에도 비극적으로 전원 사망했습니다.
  * 단순히 성별/나이만 보는 것이 아니라, **"내 가족/동행자가 살았는가, 죽었는가"**가 개인의 생존 여부를 결정짓는 가장 강력한 신호입니다.
  * 성씨(`LastName`)뿐만 아니라 동일한 티켓 번호(`Ticket`)를 가진 사람들도 같은 일행이므로, 티켓 기반 그룹핑을 함께 적용했을 때 결정적인 판별력을 가집니다.

### 2. 1인당 실제 요금 (`FarePerPerson = Fare / TicketFreq`)
* **원리**:
  타이타닉 원본 데이터의 `Fare`는 **"티켓 1장당 가격"**입니다.
  3등석 대가족의 경우 가족 5명의 티켓 요금이 한꺼번에 30파운드로 찍혀 있어, 모델이 1등석 요금으로 오인하는 문제가 발생합니다.
  * 요금을 티켓 빈도수로 나눈 실제 1인당 요금을 계산하여 로그 변환(`np.log1p`)함으로써, 3등석 대가족의 요금 착시를 교정하고 Pclass와의 다중공선성을 올바르게 정렬했습니다.

### 3. 왜도(Skewness)와 Log1p 변환의 수학적 원리
* 요금 데이터는 대부분 10~30달러에 집중되어 있고 소수가 500달러를 넘는 극단적 오른쪽 꼬리 분포(Right-skewed)를 가집니다.
* $y = \log(1 + x)$ 변환을 적용하면, $0$달러 승객의 $\log(0)$ 무한대 에러를 방지하면서 수치 범위를 정규분포에 가깝게 압축하여 트리 모델의 분기점 탐색 효율을 극대화합니다.

### 4. 6대 트리 모델의 다양성과 Weighted Blending
* **Bagging (Random Forest, Extra Trees)**: 데이터와 특성을 무작위 샘플링하여 트리 간 상관성을 낮추고 분산(Variance)을 줄임.
* **Boosting (Gradient Boosting, XGBoost, LightGBM, CatBoost)**: 이전 트리가 틀린 잔차를 순차적으로 집중 학습하여 편향(Bias)을 줄임.
* v2에서는 OOF ROC-AUC가 가장 뛰어난 **CatBoost(25%), GradientBoosting(20%), XGBoost(20%)**에 더 높은 가중치를 부여하는 **Weighted Soft Voting**을 적용하여 OOF ROC-AUC를 **`0.8980`**까지 끌어올렸습니다.

---

## ⚖️ 3. 리더보드 진동(Shake-up)과 Public Score 해석의 원리

* **현상**:
  * v1 (Soft Voting): CV Acc `0.8507`, CV AUC `0.8715` ➡️ Public Score `0.79186`
  * v2 (Weighted Blending): CV Acc `0.8440`, CV AUC `0.8980` ➡️ Public Score `0.78708`
* **통계적 해석 (Statistical Explanation)**:
  1. 현재 Titanic leaderboard는 **418개 test row 전체**를 사용해 Accuracy를 계산합니다.
  2. 따라서 단 **1명의 예측이 바뀌면 점수가 $\frac{1}{418} \approx 0.00239$ 변동**합니다.
  3. v1(`0.79186`)과 v2(`0.78708`)의 차이는 현재 418-row 기준으로 약 2명 수준의 순정답 차이에 해당합니다.
  4. Titanic은 rolling leaderboard이므로 오래된 submission의 노출/순위는 바뀔 수 있지만, 동일한 418개 예측 파일 자체의 Accuracy가 주기적으로 바뀌는 구조는 아닙니다.
  5. 다만 현재 v2의 `GroupSurvival`에서 fold 경계 누수가 확인되었으므로, **OOF ROC-AUC `0.8980`은 fold-safe 재측정 전까지 참고치로만 취급**합니다. Public Score보다 CV를 우선하되, 그 CV 자체가 누수 없는지 먼저 검증하는 것이 핵심입니다.

---

## 🧪 4. v4 Model Zoo에서 새로 확인한 것

### 1. 강한 단독 모델과 강한 제출 후보는 다를 수 있다
* TabICLv2 단독 fold-safe OOF: Accuracy **0.85073**, ROC-AUC **0.90086**.
* 기존 v1 fold-safe: Accuracy **0.84848**, ROC-AUC **0.88914**.
* 로컬에서는 TabICLv2가 우세했지만 Public Score는 **0.78708**로 v1의 0.79186보다 낮았습니다.
* 작은 train set에서는 OOF 우위가 public subset에서 그대로 재현되지 않을 수 있으므로 champion을 즉시 교체하지 말고 제출 증거와 함께 판단해야 합니다.

### 2. 작은 가중치의 보수적 ensemble이 효과적이었다
* v1 probability 90% + TabICLv2 10%는 OOF Accuracy **0.84961**, ROC-AUC **0.89004**.
* fold Accuracy는 v1 대비 4개 fold에서 동일, 1개 fold에서 개선되었습니다.
* test 418명 중 v1과 **단 1명만 예측이 달랐고**, Public Score는 **0.79425**로 새로운 최고 기록을 만들었습니다.
* 교훈: 작은 데이터에서는 새 모델로 완전히 교체하기보다 검증된 champion에 새로운 inductive bias를 소량 혼합하는 방식이 더 안정적일 수 있습니다.

### 3. 다양성은 낮은 correlation만으로 판단하면 안 된다
* GaussianNB/QDA는 v1과 correlation이 낮았지만 단독 정확도가 크게 낮아 ensemble 재료로 부적절했습니다.
* TabICLv2는 v1과 probability correlation이 약 **0.986**으로 높았지만, v1이 틀린 14명을 맞히고 반대로 12명을 잃어 순이익 +2를 만들었습니다.
* 따라서 ensemble 후보 평가는 **개별 성능 + 오류 상보성 + fold 안정성**을 함께 봐야 합니다.

---

## 🧬 5. v5 Aggressive Model Zoo에서 새로 확인한 것

### 1. Neural model의 최고점은 seed 검증이 필수다
* MLP-PLR seed42는 OOF Accuracy **0.85297**로 단일 모델 최고 기록을 만들었습니다.
* 그러나 동일 fold에서 seed142는 **0.84063**, seed242는 **0.84287**로 내려갔고 3-seed 확률 평균도 **0.84400**이었습니다.
* 따라서 한 seed의 최고점만 보고 champion으로 승격하면 작은 Titanic 데이터에서 seed noise에 과적합할 수 있습니다.

### 2. 서로 다른 family의 Hard Vote가 seed noise를 흡수했다
* 구성: **v4b champion + RuleFit + MLP-PLR**.
* seed42 MLP 사용 시 OOF Accuracy **0.85522**.
* MLP seed를 142/242로 바꿔도 hard vote는 각각 **0.85410 / 0.85073**을 유지했습니다.
* 3개 MLP seed 평균 확률을 하나의 투표자로 쓴 robust vote는 **0.85410**.
* 즉 MLP 자체는 흔들려도 Tree/Foundation 기반 기존 champion + Rule model과의 다수결이 variance를 줄였습니다.

### 3. 더 많은 모델을 넣는다고 더 좋아지지는 않았다
* v4b + RuleFit + MLP 3개 seed의 5-vote는 Accuracy **0.85297**.
* TabPFN v2 / TabR까지 포함한 5-vote와 7-vote는 **0.84961**로 하락했습니다.
* 좋은 ensemble은 모델 수가 아니라 **강한 개별 모델 + 상보적 오류 + 낮은 자유도**가 중요합니다.

### 4. v5 주요 단일 모델 결과
* MLP-PLR: **0.85297 / AUC 0.88595**
* RuleFit: **0.84961 / AUC 0.89598**
* TabPFN v2: **0.84287 / AUC 0.89663**
* TabR: **0.84175 / AUC 0.88721**
* RealMLP: **0.84063 / AUC 0.87819**
* FT-Transformer: **0.83838 / AUC 0.88801**
* TabPFN v3 / v2.5는 코드 실패가 아니라 **Prior Labs 1회 라이선스 승인/인증 대기** 상태입니다.

### 5. Public 검증에서도 robust vote가 개선됐다
* v5 seed-robust hard vote 제출 ID **56827698**.
* Public Score **0.79665**로 v4b의 **0.79425**를 넘어 새 champion이 되었습니다.
* 로컬 OOF Accuracy는 **0.85410**으로 seed42 hard vote의 0.85522보다 조금 낮았지만, seed 안정성을 우선한 보수적 선택이 실제 Public에서도 개선으로 이어졌습니다.
* 다음 단계에서도 한 번의 최고 OOF보다 **seed/fold robustness + 새로운 오류 상보성**을 승격 기준으로 유지합니다.

---

## 🧰 6. v6 Public Feature Engineering Audit

### 1. 공개 feature를 많이 넣는 것이 답은 아니었다
* Cabin/missingness, relational count, Sex×Pclass/Title×Pclass, Age/Fare band 등을 block으로 추가했지만 대부분 Accuracy가 유지되거나 하락했습니다.
* 구조 feature 전체를 한 번에 넣은 panel은 baseline **0.84287**에서 **0.83277**까지 떨어졌습니다.
* Titanic 891행에서는 feature accumulation보다 작은 가설 단위의 ablation이 더 중요합니다.

### 2. 단일 feature 중 유효했던 것
* `FarePerFamily`: panel Accuracy **+0.00224**, AUC **+0.00120**.
* `AdultMale`: panel Accuracy **+0.00224**; CatBoost에서는 **0.84512 → 0.85297**.
* `TicketWCGDetail`: fixed threshold Accuracy는 개선되지 않았지만 CatBoost AUC **+0.00568**.

### 3. 공개 Family Survival의 LastName + Fare 아이디어가 실제로 전이됐다
* `LastName + Fare`를 가족 키로 만들고 peer survival을 fold-safe하게 생성.
* 3-model panel: **0.84287 / AUC 0.89786 → 0.84624 / 0.90356**.
* GradientBoosting: **0.84961 / AUC 0.90125**.
* 즉 이 표현은 실제 신호지만 현재 v5 robust hard vote Accuracy **0.85410**보다 낮아 단독 승격은 하지 않음.

### 4. AUC 상승과 Accuracy 상승은 다시 분리됐다
* FamilyFare feature를 RuleFit/MLP에 넣으면 각각 AUC가 약 **+0.00583 / +0.00644** 올랐지만 Accuracy는 **-0.00786 / -0.00898** 하락.
* cross-fitted logistic stacking에서도 AUC는 0.90대까지 올라가지만 Accuracy는 champion을 넘지 못함.
* 따라서 다음 단계는 threshold 최적화 반복이 아니라 **새 모델 family 또는 더 좋은 group representation**을 우선함.

---

## 🔬 7. v7 Preprocessing / HPO Audit

### 1. CatBoost native categorical은 이번 데이터에서 우위가 없었다
* Title/Deck/Ticket/LastName/FamilyFare 등을 raw categorical로 직접 넣어 비교했지만 최고 Accuracy는 **0.84512**.
* 기존 one-hot + numeric representation보다 약해 이 branch는 종료.

### 2. 관계형 결측치 보간이 더 그럴듯해도 metric이 좋아지지는 않는다
* 결측 Age 263명 중 62명은 가족/티켓의 관측 나이로 보간 가능했고, Cabin 결측 중 39명은 관계 기반 Deck 전파가 가능했습니다.
* 하지만 3-model panel 최고 Accuracy는 **0.84512**에 그쳤습니다.

### 3. 작은 HPO는 representation 병목을 해결하지 못했다
* AdultMale CatBoost의 안정적인 `slow_d3_l2_6` 설정은 trusted fold **0.85073**, alternate fold seeds 123/777 모두 **0.84624**.
* RuleFit `no_linear`는 seed42 단독 **0.85073**까지 올랐지만 v5 robust vote는 **0.85410**으로 변화 없음.
* 추가 tiny HPO의 정보가치는 낮다고 판단.

### 4. Subgroup rule은 fold 반복성이 없으면 승격하지 않는다
* Large family에서 FamilyFareGB가 champion 대비 +2 correct였지만 두 개선 모두 fold 4에서만 발생.
* 전체 OOF gain만 보고 rule을 만들었다면 validation overfit 위험이 컸습니다.

---

## 🔬 8. v10-v12: Public-score 재현과 Trusted Validation 분리

### 1. Gunes 원본 Public 강점은 fold-safe로는 재현되지 않는다
* v10 original-style Public **0.81578**.
* 같은 26-feature 구조를 유지하고 target encoding만 fold-safe로 바꾸면 RF Accuracy가 **0.83502 / 0.82941**.
* historical global Family/Ticket survival-rate와 test-aware group selection이 Public 성능의 중요한 부분임을 확인.

### 2. TabPFN v2.5/v3는 FamilyFare와 Gunes-rate를 실제로 활용한다
* v2.5/v3 baseline보다 FamilyFare 또는 Gunes-rate block이 Accuracy를 **0.84848**까지 올림.
* feature를 전부 합치는 것은 오히려 악화. TabPFN에서도 feature accumulation보다 block selection이 중요.

### 3. Diversity는 생겼지만 trusted ensemble gain은 아직 없다
* v3 + FamilyFare + Gunes all은 v5 오류 14명을 rescue하지만 20명을 harm.
* `RuleFit + MLP + TabPFN v3 FamilyFare`는 v5 OOF와 정확히 동률이고 test를 1명만 변경.
* 따라서 Public probing보다 추가 causal evidence가 먼저.

---

## 🔬 9. v13: v10 Test-Aware Signal과 Conflict Selector

### 1. v10의 핵심은 현재 test에 실제로 재등장하는 그룹이다
* Family/Ticket이 train과 test 양쪽에 있을 때만 train survival median을 test feature로 전달.
* test label은 사용하지 않지만 test의 entity membership을 이용하므로 transductive preprocessing.
* SurvivalRateNA=1은 family와 ticket 두 신호 모두 있음, 0.5는 한쪽만, 0은 없음.

### 2. v10-v5 충돌 22명 중 절반은 양쪽 관계 신호를 모두 가진다
* NA=1: **11명**, NA=0.5: **3명**, NA=0: **8명**.
* PassengerId 1017(Cribb)은 family/ticket train peer가 모두 사망하여 rate 0; v10/TabPFN은 0, v5는 1.
* PassengerId 1094(Astor)는 family peer 1/1, ticket peer 3/4 생존으로 median rate 1; v10은 1, trusted models는 0.

### 3. fold-safe transductive analogue도 Public을 설명하지 못한다
* fold-train/fold-validation entity eligibility + fold-train label-only rate로 RF를 재검증했지만 Accuracy **0.83277**.
* v5와 다른 43 OOF rows에서는 v5가 31개, analogue RF가 12개 correct.
* ordinary train-fold relation structure가 실제 test relation structure를 잘 모사하지 못함.

### 4. 가장 낮은 자유도의 후보는 2-row guard
* v10을 기본값으로 유지.
* 관계 신호 없음 + v5와 TabPFN v3 FF가 v10 반대에 동의할 때만 변경.
* 실제 test 변경: PassengerId **929, 1231**.
* 이 후보는 Public-LB 가설일 뿐 trusted champion 승격 대상은 아님.

---

## 🔬 10. v14-v20: Pseudo-test Validation과 Typed Relational Stress Test

### 1. 실제 test 관계구조를 모사한 pseudo-test를 만들었다
* split 선택에는 `Survived`를 사용하지 않고 실제 test의 Family/Ticket 연결률, 동시 연결률, peer 수, Sex/Pclass/child/Age-missing 비율을 맞췄습니다.
* 실제 test target relation geometry: Family 연결 약 **0.3397**, Ticket 연결 약 **0.3636**, 둘 다 연결 약 **0.2823**.
* v10 원본과 동일하게 Family eligibility를 **reference-train family-size median > 1**로 정의한 뒤 pseudo split을 재생성했습니다.

### 2. typed22 relational feature는 실제 신호지만 효과 크기는 초기 관찰보다 작았다
* corrected pseudo-test 5개 평균:
  * `legacy2`: Accuracy **0.84270**, AUC **0.88835**
  * `typed22`: Accuracy **0.85618**, AUC **0.90260**
* `typed22 - legacy2`: 평균 Accuracy **+0.01348**, AUC **+0.01424**.
* Accuracy는 5개 중 3개 split에서 개선, AUC는 5/5 개선.
* trusted fixed folds에서도 `legacy2 0.83277 → typed22 0.84287`로 **+0.01010**.

### 3. FamilyFare + typed22 CatBoost/TabPFN은 강했지만 v5를 단독으로 이기지는 못했다
* FamilyFare + typed22 CatBoost: fixed OOF Accuracy **0.85297**.
* alpha/role/graph variants를 추가로 스크린했으나 단일 모델은 v5 robust **0.85410**을 안정적으로 넘지 못했습니다.
* graph는 AUC를 올리는 경우가 있었지만 Accuracy 이득은 재현되지 않아 주력 branch에서 제외했습니다.

### 4. typed consensus는 fixed OOF와 pseudo-test에서 모두 로컬 개선을 만들었다
* `v5 -> TabPFN v3 + v2.5 + CatBoost unanimous switch`:
  * fixed OOF **0.85859**, v5 대비 **+0.00449**, fold deltas `[0,+1,0,+2,+1]`.
  * v19에서 v5와 typed members를 pseudo split마다 처음부터 재학습한 결과 평균 **+0.00225**, 5개 중 4개 non-negative.
* 더 느슨한 `TabPFN v3 + CatBoost` pair switch는 pseudo-test 평균 **+0.00449**, 5/5 non-negative였지만 fixed OOF 개선은 더 작았습니다.

### 5. 그러나 실제 Public은 개선되지 않았다 — pseudo-test도 완전한 test proxy는 아니다
* strict typed consensus 제출 ID **56831596** → Public **0.79425**.
* v10 기반 conservative `all3_weak` guard는:
  * fixed transductive analogue: **+0.00673**, 5/5 fold non-negative.
  * pseudo-test: 평균 **+0.00562**, 5/5 non-negative.
  * 실제 v10 test prediction 418개 중 **6개만 변경**.
  * 제출 ID **56831621** → Public **0.81100**, v10 champion **0.81578**보다 낮음.
* 결론: **관계구조를 맞춘 pseudo-test는 ordinary CV보다 진단 가치가 높지만, 현재 Titanic actual test를 충분히 설명하는 promotion surface는 아니다.**
* 두 독립 local winner가 모두 Public에서 하락했으므로 추가 row-level/LB probing은 중단합니다. Public champion은 계속 v10 `0.81578`, trusted clean champion은 v5 OOF `0.85410`으로 분리 유지합니다.

---

## 🔬 11. v21-v25: Deotte Exact, Group-Aware CV, Limited HPO, Final Majority

### 1. Deotte WCG의 정확한 그룹 정의는 기존 Family/Ticket 표현보다 더 정밀했다
* 공개 R notebook의 핵심 `GroupId`는 `Surname + Pclass + 마지막 글자를 X로 가린 Ticket + Fare + Embarked`입니다.
* adult male은 `noGroup`으로 두고, singleton WCG를 제거한 뒤 같은 `TicketId`의 nanny/relative를 WCG에 연결합니다.
* PassengerId 893(Wilkes)을 775(Hocking) 계열에 수동 연결하는 공개 구조 규칙도 원본과 동일하게 반영했습니다.
* 공개 notebook 실행 출력에서 WCG는 **80개 group, nanny/relative 9명, test boy-survive 8명, female-die 14명**을 보고하며, Python 포트도 동일한 8/14 WCG 변경을 재현했습니다.

### 2. R XGBoost의 exact test decisions도 공개 실행 출력에서 복구했다
* adult male survive: `926, 942, 1094, 1215`.
* solo female perish: `928, 990, 1030, 1061, 1091, 1098, 1160, 1172, 1205, 1304`.
* R 런타임이 로컬에 없어 `rpart`/R-xgboost binary-level exact fit은 불가능했지만, 공개 실행 출력의 최종 test decisions를 이용해 `submission_v21_deotte_wcg_xgb_exact_public.csv`를 생성했습니다.
* 이 과정은 test label을 사용하지 않았으며 historical public notebook prediction의 재현입니다.

### 3. Deotte는 ordinary fixed CV보다 group-aware stress에서 상대적으로 강했다
* strict fixed fold Python port `WCG+male/female XGB`: Accuracy **0.85410**, v5 trusted와 동률.
* 4개 Stratified split seed의 20-fold 평균: **0.84623**로 fixed score보다 낮아 fixed-only champion 승격은 기각.
* Ticket/Family 연결요소를 통째로 holdout한 group-aware stress 2 seeds 평균:
  * Deotte **0.82604**
  * v5 robust **0.81033**
  * typed RF6 **0.79461**
  * typed CatBoost **0.79349**
* 관계 그룹이 validation으로 완전히 빠질 때에는 Deotte의 hand-crafted domain rule이 learned relational feature보다 안정적이었습니다.

### 4. 제한 HPO는 더 이상 유효한 개선축이 아니었다
* FamilyFare+typed22에서 XGBoost 12, LightGBM 10, RandomForest 8, ExtraTrees 8개의 작은 search를 수행했습니다.
* RandomForest config2는 fixed OOF **0.85522**를 기록했지만 alternate seeds에서 **0.84400 / 0.84287**로 하락했습니다.
* alternate까지 포함한 최고 평균은 RF6 **0.84998**에 그쳐 v5 trusted를 넘지 못했습니다.
* 결론: Titanic 891행에서 추가 HPO는 representation 개선보다 fixed-fold selection overfit을 만들 가능성이 더 큽니다.

### 5. 최종 low-DOF majority는 모든 local validation을 통과했지만 Public에서 실패했다
* 최종 조합: **v5 robust + Deotte exact WCG/XGB + typed RF6**, 각 모델 binary prediction의 2-of-3 majority.
* fixed OOF: **0.85410 → 0.85746 (+0.00337)**.
* group-aware: **0.81033 → 0.81930 / 0.82155**.
* matched pseudo-test 5개: **4개 개선 + 1개 동률, 악화 0개**, 평균 `0.84944 → 0.86067`, **+0.01124**.
* 제출 ID **56832161** → Public **0.79186**.
* 즉 fixed / repeated-style stress / group-aware / test-like pseudo를 모두 통과해도 현재 418-row test 성능을 보장하지 못했습니다.

### 6. 현재의 최종 운영 결론
* **Public-score champion은 계속 v10 `0.81578`**입니다.
* **trusted fold-safe champion은 계속 v5 OOF `0.85410` / Public `0.79665`**입니다.
* `submission.csv`는 v10 champion과 동일하게 보존합니다.
* v19, v20, v25 세 번의 독립적인 local promotion winner가 모두 Public champion을 넘지 못했으므로, validation을 다시 Public에 맞추는 작업은 과적합 위험이 너무 큽니다.
* 남은 TF-DF exact / WCG+KNN exact 같은 항목은 completeness 성격이며, 이미 검증한 tree/KNN/WCG family에 비해 새로운 독립 signal 기대값이 낮아 score-improvement campaign의 주력 탐색은 종료합니다.

---

## 🔬 12. v26-v32: Validation Reset, Partial Pooling, Nested Audit

### 1. Covariate shift는 주원인이 아니었다
* v26 train-vs-test adversarial classifier의 최고 OOF AUC는 약 **0.555**였습니다.
* LightGBM 계열은 거의 0.52 수준이었고, PassengerId는 의도적으로 제외했습니다.
* density-ratio weighted OOF에서도 실제 Public ranking을 설명하지 못했습니다.
* 결론: observable X-distribution shift만으로 `CV에서는 v5/v25, Public에서는 v10` 현상을 설명하기 어렵습니다.

### 2. Empirical-Bayes partial pooling은 실제 독립 signal이었다
* sparse Family/Ticket/FamilyFare 생존률을 `WomanChild/AdultMale × Pclass` prior 쪽으로 shrink했습니다.
* 4-seed mean Accuracy:
  * typed22 **0.84035**
  * EB25 **0.84708**
  * typed22+EB25 **0.84820**
* 즉 작은 group의 0/1 target rate를 그대로 믿지 않는 것이 효과적이었습니다.

### 3. Dynamic nested selector보다 fixed candidate가 나았다
* v29 nested selector:
  * outer 31415: **0.84624** vs v5 **0.83951**
  * outer 27182: **0.84512** vs v5 **0.84736**
* inner winner가 fold마다 계속 달라졌고 outer oracle과도 자주 불일치했습니다.
* v30에서 selection을 제거하고 후보를 고정하자 `typed22+EB CatBoost`가 두 outer seeds에서 **0.85971 / 0.85410**, 평균 **0.85690**으로 가장 강했습니다.

### 4. 6-seed paired repeated CV에서는 typed22+EB CatBoost가 v5보다 강했다
* v31 paired result:
  * seed42: `-0.00337`
  * seed123: `+0.00112`
  * seed777: `+0.00786`
  * seed2026: `+0.00673`
  * seed31415: `+0.02020`
  * seed27182: `+0.00673`
* 평균: v5 architecture **0.84287**, typed+EB CatBoost **0.84942**, delta **+0.00655**.
* 6개 seed 중 **5개 승리, 1개 패배**.

### 5. 그러나 final stress/headroom에서 기각됐다
* v32 group-aware: **0.79574 / 0.79237**, 기존 Deotte/v5 group-aware 수준을 넘지 못함.
* 5개 matched pseudo-test에서 v5 대비 delta: `[-0.01124, -0.01124, +0.00562, 0, -0.00562]`.
* 실제 test prediction은 v5와 16명 다르고, v10과 20명 다릅니다.
* v10-v5 정답 gap을 약 8명으로 보면, candidate가 v10을 넘으려면 16개의 switch 중 최소 **13개(81.25%)**가 올바른 방향이어야 합니다.
* pseudo/group stress는 그런 switch precision을 지지하지 않습니다.
* 결론: **repeated CV만 보면 승격감이었지만, final multi-surface gate에서는 제출 가치가 부족**합니다.

### 6. 최신 운영 결론
* 새로운 clean champion은 아직 없음. trusted clean champion은 계속 v5 `0.85410`, Public champion은 v10 `0.81578`.
* v26-v32가 보여준 핵심은 `more validation selection`보다 **고정된 독립 hypothesis + multi-surface falsification**이 더 중요하다는 점입니다.
* 현재 score-improvement clean campaign은 stop 상태. 이후 재개 조건은 새로운 외부 정보원, 완전히 다른 representation, 또는 기존 실패 원인을 직접 설명하는 강한 causal hypothesis입니다.

---

## 🔎 13. v35-v44: Residual -> Raw String -> Ticket Numeric Prefix

### 1. Residual forensics가 새 representation을 찾았다
* repeated OOF 기준 891명 중 stable-hard가 **121명 (13.58%)**였습니다.
* 특히 `male_P1`, `Mr_P1`, `female_P3`, `STONO` raw ticket-prefix에서 hard-case lift가 컸습니다.
* 기존 FE는 Title/TicketFreq/TicketPrefix/Deck 등으로 문자열을 요약했기 때문에 raw lexical pattern을 별도 검증했습니다.

### 2. Raw text는 약한 standalone model이지만 residual signal은 있었다
* Name/Ticket/Cabin char TF-IDF + LogisticRegression:
  * repeated 평균 약 `0.813~0.822`, group 약 `0.795~0.799`로 v5보다 약함.
* 그러나 confidence `>=0.75`인 v5 disagreement만 보면 개발 surfaces에서 매우 높은 precision이 관찰됐습니다.
* 실제 test에서 v10 직접 보정은 PassengerId **980 한 명**만 `1 -> 0`으로 바뀌었습니다.
* 이 한 row를 Public으로 probe하지 않고 field ablation으로 원인을 추적했습니다.

### 3. 980의 핵심은 Name이 아니라 numeric Ticket block이었다
* Passenger 980: `Ticket=364856`, Pclass 3, female, Fare 7.75, Cabin missing.
* field ablation:
  * Name-only death probability **0.3411** (gate 미통과)
  * Ticket-only **0.2240**
  * Name+Ticket **0.2331**
  * Full text **0.2447**
* train에서 `364` substring을 가진 ticket 16명의 생존률은 **0.125**였습니다.
* 반면 단순 `P3 female + Cabin missing + Fare 6~10` 생존률은 **0.59375**.
* 따라서 text model은 단순 여성 규칙이 아니라 **ticket-number block**을 포착하고 있었습니다.

### 4. Graph centrality는 추가 signal이 아니었다
* v40에서 PageRank/degree/core/articulation/betweenness/triangle/component-density를 X-only로 추가.
* Accuracy 평균 delta: fixed `-0.00336`, pseudo `-0.00562`, group `-0.00337`.
* graph branch 종료.

### 5. Ticket numeric P3 prefix EB는 genuine signal이었다
* v41에서 P2/P3 numeric prefix를 fold-safe EB로 일반화.
* v42 ablation 결과 coarse P2가 instability의 주원인이었고 **P3 full**이 가장 안정적이었습니다.
* P3 full vs FamilyFare+typed22+EB parent:
  * repeated mean **+0.00393**, 5/6 positive
  * group mean **+0.01235**, 2/2 positive
  * pseudo mean **+0.01910**, 5/5 positive
* 이는 v35 이후 발견한 가장 강한 새로운 clean representation signal입니다.

### 6. 하지만 Public champion을 넘길 headroom은 없었다
* v43 6-seed x 5-fold real-test bag:
  * 403/418 rows seed-unanimous
  * v5와 24 rows, v10과 32 rows disagreement
  * v5 score gap 8 correct를 넘어 v10을 이기려면 24 switches 중 **17 correct = 70.83%** 필요.
* v44 local paired switch precision:
  * six-seed bagged OOF Accuracy **0.85971** vs original v5 **0.85410**
  * 하지만 49 switches = 27 rescue / 22 harm = **55.10% precision**
  * repeated pooled **59.13%**, pseudo **57.14%**, group **48.23%**.
* 따라서 P3 feature 자체는 유효하지만 **test의 24 changed rows가 70.8% 이상 맞을 근거는 없음**.
* v43은 제출하지 않습니다.

### 7. 최신 결론
* Public champion: **v47 `0.83014`**.
* Trusted clean champion: v5 fixed OOF `0.85410` 유지.
* Clean OOF 단독 최고 계열 중 하나로 v44 bagged P3가 `0.85971`을 기록했지만, LB headroom gate 때문에 Public candidate로 승격하지 않습니다.
* 중요한 구분: **global Accuracy improvement != disagreement-switch precision improvement**. 이미 강한 Public champion이 존재할 때는 작은 OOF gain만으로 그 champion을 넘을 수 없습니다.
* 현 시점에서 score-improvement exploration은 종료. raw-text/native-cat/graph/numeric-ticket까지 orthogonal representation을 모두 검증했습니다.

---

## 🏁 14. v38-v47 실제 제출 캠페인 결과

* 사용자 승인 후 가능성이 있는 후보를 predeclared/evidence-backed 순서로 제출했습니다.
* **v38** `v10 + high-confidence raw-ticket text rescue`:
  * Submission ID `56841525`
  * Public **0.81818**
  * v10 대비 PassengerId 980 한 명만 `1 -> 0`; 점수 +1 correct이므로 이 보정은 실제로 적중.
* **v43** Ticket P3 full 6-seed bag:
  * ID `56841543`, Public `0.79665`
  * strong local P3 signal이 actual test 전체 후보로는 실패.
* **v45** `v38 + P3>=0.90`:
  * ID `56841584`, Public `0.81578`
  * v38에서 추가한 PassengerId 1122 `0 -> 1`이 잘못된 방향이었음을 확인.
* **v21 exact Deotte WCG+XGB**:
  * ID `56841621`, Public `0.81339`
  * historical exact reproduction은 현재 scoring set에서 v10/v38 미달.
* **v46** `v38 + conservative WCG female-death`:
  * ID `56841650`, Public **0.82775**
  * 추가 4 rows (929, 1051, 1141, 1301)가 전부 `1 -> 0`이며 v38 대비 +4 correct였으므로 네 방향 모두 적중.
* **v47** `v38 + broader Deotte female-death`:
  * ID `56841675`, Public **0.83014**
  * v46의 4 rows 외 9 rows를 더 바꾸고 v46 대비 +1 correct net; 추가 9개는 aggregate 기준 5 correct / 4 wrong.
  * **현재 Public champion**.
* **v21 WCG both standalone**:
  * ID `56841712`, Public `0.80382`.
* 현재 남은 submission은 **3개**. v47보다 근거가 약한 standalone/row-level 후보에는 쓰지 않고 보존합니다.
* `submissions/submission.csv`와 `submission_v47_score_0.83014.csv`를 현재 champion artifact로 동기화했습니다.
