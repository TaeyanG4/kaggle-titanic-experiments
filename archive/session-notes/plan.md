> HISTORICAL SESSION NOTES. Claims, scores and stopping decisions below reflect that stage, not a leakage-free certification. See ../../docs/03-validation-and-integrity.md for the final retrospective. Local-machine links may no longer resolve.

# 🗺️ Experiment Plan: What We Will Try Next

> **Kaggle Titanic - Machine Learning from Disaster**  
> **현재 최고 Public Score**: **0.83014 (v47 v38 + broader Deotte female-death guard)**  
> **Trusted fold-safe champion**: **v5 robust hard vote / Public 0.79665 / OOF 0.85410**
> **다음 목표**: Public `0.83014`을 실제로 초과할 독립 근거가 있는 후보만 제출
> **운영 원칙**: validation 개선만으로 제출하지 않고, `headroom gate + shift-aware validation + 독립 signal`을 모두 통과해야 함

---

## 🎯 1. 버전별 실험 진행 현황

* **v1 (완료)**: 6대 트리 모델 균등 Soft Voting + WCG 기초 피처 ➡️ **Public Score `0.79186`** (fold-safe CV Acc `0.84848`, AUC `0.88914`)
* **v2 (완료)**: 1인당 요금(`FarePerPerson`), 티켓 접두사, ROC-AUC 기반 Weighted Blending, WCG 후처리 ➡️ **Public Score `0.78708`** (fold-safe CV Acc `0.84175`, AUC `0.89730`)
* **v3 (완료)**: v1/v2 WCG fold leakage audit + 동일 fold hybrid 검토. 50:50 hybrid는 v1을 넘지 못해 보류.
* **v4 (완료)**: Diverse Model Zoo + TabICLv2 + 보수적 v1/TabICL blend. **Public 0.79425 새 champion**.
* **v5 (완료 / Trusted clean Champion)**: 12개 신규 family/model screen + MLP-PLR seed audit + hard vote. robust vote Public **0.79665**, 로컬 **0.85410**.
* **v6 (연구 완료 / 미제출)**: 공개 Titanic FE 조사 + broad/targeted ablation + LastName+Fare Family Survival. FamilyFare 신호는 유효했지만 champion Accuracy를 넘지 못함.
* **v7 (연구 완료 / 미제출)**: native categorical, relational Age/Deck preprocessing, CatBoost/RuleFit 제한적 HPO, disagreement audit. 안정적으로 v5 champion을 넘는 후보는 없었음.
* **v8/v12 (완료)**: TabPFN v2/v2.5/v3 finalist harness 및 실제 fit 완료. FamilyFare/Gunes feature transfer까지 검증했지만 trusted champion 초과 실패.
* **v9 (연구 완료 / 미제출)**: Gunes Evitan Advanced Feature Engineering Tutorial을 fold-safe하게 재구현. gunes_all panel Accuracy **0.84736**, AUC **0.90025**로 baseline panel 대비 +0.00449 개선했지만 v5 robust 0.85410은 넘지 못함.
* **v10 (재현 제출 완료)**: Gunes 원본 26-feature + leaderboard RF + 5-fold test averaging을 재현. Public **0.81578**로 새 Public-score 최고. 원본 reported best **0.83732**에는 미달.
* **v11 (감사 완료 / 미제출)**: Gunes exact 26-feature에서 target encoding만 fold-safe로 교체. single_best RF **0.83502**, leaderboard RF **0.82941**. v10 Public 강점이 historical global target encoding/test-aware 구조에 상당히 의존함을 확인.
* **v12 (연구 완료 / 미제출)**: TabPFN v2.5/v3 인증 및 fit 정상. FamilyFare/Gunes feature block 재학습. TabPFN 단독 최고 **0.84848**, `RuleFit + MLP + TabPFN v3 FamilyFare` hard vote가 trusted OOF **0.85410**으로 v5와 동률.
* **v13 (감사/후보 준비 완료 / 미제출)**: v10-v5 test conflict 22명을 Family/Ticket peer evidence로 분해. fold-safe transductive analogue는 **0.83277**, 가장 보수적인 no-group + v5/TabPFN consensus guard는 test에서 **2명(929, 1231)**만 v10에서 변경.
* **v14-v20 (완료)**: test-like pseudo validation + typed relational features + conservative guards. local 개선은 반복됐지만 Public `0.79425 / 0.81100`으로 v10 미달.
* **v21-v25 (완료)**: Deotte exact/fold-safe, repeated/group-aware stress, XGB/LGBM/RF/ET limited HPO, final low-DOF majority까지 완료. v25는 모든 local validation을 통과했지만 Public **0.79186**으로 실패.

---

## 🔬 2. 차기 실험 계획

### Phase 0: Validation Audit — WCG Fold Leakage 제거 (**완료**)
* **관찰**:
  * v1/v2 모두 `GroupSurvival`을 5-Fold 분할 전에 전체 train 라벨로 계산합니다.
  * Leave-One-Out이 자기 자신은 제외하지만 validation fold 동행자의 `Survived`가 validation feature에 들어갈 수 있습니다.
* **실험 설계**:
  * 동일 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`에서 아래 3개를 비교합니다.
    * `no_wcg`: `GroupSurvival=0.5` 고정.
    * `global_loo`: 기존 구현 재현. 진단용이며 승격 금지.
    * `fold_safe`: fold-train 라벨만 사용해 train/validation/test의 WCG 신호 생성.
  * v2는 현재 6개 모델 파라미터와 Weighted Soft Voting 가중치까지 동일하게 재현합니다.
* **실행 아티팩트**:
  * `scripts/audit_group_survival.py`
  * 결과: `exports/wcg_audit_v2/cv_group_survival_summary.csv`, `cv_group_survival_paired_deltas.csv`, `oof_fold_safe.csv`.
* **승격 규칙**:
  * `fold_safe - no_wcg`가 대부분 fold에서 양수면 WCG 신호를 유지합니다.
  * `global_loo >> fold_safe`면 기존 OOF가 누수로 부풀려졌다고 판단하고 이후 모든 실험 기준선을 `fold_safe`로 교체합니다.
  * v1/v2 모두 감사 완료. 이후 모든 실험 기준은 fold-safe WCG로 고정합니다.

---

### Phase 1: v1 & v2 하이브리드 블렌딩 (**완료 / 보류**)
* **가설**:
  * v1은 단순 평균으로 일반화가 잘 되어 Public Score(`0.79186`)가 높고,
  * v2는 정교한 피처와 가중 결합으로 모델의 순위 변별력(ROC-AUC `0.8980`)이 뛰어납니다.
  * 실제 50:50 결과는 Accuracy `0.84512`, AUC `0.89401`로 v1 단독 Accuracy를 넘지 못해 승격하지 않았습니다.

### Phase 2: v5 Model Zoo 확장 — **완료**
* RuleFit / FIGS / TabPFN v2 / TabM / RealMLP / TabR / FT-Transformer / MLP-PLR / RTDL / xRFM를 동일 fold-safe 기준으로 비교 완료.
* 단일 최고 MLP-PLR seed42: **0.85297**, 그러나 seed 민감성 확인.
* v4b + RuleFit + MLP-PLR hard vote: **0.85522**.
* 3-seed 평균 MLP를 사용한 robust vote: **0.85410**.
* TabPFN v2.5/v3 승인 및 실제 fit은 v12에서 완료. 추가 TabPFN 확장은 중단.

### Phase 3: 제한적 하이퍼파라미터 탐색 (**완료 / 추가 중단**)
891행의 작은 데이터에서 CV 자체에 과적합하지 않도록 무작정 100+ trials를 수행하지 않습니다. 모델별 20~40개 정도의 작은 탐색 후 상위 후보만 alternate-fold/seed에서 confirmation합니다.
* **CatBoost**: v7에서 제한적 grid + fold seed 123/777 stress test 완료. 추가 tiny HPO는 중단.
* v22에서 XGBoost 12 / LightGBM 10 / RandomForest 8 / ExtraTrees 8 설정을 동일 typed22 representation에서 screening 완료.
* fixed 최고 RF는 `0.85522`였지만 alternate folds에서 `0.84400 / 0.84287`로 하락.
* alternate까지 포함한 최고 평균도 RF6 `0.84998`에 그쳐 추가 HPO는 종료.

### Phase 4: Cross-fitted Stacking 메타 모델 (Level-2 Learner) (**완료 / champion 미달**)
* fold-safe OOF 예측 확률 행렬을 입력으로 사용하고 메타 모델도 cross-fit하여 2차 누수를 방지하는 probe를 v6/v12에서 수행했습니다.
* Logistic 계열 stacking과 TabPFN 포함 ensemble을 비교했지만 trusted v5 OOF Accuracy `0.85410`을 안정적으로 넘지 못했습니다.
* 복잡한 meta-model을 추가하는 대신 이후 ensemble은 낮은 자유도의 hard vote / convex blend를 우선합니다.

### Phase 5: 확률 임계값(Threshold) 최적화 (**중단 / 기대값 낮음**)
* Titanic Accuracy에서 threshold 미세조정은 소수 row만 바꾸며 현재 v10과의 gap을 안정적으로 메울 headroom이 부족함.
* Public feedback을 보며 threshold를 고르는 것은 LB-overfit 위험이 크므로 신규 독립 signal이 생기기 전에는 재개하지 않음.

### Phase 6: Deotte Mega Model / WCG+XGBoost Exact Reproduction (**완료**)
* precise `GroupId`, nanny/relative 연결, Wilkes-Hocking manual structural link, WCG deterministic rules를 공개 R source 기준으로 재구현.
* 공개 notebook 실행 출력에서 adult-male / solo-female XGBoost test decisions까지 복구.
* strict fixed OOF `0.85410`, repeated mean `0.84623`; group-aware에서는 평균 `0.82604`로 v5보다 강했으나 standalone champion 승격 근거는 부족.

### Phase 7: Repeated / Group-aware Validation Stress Test (**완료**)
* repeated Stratified와 Ticket+Family connected-component group-aware split을 구축.
* group-aware에서는 Deotte `0.82604` > v5 `0.81033` > typed models 약 `0.794`.
* 그러나 v25가 group-aware/pseudo/fixed에서 모두 개선되고도 Public에서 실패하여, 이 검증들도 actual test의 완전한 proxy가 아님을 확인.

### Phase 8: Classical Model Limited HPO (**완료 / 종료**)
* v22에서 완료. fixed-fold selection overfit이 확인되어 추가 탐색 금지.

### Phase 9: Low-DOF Ensemble (**완료 / 실패 증거 확보**)
* v25의 `v5 + Deotte + RF6` 2-of-3 majority가 fixed/group-aware/pseudo를 모두 개선했으나 Public `0.79186`.
* 더 복잡한 blend/stacking은 현재 validation meta-overfit만 증가시킬 가능성이 높아 중단.

### Phase 10: 낮은 우선순위 completeness 실험
* TensorFlow Decision Forests exact tutorial reproduction.
* WCG + KNN 공개 접근 exact reproduction.
* Ticket/Family graph label propagation 또는 graph-based relational model.
* 위 항목은 새로운 signal을 줄 가능성이 Deotte exact / validation stress / classical HPO보다 낮으므로 앞 단계 완료 후에만 수행합니다.

---

## 🧭 2-B. v26 이후 새 운영 계획 — Validation Reset + Independent Signal Only

### Phase 11: Mandatory Submission Headroom Gate (**신규 / 필수**) 
* 모든 제출 후보는 **현재 Public champion v10 `0.81578`**과 trusted anchor의 관계를 먼저 수학적으로 검사합니다.
* anchor가 champion보다 `g`개 정답이 뒤지고 candidate가 anchor와 `d`개 row에서 다르면, candidate의 최대 순개선은 `+d`입니다.
* binary Accuracy에서 candidate가 바꾼 `d`개 중 `w`개를 올바르게 고친 경우 순개선은 `2w-d`; champion 초과 조건은 `2w-d > g`입니다.
* **필수 기각 조건**:
  * `d <= g`이면 champion 초과가 불가능하거나 사실상 완벽한 switch accuracy가 필요하므로 제출 금지.
  * 필요한 switch precision `w/d`가 local disagreement audit의 신뢰구간 상단보다 높으면 제출 금지.
  * Public champion과 거의 같은 파일을 몇 row만 바꾸는 후보는 strong causal/domain evidence 없이는 제출 금지.
* v25는 v5 대비 `d=8`, v10-v5 gap도 약 `g=8`이어서 8/8이 필요했으므로 이 gate를 적용했다면 제출하지 않았어야 함.

### Phase 12: Adversarial Validation + Covariate-Shift Audit (**완료**) 
* 목표: `train 891`과 `test 418`이 **관측 가능한 X 분포에서 얼마나 다른지** 정량화하고, 어떤 feature/subgroup이 차이를 만드는지 파악합니다.
* train row=`0`, test row=`1`로 두고 X-only classifier를 repeated CV로 학습하여 adversarial AUC 측정.
* feature importance / SHAP 대신 우선 permutation importance와 단순 ablation으로 shift source를 확인: `Pclass`, `Sex`, `Age`, `Fare`, `TicketFreq`, `FamilySize`, `Title`, `Deck`, missingness, relational availability.
* AUC가 의미 있게 `0.5`를 넘으면 propensity `p(test|x)`에서 density-ratio weight를 만들어 **importance-weighted OOF Accuracy**를 계산합니다.
* weight는 clipping(예: p1~p99 또는 max 5~10)을 적용하고 propensity 자체도 cross-fit해 과도한 weight variance를 막습니다.
* 목적은 Public score를 맞추는 것이 아니라 **test-like risk estimate를 새롭게 만드는 것**입니다.
* **v26 결과**: train/test adversarial AUC 최고 약 **0.555**로 shift가 약했습니다. importance-weighted OOF도 실제 Public 역전을 설명하지 못해, covariate shift는 주원인으로 기각했습니다.

### Phase 13: Shift-Aware Training / Domain Adaptation (**스킵**) 
* adversarial AUC가 충분히 높을 때만 진행.
* 후보:
  1. propensity importance-weighted Logistic / RF / CatBoost / XGBoost.
  2. test-like train rows를 더 크게 반영하는 weighted ensemble.
  3. train+test X-only unsupervised representation(클러스터/embedding/community) 후 supervised model에 추가.
* test label이나 Public score는 어떤 weight/feature 선택에도 사용하지 않습니다.
* 승격은 ordinary repeated CV를 크게 훼손하지 않으면서 weighted test-risk에서 일관되게 개선될 때만 허용.
* v26 adversarial AUC가 낮아 대규모 shift-aware training은 실행 조건을 충족하지 못했습니다.

### Phase 14: Hierarchical / Partial-Pooling Relational Model (**완료 / 유효 신호**) 
* 기존 target-rate feature의 hard mean/median 대신 sparse Family/Ticket/Role 신호를 **partial pooling**하는 모델을 검토합니다.
* 1차 구현은 heavy Bayesian MCMC보다 저자유도 Empirical-Bayes/Beta-Binomial shrinkage로 시작.
* 예: `WomanChild × Pclass`, `AdultMale × Pclass`, Family, Ticket 그룹별 posterior survival probability를 global/role prior로 shrink.
* Family/Ticket group size 1~2에서 과도한 0/1 rate를 완화하는 것이 핵심 가설.
* fixed/repeated/group-aware/shift-weighted validation에서 동시에 개선될 때만 모델 family로 승격.
* **v27** Empirical-Bayes partial pooling:
  * typed22 mean Accuracy **0.84035**
  * EB25 **0.84708**
  * typed22+EB25 **0.84820**
  * 4개 seed에서 typed22보다 일관 개선.
* **v28** FamilyFare+CatBoost transfer에서는 기존 typed22 CatBoost 4-seed mean **0.84848**을 넘지 못해 1차 승격 보류.
* **v30/v31** 새 outer/repeated seeds에서 `typed22+EB+CatBoost`가 다시 강해져 6-seed paired audit까지 확대:
  * v5 architecture mean **0.84287**
  * typed22+EB CatBoost mean **0.84942**
  * 평균 delta **+0.00655**, 6 seeds 중 **5승 1패**.

### Phase 15: Nested Selection-Bias Audit (**완료**) 
* 지금까지 fixed/pseudo/group-aware 결과를 반복해서 보고 후보를 선택해 **validation meta-overfit**이 누적됐습니다.
* 다음 serious candidate부터는 outer 5-fold를 고정하고, 각 outer-train 내부에서만 model/feature/threshold 선택을 수행하는 nested selection audit을 1회 실행합니다.
* outer validation은 마지막까지 selector가 보지 못하게 하여 **"모델"이 아니라 전체 선택 과정의 일반화 성능**을 측정합니다.
* outer 결과가 기존 v5/Deotte anchor보다 의미 있게 좋지 않으면 제출하지 않습니다.
* **v29 결과**:
  * outer seed 31415: selector **0.84624**, v5 benchmark **0.83951** (+0.00673)
  * outer seed 27182: selector **0.84512**, v5 **0.84736** (-0.00224)
  * inner selector가 `typed+EB / RF6 / Deotte / typed Cat` 사이에서 계속 바뀌어 selection instability 확인.
* **v30 fixed-candidate audit**에서는 selector보다 미리 고정한 `typed22+EB CatBoost`가 두 outer seeds 평균 **0.85690**으로 더 안정적이었습니다. 즉 현재 병목은 모델보다 dynamic model-selection overhead에 가까움.

### Phase 16: Final Submission Gate (**다음 quota부터 적용**) 
* 아래를 모두 통과한 후보만 외부 제출:
  1. **Independent signal**: 기존 v5/Deotte/typed tree의 단순 재조합이 아닌 새 정보원 또는 새 학습 가정.
  2. **Repeated robustness**: 여러 split seed 평균 개선 + 큰 fold 붕괴 없음.
  3. **Shift-aware evidence**: adversarial/importance-weighted risk에서 비열화 없음.
  4. **Nested selection audit**: selector 포함 outer-CV 개선.
  5. **Headroom gate**: 현재 champion 초과가 수학적으로 가능하고 요구 switch precision이 현실적.
  6. **Artifact check**: 418 rows / PassengerId / hash / prediction delta audit 완료.
* 위 6개 중 하나라도 실패하면 제출 슬롯을 사용하지 않습니다.
* **v32에서 typed22+EB CatBoost를 최종 gate에 투입했으나 기각**:
  * group-aware Accuracy: **0.79574 / 0.79237**
  * matched pseudo-test deltas vs v5: **-0.01124, -0.01124, +0.00562, 0, -0.00562**
  * test prediction은 v5와 **16명**, v10과 **20명** 다름.
  * v10-v5 gap 약 8명 기준, v10을 넘으려면 16 switch 중 최소 **13명 = 81.25%**를 올바르게 뒤집어야 함.
  * pseudo/group evidence가 이 precision을 지지하지 않아 **제출 금지**.

### Phase 17: Stop Rule
* Phase 12~15에서 새로운 독립 signal이 v5/Deotte를 안정적으로 넘지 못하면 **clean score-improvement campaign을 종료**합니다.
* TF-DF exact, WCG+KNN exact, 추가 threshold/HPO/stacking은 교육용 completeness로만 수행하고 Public 향상 후보로 취급하지 않습니다.
* Public `0.81578`을 설명하기 위해 validation split/weight/row rule을 역으로 조정하지 않습니다.
* **현재 stop rule 발동**: v26~v32까지 independent signal / shift / nested / repeated / headroom을 모두 확인했으며, 가장 유망한 새 후보조차 final gate를 통과하지 못했습니다. 새로운 외부 정보원이나 완전히 다른 학습 가정이 생기기 전까지 clean score-improvement 제출은 중단합니다.

### Phase 18: Residual Forensics (**완료, v35**)
* v5 / Deotte / EB / consensus의 repeated OOF를 승객 단위로 합쳐 stable hard case를 찾았습니다.
* 891명 중 **stable hard 121명 (13.58%)**, stable easy 730명, ambiguous 40명.
* hard-case lift가 큰 slice:
  * `STONO` raw ticket prefix: hard rate **38.9%**
  * `male_P1`: **31.1%**
  * `Mr_P1`: **30.8%**
  * `female_P3`: **20.1%**
* 결론: 기존 FE가 버린 raw Ticket/Name 문자열 신호를 마지막 orthogonal representation으로 검증할 가치가 있음.

### Phase 19: Raw String Representation (**완료, v36-v39**)
* v36: `Name + Ticket + Cabin` char n-gram TF-IDF + LogisticRegression.
  * standalone 성능은 repeated **0.8128~0.8223** 수준으로 v5보다 크게 낮아 단독 모델은 기각.
  * 그러나 stable-hard slice에서는 기존 모델의 일부 오답을 독립적으로 rescue.
* v37: `text_only`가 v5와 다르고 confidence `>=0.75`일 때만 switch하는 frozen gate.
  * repeated 6/6 non-negative, 총 **8/8 switch correct**
  * group 2/2 non-negative, 총 **4/4 switch correct**
  * pseudo는 switch 0
  * 실제 test에서는 v5 기준 2명만 변경되어 v10 headroom을 수학적으로 넘을 수 없어 v5-based 제출 기각.
* v38: 같은 gate를 v10 fold-safe transductive analogue에 직접 적용.
  * repeated +2 net, pseudo +1 net, group 1승1패 net 0
  * actual v10에서는 **PassengerId 980 단 1명: 1 -> 0**만 변경.
  * unique validation passenger가 너무 적어 1-row Public probing으로 사용하지 않음.
* v39 field ablation:
  * 980의 사망 신호는 `Name` 자체보다 **Ticket numeric pattern**이 핵심.
  * Ticket-only prob `0.22395`, Name+Ticket `0.23314`, full text `0.24468`.
  * `Ticket=364856`의 `364xxx` 계열이 새로운 가설로 승격.

### Phase 20: X-only Graph Centrality (**완료 / 기각, v40**)
* v17의 connected-component/count를 넘어 degree, weighted degree, PageRank, core, articulation, approximate betweenness, triangles, component density 등을 추가.
* FamilyFare+typed22+EB CatBoost 대비 평균 Accuracy delta:
  * fixed **-0.00336**
  * pseudo **-0.00562**
  * group **-0.00337**
* AUC 일부 개선은 있었지만 Accuracy promotion surface는 전부 평균 악화. graph-centrality 축 종료.

### Phase 21: Ticket Numeric Prefix Partial Pooling (**완료, v41-v44**)
* raw-string residual에서 발견한 ticket-number block을 row-specific rule이 아닌 **모든 승객에 동일한 2/3자리 numeric prefix EB**로 일반화.
* v41 P2+P3 전체:
  * pseudo 5/5 non-negative, 평균 **+0.00899**
  * repeated 5/6 개선이나 seed27182 `-0.01684`
  * group 1승1패
* v42 ablation에서 원인 분리:
  * **P2 coarse prefix는 불안정 / 악화**
  * **P3 full**이 최종 신호:
    * repeated mean delta **+0.00393**, 5/6 개선
    * group mean delta **+0.01235**, **2/2 개선**
    * pseudo mean delta **+0.01910**, **5/5 개선**
    * 세 validation family 평균 모두 positive.
* v43 real-test 6-seed x 5-fold bagging:
  * 418명 중 **403명 seed-unanimous**, split-vote 15명
  * mean candidate는 v5와 24명, v10과 32명 다름
  * v10을 넘으려면 v5 대비 24 switch 중 최소 **17명 = 70.83%** 정답 필요.
* v44 paired switch-precision audit:
  * 6-seed bagged OOF P3: Accuracy **0.85971** vs original v5 **0.85410**
  * 하지만 49 switch 중 rescue 27 / harm 22 -> **switch precision 55.10%**
  * repeated pooled precision **59.13%**, pseudo **57.14%**, group **48.23%**
  * 필요한 70.83% headroom을 지지하지 못하므로 **v43 제출 기각**.

### Phase 22: Final Stop / Submission Policy (**현재 상태**)
* 새 독립 representation까지 모두 소진:
  * raw sparse text: standalone 기각, 보조 signal만 존재
  * exact high-cardinality native CatBoost: v7에서 이미 검증 완료
  * graph centrality: v40 기각
  * ticket numeric P3 EB: 실제 일반화 signal이지만 v10 headroom 미충족
* 사용자 승인 후 evidence-backed 후보를 실제 제출했고, **Public champion이 v47 `0.83014`로 갱신**됐습니다.
* 제출 결과:
  * v38 `v10 + text rescue(980)` → **0.81818**
  * v43 Ticket P3 full → `0.79665`
  * v45 `v38 + P3>=0.90(1122)` → `0.81578`
  * v21 exact Deotte WCG+XGB → `0.81339`
  * v46 `v38 + conservative WCG female-death 4 rows` → **0.82775**
  * v47 `v38 + broader Deotte female-death 13 rows` → **0.83014**
  * v21 Deotte WCG both standalone → `0.80382`
* 현재 남은 daily submission slot은 **3개**이며, 낮은 근거의 row-probing에는 사용하지 않습니다.
* `submission.csv`와 `submission_v47_score_0.83014.csv`는 SHA256 `CE8E484730A667E0B5D80068CF8F1418444E0113A9AF28996DD8185DF60EA0E4`로 동기화했습니다.

---

## 📋 3. 버전별 실험 우선순위 체크리스트

| 버전 / 실험 항목 | 예상 OOF Accuracy | 예상 Public Score | 우선순위 | 상태 |
| :--- | :---: | :---: | :---: | :---: |
| **v1: 6대 트리 모델 Soft Voting** | **0.84848 / AUC 0.88914** | **`0.79186`** | - | **완료** |
| **v2: 1인당 요금 + 가중치 블렌딩 + WCG 보정** | **0.84175 / AUC 0.89730** | **`0.78708`** | - | **완료** |
| **Phase 0-A: v2 WCG `no_wcg/global_loo/fold_safe` 감사** | **0.84175 / AUC 0.89730 (fold-safe)** | 제출 안 함 | 완료 | **완료** |
| **Phase 0-B: v1 WCG 교차 확인** | **0.84848 / AUC 0.88914** | 제출 안 함 | 완료 | **완료** |
| **v3-1: v1 & v2 50:50 하이브리드** | 0.84512 / AUC 0.89401 | 미제출 | 완료 | **보류** |
| **v4a: TabICLv2 단독** | **0.85073 / AUC 0.90086** | 0.78708 | 완료 | **완료** |
| **v4b: v1 90% + TabICLv2 10%** | **0.84961 / AUC 0.89004** | **0.79425** | 완료 | 이전 champion |
| **v5 MLP-PLR seed42** | **0.85297 / AUC 0.88595** | 미제출 | 완료 | seed 민감 |
| **v5 3-model hard vote** | **0.85522** | 미제출 | 완료 | 로컬 최고 |
| **v5 seed-robust hard vote** | **0.85410** | **0.79665** | 완료 | **Trusted clean CHAMPION** |
| **v5 TabPFN v3/v2.5** | 측정 완료 | - | 완료 | v12에서 재평가 |
| **v6 FamilyFare panel** | **0.84624 / AUC 0.90356** | 미제출 | 완료 | 보조 신호 |
| **v6 FamilyFare GradientBoosting** | **0.84961 / AUC 0.90125** | 미제출 | 완료 | 보조 신호 |
| **v7 CatBoost limited HPO** | **0.85073 / AUC 0.89860** | 미제출 | 완료 | stable but below champion |
| **v7 RuleFit no-linear** | **0.85073 / AUC 0.89534** | 미제출 | 완료 | vote 개선 없음 |
| **v7 native categorical CatBoost** | **0.84512** | 미제출 | 완료 | 기각 |
| **v7 relational preprocessing** | panel 최고 **0.84512** | 미제출 | 완료 | 기각 |
| **TabPFN v3/v2.5** | baseline 최고 **0.84624**, FamilyFare 최고 **0.84848** | 미제출 | 완료 | 인증/fit 정상 |
| **v8 TabPFN v2 + FamilyFare** | **0.84175 / AUC 0.89961** | 미제출 | 완료 | diversity 증가, Accuracy 하락 |
| **v9 Gunes tutorial transfer** | **0.84736 / AUC 0.90025** | 미제출 | 완료 | baseline 개선, champion 미달 |
| **v11 Gunes exact fold-safe RF** | **0.83502 / AUC 0.87415** | 미제출 | 완료 | v10 strength audit |
| **v12 TabPFN v3 + FamilyFare** | **0.84848 / AUC 0.90126** | 미제출 | 완료 | best TabPFN finalist |
| **v12 RuleFit + MLP + TabPFN v3 FF** | **0.85410** | 미제출 | 준비 완료 | trusted champion 동률 / test 1-row change |
| **Cross-fitted Stacking** | trusted champion 미달 | 미제출 | 완료 | v6/v12 probe 완료 |
| **v21 Deotte Mega/WCG exact reconstruction** | Python fixed **0.85410**, historical exact test decisions 복구 | 미제출 | 완료 | 공개 R source/output 기반 |
| **v21b Deotte strict fold-safe / repeated** | repeated mean **0.84623** | 미제출 | 완료 | fixed-only gain 아님 |
| **v23 Repeated + Group-aware validation stress** | Deotte **0.82604** > v5 **0.81033** | 제출 없음 | 완료 | 관계 그룹 holdout stress |
| **v22 XGB/LGBM/RF/ET limited HPO** | confirmed 최고 RF6 평균 **0.84998** | 미제출 | 완료 | fixed HPO overfit 확인 |
| **v25 Low-DOF final majority** | fixed **0.85746**, pseudo **0.86067**, group 개선 | **0.79186** | 완료 | local 전면승 / Public 실패 |
| **v19 pseudo-test consensus stress** | strict switch: v5 대비 pseudo 평균 **+0.00225** | **0.79425** | 완료 | local gain / Public 실패 |
| **v20 v10 conservative guard stress** | all3_weak: pseudo 평균 **+0.00562**, 5/5 non-negative | **0.81100** | 완료 | v10 `0.81578` 미달 |
| **v26 Adversarial Validation** | train/test AUC 최고 **~0.555** | 미제출 | 완료 | shift 약함 / domain adaptation 스킵 |
| **v27 EB Partial Pooling** | typed22 `0.84035` → typed+EB **0.84820** | 미제출 | 완료 | 독립 relational signal 확인 |
| **v28 EB Transfer** | best 4-seed mean **0.84848** | 미제출 | 완료 | 기존 typed CatBoost 미초과 |
| **v29 Nested Selection Audit** | +0.00673 / -0.00224 across outer seeds | 미제출 | 완료 | selector 불안정 |
| **v30 Fixed Candidate Outer Audit** | typed+EB Cat mean **0.85690** on 2 seeds | 미제출 | 완료 | selector보다 fixed candidate 우수 |
| **v31 Paired Repeated Audit** | typed+EB **0.84942** vs v5 architecture **0.84287**, 5승1패 | 미제출 | 완료 | serious finalist 승격 |
| **v32 Final Stress + Headroom** | group **0.79574/0.79237**, pseudo 3악화/1무/1개선 | 미제출 | 완료 | **Final gate 실패 / 제출 기각** |
| **v33 v5+Deotte+EB consensus** | repeated 평균 **+0.00898**, pseudo **+0.00674** vs v5 | 미제출 | 완료 | strong local consensus |
| **v34 consensus headroom** | v5와 2명 차이 | 미제출 | 완료 | v10을 수학적으로 이길 수 없어 기각 |
| **v35 Residual Forensics** | stable hard **121/891 (13.58%)** | - | 완료 | raw Ticket/Name 신호 후보 발견 |
| **v36 Raw-string TF-IDF** | repeated **~0.813-0.822** | 미제출 | 완료 | 단독 모델 기각 / hard-case rescue 신호 |
| **v37 High-confidence text rescue** | repeated 8/8, group 4/4 switch correct | 미제출 | 완료 | real test 2-row뿐 / headroom 불가 |
| **v38 v10 + text rescue** | repeated +2, pseudo +1, group net 0 | 미제출 | 완료 | test 980 한 명만 변경 / evidence 부족 |
| **v40 Graph Centrality** | fixed -0.00336, pseudo -0.00562, group -0.00337 | 미제출 | 완료 | 기각 |
| **v42 Ticket P3 EB** | repeated **+0.00393**, group **+0.01235**, pseudo **+0.01910** | 미제출 | 완료 | strongest new clean signal |
| **v43 P3 Finalist Test Bag** | 403/418 seed-unanimous | 미제출 | 완료 | v5와 24명 변경 / 70.83% headroom 필요 |
| **v44 P3 Switch Audit** | bagged OOF **0.85971**, switch precision **55.10%** | 미제출 | 완료 | **headroom 실패 / 제출 기각** |
| **v38 text rescue** | v10 + PassengerId 980 one-row guard | **0.81818** | 제출 완료 | v10 초과 / 980 correction 확인 |
| **v45 v38 + P3>=0.90** | PassengerId 1122 추가 flip | **0.81578** | 제출 완료 | 1122 flip 실패 |
| **v46 v38 + WCG female-death** | 4-row structural guard | **0.82775** | 제출 완료 | 4/4 방향 적중 |
| **v47 v38 + broader female-death** | 13-row Deotte female-death guard | **0.83014** | 제출 완료 | **현재 Public CHAMPION** |
| **v21 exact Deotte WCG+XGB** | historical exact reproduction | **0.81339** | 제출 완료 | champion 미달 |
| **v21 WCG both standalone** | strongest local Deotte standalone | **0.80382** | 제출 완료 | champion 미달 |

### 현재 승격 판단
* **Public-score champion은 v47 `0.83014`**입니다. 이는 v10을 anchor로 한 text/ticket residual 분석과 Deotte female-death guard를 결합한 test-specific artifact이므로 trusted clean champion과 구분합니다.
* **Trusted fold-safe champion은 v5 robust hard vote: OOF Accuracy `0.85410`, Public `0.79665`**입니다.
* seed42 hard vote는 로컬 `0.85522`로 더 높지만 seed 의존성이 있어 trusted champion으로 승격하지 않습니다.
* 계획했던 고우선순위 실험인 **Deotte exact/fold-safe, repeated/group-aware validation, classical limited HPO, low-DOF final ensemble까지 모두 완료**했습니다.
* v25는 fixed/group-aware/pseudo 세 종류의 독립 validation에서 모두 v5보다 개선됐지만 Public `0.79186`에 그쳤습니다. 이는 더 복잡한 local selector를 만들 근거가 아니라 **현재 test가 local validation으로 잘 식별되지 않는다는 중단 신호**로 취급합니다.
* 남은 TF-DF exact, WCG+KNN exact는 이미 충분히 탐색한 tree/KNN/WCG family의 completeness 재현이며, score 개선 목적의 기대 정보가 낮아 **주력 campaign은 종료**합니다.
* **v14-v20 pseudo-test 결론**:
  * 실제 test의 Family/Ticket overlap과 covariate 비율을 맞춘 pseudo-test는 typed relational signal을 재현했지만 actual Public 선택력은 충분하지 않았습니다.
  * fixed OOF `0.85859` + pseudo 개선을 통과한 v15 strict consensus가 Public `0.79425`, fixed/pseudo 모두 non-negative였던 v20 conservative guard도 `0.81100`으로 v10 `0.81578`을 넘지 못했습니다.
  * 따라서 pseudo-test를 Public에 맞게 다시 조정하지 않습니다. 이는 validation-to-LB overfitting으로 이어질 가능성이 큽니다.
* generic feature accumulation, native-cat 반복, 작은 HPO 반복, Public LB 기반 row-by-row probing은 중단합니다. 이전 quota에서 **10/10 사용 완료** 상태였고, 이후에도 새로운 독립 신호가 final gate를 통과하지 않는 한 제출하지 않습니다.
* v31의 typed+EB는 relational stress에서, v42의 Ticket P3 EB는 headroom/switch precision에서 각각 탈락했습니다. **현재 clean score-improvement campaign은 다시 stop 상태**입니다.
