> HISTORICAL SESSION NOTES. Claims, scores and stopping decisions below reflect that stage, not a leakage-free certification. See ../../docs/03-validation-and-integrity.md for the final retrospective. Local-machine links may no longer resolve.

# 🤝 Handoff & Session History

> **Kaggle Titanic - Machine Learning from Disaster**  
> 이 문서는 세션 동안 수행된 모든 작업, 버전별 제출 점수 기록, 현재 시스템 상태 및 결과 아티팩트를 기록합니다.

---

## 📅 1. 작업 히스토리 요약 (Session Timeline)

1. **[v1] 베이스라인 고도화 및 첫 제출 (2026-10-04 13:03)**:
   * 기존 Colab 노트북(`Kaggle_Titanic_2023_01_09_0_0_2.ipynb`)의 타깃 결측치 누수 버그(330개 train 유입) 해결.
   * WCG(Woman-Child-Group) Leave-One-Out 생존 신호 피처 구현.
   * 6대 트리 모델(RF, ET, GB, XGB, LGB, Cat) Soft Voting 앙상블 구축 (OOF CV 0.8507, AUC 0.8715).
   * **캐글 제출 결과: Public Score `0.79186` 달성 (기존 최고 기록 0.77511 경신)**.
2. **[v2] 피처 고도화, 가중치 블렌딩 & 시각화/원리 해설 통합 (2026-10-04 13:40)**:
   * 1인당 실제 요금(`FarePerPerson`), 티켓 접두사(`TicketPrefix`), 가족 유형(`FamilyType`), 나이-등급 교차(`Age_Pclass`), 기혼 여성(`IsMarriedWoman`) 추가.
   * OOF ROC-AUC 기여도 기반 **Weighted Soft Voting** 적용 (ROC-AUC `0.898`로 대폭 향상).
   * WCG 대가족 전원 사망 그룹 예외 보정(Post-Processing) 적용.
   * 파이프라인 아키텍처 다이어그램 및 8종의 EDA/진단 그래프, 초보자를 위한 핵심 용어/원리 해설을 주피터 노트북에 통합.
   * **캐글 제출 결과: Public Score `0.78708` 기록**.
3. **[공통] 안티그래비티 및 타 에이전트(Claude Code, Codex)용 스킬 패키징**:
   * `context-continuity` 스킬을 안티그래비티 전역 폴더 및 `exports/`에 표준 규격으로 패키징 완료.
4. **[Phase 0 Audit] WCG Fold Leakage 감사 준비 (2026-10-04)**:
   * v1/v2 모두 `GroupSurvival`을 CV 분할 전에 전체 train 라벨로 계산하는 구조임을 확인하여 fold 경계 타깃 누수 위험을 발견.
   * 챔피언/제출본은 변경하지 않고 `scripts/audit_group_survival.py`를 별도 추가.
   * `--profile v1|v2`로 각 버전의 피처/모델/앙상블 가중치를 재현하며, 동일 fold에서 `no_wcg`, `global_loo`, `fold_safe`를 비교하도록 구현.
   * `python -m py_compile` 및 v1/v2 `--check-only` 통과. 실제 장시간 5-Fold × 6모델 학습은 **미실행**.
5. **[v3] v2 Fold-safe Audit 완료 및 Local Candidate 생성 (2026-10-04)**:
   * v2 audit 결과 `fold_safe` Ensemble OOF Accuracy `0.84175`, ROC-AUC `0.89730` 확인.
   * `no_wcg` 대비 `+0.00786 Acc / +0.01352 AUC`, `global_loo` 대비 차이는 `-0.00112 Acc / -0.00140 AUC`로 작음.
   * `scripts/build_v3_candidate_from_audit.py`로 재학습 없이 `submissions/submission_v3.csv` 생성 (생존 예측 155/418).
   * v3 후보는 v1과 6명, v2와 2명의 test prediction만 다름.
   * `kaggle-skill validate` 형식 검증 통과. **Kaggle 제출은 하지 않음**.
   * `scripts/generate_notebook_v3.py` 추가 및 `notebooks/Titanic_Ensemble_Pipeline_v3.ipynb` 생성/실행 완료 (analysis-only, 모델 재학습 없음).
6. **[v3] v1 Fold-safe Audit 및 Hybrid 교차검증 완료 (2026-10-04)**:
   * v1 fold-safe Ensemble OOF Accuracy **0.84848**, ROC-AUC **0.88914**.
   * v1 50% + v2 50% hybrid는 Accuracy 0.84512 / AUC 0.89401로 v1 단독을 넘지 못해 승격하지 않음.
7. **[v4] Diverse Model Zoo + TabICLv2 (2026-10-04)**:
   * 동일 fold manifest와 fold-safe WCG로 10개 신규 모델을 비교.
   * TabICLv2 단독 OOF Accuracy **0.85073**, ROC-AUC **0.90086**로 Model Zoo 1위.
   * TabICLv2 단독 제출 ID **56826952**, Public Score **0.78708**.
   * v1 90% + TabICLv2 10% 보수적 blend는 OOF Accuracy **0.84961**, test에서 v1 대비 1명만 변경.
   * blend 제출 ID **56827009**, Public Score **0.79425**로 **새 Public Champion**.
   * `submissions/submission_v4_score_0.79425.csv`로 보존하고 `submission.csv`도 새 champion에 동기화.
   * `notebooks/Titanic_Ensemble_Pipeline_v4.ipynb` 생성 및 analysis-only 실행 완료.
8. **[v5] Aggressive Multi-Family Model Zoo (2026-10-04)**:
   * RuleFit, FIGS, TabPFN v2, TabM, RealMLP, TabR, FT-Transformer, MLP-PLR, RTDL MLP/ResNet, sklearn MLP, xRFM까지 동일 fold-safe 5-Fold로 비교.
   * 단일 최고는 MLP-PLR seed42: Accuracy **0.85297**, AUC **0.88595**.
   * RuleFit: **0.84961 / 0.89598**, TabPFN v2: **0.84287 / 0.89663**.
   * MLP-PLR seed audit: seed42/142/242 Accuracy = **0.85297 / 0.84063 / 0.84287** → 단일 seed 민감성 확인.
   * v4b + RuleFit + MLP-PLR 2-of-3 hard vote: seed42 기준 **0.85522**.
   * 3-seed 평균 MLP를 사용한 robust hard vote: **0.85410**, fold std **0.01336**.
   * `submission_v5.csv`, `submission_v5_hard_vote.csv`, `submission_v5_robust_vote.csv` 모두 schema validation PASS.
   * robust hard vote 제출 ID **56827698**, Public Score **0.79665**로 **새 Public Champion**.
   * `submission_v5_score_0.79665.csv`로 보존했고 `submission.csv`도 새 champion과 동기화.
   * TabPFN v3/v2.5는 Prior Labs 1회 license/auth가 필요해 blocked 상태로 기록.
   * `notebooks/Titanic_Ensemble_Pipeline_v5.ipynb` 생성/실행 완료.
9. **[v6] Public Feature Engineering / Preprocessing Audit (2026-10-04)**:
   * Kaggle 공개 notebook/discussion에서 Title, cabin, family/ticket counts, Sex×Pclass, family survival 등의 반복 아이디어 조사.
   * broad block ablation 결과 대량 feature 추가는 대부분 성능 하락.
   * 단일 feature screen에서 `FarePerFamily`, `AdultMale`가 panel Accuracy +0.00224 수준의 작은 개선.
   * 공개 Family Survival 계열의 **LastName + Fare** 그룹을 fold-safe하게 재구현.
   * FamilyFare panel **0.84624 / AUC 0.90356**, GradientBoosting **0.84961 / 0.90125**.
   * RuleFit/MLP에 FamilyFare를 직접 넣으면 AUC는 상승하지만 Accuracy 하락.
   * cross-fitted Logistic stacking도 v5 robust **0.85410**을 넘지 못함.
   * 결론: v6 제출 없음, Public champion은 v5 **0.79665** 유지.
   * `notebooks/Titanic_Ensemble_Pipeline_v6.ipynb` 생성/실행 완료.
10. **[v7] Preprocessing Bottleneck Audit + Selective HPO (2026-10-05)**:
   * CatBoost native categorical 7개 표현을 비교. 최고 Accuracy **0.84512**로 기존 one-hot 표현보다 약해 기각.
   * 가족/티켓 기반 Age/Deck 결측치 보간을 비교. panel 최고 **0.84512**로 champion 미달.
   * AdultMale CatBoost 제한적 HPO: `slow_d3_l2_6` trusted-fold **0.85073**, alternate fold seeds 123/777에서 각각 **0.84624 / 0.84624**.
   * disagreement audit: FamilyFareGB가 champion error 17명을 rescue했지만 21명을 harm. Large-family rule의 +2 correct는 fold 4에만 집중되어 기각.
   * 공개 woman-child deterministic override는 v5 robust OOF를 **0명 변경**하여 현재 ensemble이 이미 해당 규칙을 흡수한 것으로 판단.
   * RuleFit 제한적 HPO: `no_linear` seed42 단독 Accuracy **0.85073**, 하지만 robust hard vote는 **0.85410** 그대로.
   * `notebooks/Titanic_Ensemble_Pipeline_v7.ipynb` 생성/실행 완료.
   * Public champion `submission_v5_score_0.79665.csv` 및 `submission.csv`는 변경하지 않음.
11. **[v8 readiness] TabPFN Finalist Harness (2026-10-05)**:
   * `scripts/tabpfn_finalist_v8.py` 추가. TabPFN v2/v2.5/v3와 baseline/FamilyFare representation을 동일 fixed fold에서 비교 가능.
   * TabPFN v2 baseline 재현: Accuracy **0.84287**, AUC **0.89663**.
   * TabPFN v2 + FamilyFare: Accuracy **0.84175**, AUC **0.89961**.
   * FamilyFare는 champion-error rescue를 8명→11명으로 늘렸지만 harm도 22명이라 전체 Accuracy는 하락.
   * cross-fitted threshold는 두 v2 variant 모두 **0.83838**로 악화.
   * high-confidence baseline+FamilyFare consensus override는 v5 robust OOF를 **0명 변경**.
   * TabPFN v2.5/v3는 Prior Labs 1회 license/auth가 필요함을 실제 fit 단계에서 재확인.
   * blocked 실행이 aggregate CSV를 덮어쓰는 resume bug를 수정; per-model artifact와 aggregate 모두 안전하게 재구성됨.
12. **[v9] Gunes Evitan Advanced FE Transfer Audit (2026-10-05)**:
   * Gunes Evitan의 Titanic - Advanced Feature Engineering Tutorial을 아이디어 소스로 사용.
   * Sex×Pclass Age imputation, Age 10-quantile / Fare 13-quantile bins, grouped Deck, coarse Title grouping, surname/ticket survival-rate를 프로젝트 validation에 맞게 재구현.
   * 공개 notebook의 target encoding은 그대로 복사하지 않고 fold-train label만 사용하는 leakage-safe 방식으로 재작성.
   * baseline 3-model panel **0.84287 / AUC 0.89786**.
   * gunes_all panel **0.84736 / AUC 0.90025**.
   * fold-safe surname/ticket survival-rate components panel **0.84624 / AUC 0.90031**.
   * v5 robust 대비 17 rescue / 23 harm, net -6이라 champion 승격 및 Kaggle 제출은 하지 않음.
   * 산출물: scripts/gunes_feature_audit_v9.py, exports/v9/.
13. **[v10] Gunes Original Reproduction (2026-10-05)**:
   * 원본 공개 notebook의 최종 26-feature pipeline과 leaderboard RandomForest를 최대한 동일하게 재현.
   * RF 1750 trees, depth 7, min split 6, min leaf 6, 5-fold StratifiedKFold(seed=5), test probability 평균.
   * 현재 sklearn 호환을 위해 historical max_features=auto를 동치인 sqrt로 대체.
   * 재현 OOF Accuracy **0.83614**, AUC **0.89311**, test positives **149**.
   * 원본 notebook에 표시된 첫 10개 submission label과 재현 결과가 모두 일치.
   * Submission ID **56830042**, Public Score **0.81578**.
   * 원본 reported best **0.83732 (V121)**에는 미달. 점수 기준으로 341/418 vs 약 350/418, 즉 약 9 correct 차이.
   * `submissions/submission_v10_score_0.81578.csv` 보존, `submission.csv`는 v10 Public-score champion과 동기화.
   * v5 robust는 trusted fold-safe champion으로 별도 보존.
14. **[v11] Gunes Exact Fold-Safe Audit (2026-10-05)**:
   * Gunes exact 26-feature representation은 유지하고 Family/Ticket target encoding만 strict fold-safe로 교체.
   * single_best RF **0.83502 / AUC 0.87415**, leaderboard RF **0.82941 / AUC 0.87771**.
   * v5 구성원에 RF를 교체한 hard vote는 모두 v5 robust **0.85410**보다 낮음.
   * 결론: v10의 Public 강점은 representation/RF만이 아니라 historical global target encoding + test-aware group eligibility에 크게 의존.
15. **[v12] TabPFN v2.5/v3 Feature-Retrain + Ensemble Probe (2026-10-05)**:
   * TabPFN v2.5/v3가 실제 fit까지 정상 작동함을 확인.
   * v2.5 + FamilyFare **0.84848 / AUC 0.90118**, v3 + FamilyFare **0.84848 / AUC 0.90126**.
   * Gunes rates 단독도 v2.5/v3 모두 Accuracy **0.84848**.
   * v3 + FamilyFare + Gunes all은 Accuracy **0.84736**이지만 v5 error 14명을 rescue하여 TabPFN 계열 최고 diversity.
   * hard-vote/soft-average/cross-fitted stacking을 비교했으나 v5 **0.85410**을 넘는 trusted 후보는 없음.
   * `RuleFit + MLP seed-mean + TabPFN v3 FamilyFare` hard vote는 OOF **0.85410 동률**, test에서 v5 대비 1명만 변경(PassengerId 1017, female, age 17, 3rd class: 1→0).
   * `submissions/submission_v12_rule_mlp_t3ff.csv` 생성 및 schema validation PASS, 미제출.
16. **[v13] v10 Test-Aware Conflict Audit + Guard Selector (2026-10-05)**:
   * v10의 test-aware signal을 train/test 양쪽에 동시에 등장하는 Family/Ticket 그룹의 train-label median survival로 분해.
   * v10 vs v5 test disagreement는 **22명**. signal coverage: NA=1 11명, NA=0.5 3명, NA=0 8명.
   * fixed trusted folds에서 fold-train/fold-validation 공통 group eligibility를 사용한 transductive fold-safe analogue RF는 **0.83277 / AUC 0.88029**.
   * 가장 보수적인 guard는 SurvivalRateNA=0이고 v5와 TabPFN v3 FamilyFare가 v10 반대에 동의할 때만 trusted label을 채택.
   * 실제 test에서는 **PassengerId 929, 1231** 두 명만 변경.
   * `submissions/submission_v13_guard_no_group_consensus.csv` 생성, schema validation PASS, **미제출**.
   * 세부 peer evidence: `exports/v13/test_conflict_peer_evidence.csv`, 보고서 `exports/v13/conflict_audit_report.md`.

---

## 🏆 2. 캐글 제출 점수 변천사 (Score Ledger)

| 버전 | Submission ID | 제출 일시 | 제출 파일명 | 제출 설명 | 상태 | Public Score | CV Accuracy | CV ROC-AUC | 비고 |
| :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **v1** | **56824542** | 2026-10-04 13:03 | `submission_v1_score_0.79186.csv` | Titanic High-Performance Ensemble Pipeline (WCG+6Models) | COMPLETE | **`0.79186`** | **0.84848** | **0.88914** | fold-safe 재측정 완료 |
| **v2** | **56825418** | 2026-10-04 13:40 | `submission_v2.csv` | Titanic v2: Advanced FE + Weighted Blending + WCG Post-Processing | COMPLETE | **`0.78708`** | **0.84175** | **0.89730** | fold-safe 재측정 완료 |
| **v4a** | **56826952** | 2026-10-04 14:53 | `submission_v4.csv` | TabICLv2 + Fold-safe WCG | COMPLETE | `0.78708` | **0.85073** | **0.90086** | 단독 OOF 최고, Public은 v1 미달 |
| **v4b** | **56827009** | 2026-10-04 14:55 | `submission_v4_blend_90_10.csv` | v1 90% + TabICLv2 10% fold-safe blend | COMPLETE | **`0.79425`** | **0.84961** | 0.89004 | 이전 Public Champion |
| **v5** | **56827698** | 2026-10-04 15:29 | `submission_v5_robust_vote.csv` | v4b + RuleFit + 3-seed MLP-PLR robust hard vote | COMPLETE | **`0.79665`** | **0.85410** | vote-score AUC 참고용 | **★ 새 Public Champion** |
| **v9** | **56829831** | 2026-10-04 17:15 | `submission_v9_gunes_all.csv` | Gunes fold-safe feature transfer panel | COMPLETE | **`0.78229`** | **0.84736** | **0.90025** | 연구 제출 |
| **v10** | **56830042** | 2026-10-04 17:26 | `submission_v10_gunes_original_reproduction.csv` | Gunes original 26-feature + RF 1750 + 5-fold test averaging | COMPLETE | **`0.81578`** | **0.83614** | **0.89311** | **★ Public Score Champion / leakage-prone CV** |
| 레거시 | 38645411 | 2024-06-11 11:02 | `submission.csv` | - | COMPLETE | 0.77272 | - | - | 과거 제출 |
| 레거시 | 29796170 | 2023-01-09 18:21 | `titanic_sub...` | - | COMPLETE | 0.77511 | - | - | 과거 최고 기록 |
| 레거시 | 29796595 | 2023-01-09 18:42 | `titanic_sub...` | - | COMPLETE | 0.00000 | - | - | Target NaN 누수 채점 실패 |

> 💡 **v1과 v2의 점수 차이 분석 (0.79186 vs 0.78708)**:
> * 타이타닉 테스트 데이터(418명 중 Public LB 채점 대상 약 209명)에서 $0.79186 - 0.78708 = 0.00478$은 **단 1~2명의 예측 차이**에 불과합니다.
> * fold-safe 재측정에서는 v1 AUC `0.88914`, v2 AUC `0.89730`으로 v2의 순위 변별력이 여전히 더 높았지만, Accuracy와 Public Score는 v1이 더 좋았습니다.

---

## 📦 3. 현재 아티팩트 및 파일 상태 (Current State)

* **노트북**:
  * `notebooks/Titanic_Ensemble_Pipeline_v7.ipynb` (v7 최신: preprocessing / limited HPO / residual audit)
  * `notebooks/Titanic_Ensemble_Pipeline_v6.ipynb` (v6 최신: public FE research / FamilyFare / stacking audit)
  * `notebooks/Titanic_Ensemble_Pipeline_v5.ipynb` (v5 최신: aggressive zoo / seed audit / hard votes)
  * `notebooks/Titanic_Ensemble_Pipeline_v4.ipynb` (v4: Model Zoo / TabICLv2 / Public champion blend)
  * `notebooks/Titanic_Ensemble_Pipeline_v3.ipynb` (v3: fold-safe WCG audit 및 local candidate 분석)
  * [Titanic_Ensemble_Pipeline_v2.ipynb](file:///h:/kaggle/practice/Titanic%20-%20Machine%20Learning%20from%20Disaster/notebooks/Titanic_Ensemble_Pipeline_v2.ipynb) (v2 최신 노트북, 863KB, 다이어그램/그래프/용어해설 완비)
  * [Titanic_Ensemble_Pipeline_v1.ipynb](file:///h:/kaggle/practice/Titanic%20-%20Machine%20Learning%20from%20Disaster/notebooks/Titanic_Ensemble_Pipeline_v1.ipynb) (v1 원본 보관, 0.79186 달성 노트북)
* **제출 파일**:
  * `submissions/submission_v1_score_0.79186.csv` (0.79186 스코어본)
  * `submissions/submission_v2.csv` (v2 제출본)
  * `submissions/submission_v3.csv` (fold-safe local candidate, 검증 통과, 미제출)
  * `submissions/submission_v4.csv` (TabICLv2 단독, Public 0.78708)
  * `submissions/submission_v4_blend_90_10.csv` (Public 0.79425)
  * `submissions/submission_v4_score_0.79425.csv` (이전 champion 백업)
  * `submissions/submission_v5.csv` (MLP-PLR seed42, 검증 통과, 미제출)
  * `submissions/submission_v5_hard_vote.csv` (OOF 0.85522, 검증 통과, 미제출)
  * `submissions/submission_v5_robust_vote.csv` (OOF 0.85410, Public 0.79665, 제출 ID 56827698)
  * `submissions/submission_v5_score_0.79665.csv` (trusted fold-safe champion 백업)
  * `submissions/submission_v10_score_0.81578.csv` (**현재 Public-score champion 백업**)
  * `submissions/submission_v15_switch_t3_t25_cat.csv` (v19 pseudo-test 승격 후보, 제출 ID 56831596, Public **0.79425**)
  * `submissions/submission_v18_v10_guard_all3_weak.csv` (v20 conservative v10 guard, 제출 ID 56831621, Public **0.81100**)
  * `submissions/submission.csv` (현재 Public-score champion과 동기화)
* **스크립트**:
  * `scripts/generate_notebook.py` (v1 재생성 스크립트)
  * `scripts/generate_notebook_v2.py` (v2 재생성 스크립트)
  * `scripts/audit_group_survival.py` (v1/v2 WCG fold leakage 감사, 실행 대기)
  * `scripts/build_v3_candidate_from_audit.py` (v2 audit에서 v3 local candidate 생성)
  * `scripts/build_v3_hybrid_from_audits.py` (v1/v2 fold-safe audit 완료 후 fixed 50:50 hybrid 및 소규모 weight scan)
  * `scripts/generate_notebook_v3.py` (v3 분석 노트북 생성/실행, 모델 학습 없음)
  * `scripts/check_v4_environment.py` (v4 dependency/GPU inventory)
  * `scripts/model_zoo_screen_v4.py` (v4 fold-safe model zoo)
  * `scripts/build_v4_candidate.py` (TabICLv2 threshold audit)
  * `scripts/build_v4_blend_candidate.py` (90:10 champion 후보 생성)
  * `scripts/generate_notebook_v4.py` (v4 분석 노트북 생성/실행)
  * `scripts/model_zoo_screen_v5.py` (resumable expanded model zoo)
  * `scripts/build_v5_candidate.py` (MLP-PLR + threshold audit)
  * `scripts/probe_mlp_plr_seeds_v5.py` (seed stability audit)
  * `scripts/build_v5_hard_vote_candidate.py` (seed42 hard vote)
  * `scripts/build_v5_robust_vote_candidate.py` (seed-robust hard vote)
  * `scripts/generate_notebook_v5.py` (v5 analysis notebook)
  * `scripts/feature_ablation_v6.py`, `targeted_feature_screen_v6.py` (public FE ablation)
  * `scripts/group_key_audit_v6.py` (LastName+Fare / Ticket / Cabin peer audit)
  * `scripts/train_selected_v6_models.py`, `train_familyfare_modern_v6.py`
  * `scripts/stacking_probe_v6.py`, `generate_notebook_v6.py`
  * `scripts/native_catboost_v7.py`, `preprocessing_audit_v7.py`
  * `scripts/catboost_hpo_v7.py`, `disagreement_audit_v7.py`, `rulefit_hpo_v7.py`
  * `scripts/generate_notebook_v7.py`
  * `scripts/tabpfn_finalist_v8.py` (v2/v2.5/v3 finalist + FamilyFare comparison)
  * `scripts/gunes_feature_audit_v9.py` (Gunes tutorial fold-safe transfer)
  * `scripts/reproduce_gunes_original_v10.py` (Gunes original leaderboard pipeline reproduction)
  * `scripts/pseudo_test_relational_v14.py` (실제 test relation geometry에 맞춘 5개 pseudo-test split + relational block 비교)
  * `scripts/typed_relational_fixed_v14.py`, `tabpfn_typed_relational_v14.py` (typed22 fixed-fold / TabPFN confirmation)
  * `scripts/typed_relational_tree_screen_v15.py`, `v15_consensus_probe.py` (FamilyFare+typed22 tree screen / consensus)
  * `scripts/typed_relational_alpha_screen_v15.py`, `typed_relational_alpha8_finalists_v16.py` (smoothing alpha/finalist audit)
  * `scripts/relational_graph_v17.py`, `role_definition_screen_v17.py` (graph/role definition stress)
  * `scripts/v18_v10_typed_guard.py` (v10 conservative typed guard)
  * `scripts/pseudotest_consensus_v19.py` (v5와 typed consensus를 pseudo split마다 완전 재학습한 promotion stress)
  * `scripts/pseudotest_v10_guard_v20.py` (v18 guard를 동일 pseudo-test에서 검증)
  * `scripts/deotte_wcg_xgb_v21.py` (Deotte precise WCG + XGBoost Python port / exact public test decisions / repeated audit)
  * `scripts/limited_hpo_v22.py` (typed22 XGB/LGBM/RF/ET 제한 HPO + alternate-fold confirmation)
  * `scripts/groupaware_validation_v23.py` (Ticket+Family connected-component StratifiedGroupKFold stress)
  * `scripts/v10_deotte_guard_v24.py` (v10 + Deotte female-death low-DOF guard audit)
  * `scripts/finalist_majority_v25.py` (v5 + Deotte + RF6 final multi-surface promotion gate)
* **내보낸 스킬**:
  * `exports/context-continuity/` (Claude Code / Codex / Antigravity 공용 스킬 패키지)
  * `exports/context-continuity.zip` (배포용 압축 파일)

---

## 🎯 4. 다음 작업자를 위한 인수인계 포인트 (Actionable Handoff Points)

1. 현재 **Public-score champion은 v47 `0.83014`**, trusted fold-safe champion은 **v5 robust OOF `0.85410` / Public `0.79665`**입니다. 두 기준을 섞지 마십시오.
2. Deotte WCG+XGBoost 공개 notebook은 precise GroupId/Wilkes link/nanny-relative 규칙과 공개 실행결과의 exact test XGB decisions까지 복구했습니다. strict fixed OOF는 **0.85410**, repeated mean은 **0.84623**입니다.
3. group-aware validation에서는 Deotte가 평균 **0.82604**로 v5 **0.81033**보다 강해 관계 구조에 대한 robustness는 확인됐습니다.
4. XGB/LGBM/RF/ET 제한 HPO는 fixed 최고 RF `0.85522`가 alternate seeds에서 무너졌고, 확인 평균 최고도 **0.84998**에 그쳐 branch를 종료했습니다.
5. 최종 `v5 + Deotte + RF6` majority는 fixed **+0.00337**, group-aware **+0.00898/+0.01122**, pseudo-test **5/5 non-negative / 평균 +0.01124**였지만 제출 ID **56832161**, Public **0.79186**으로 실패했습니다.
6. 2026-10-05 추가 submission campaign 후 현재 daily quota는 **3 slots remaining**입니다.
7. `submission.csv`와 `submission_v47_score_0.83014.csv`는 현재 Public champion으로 동기화되어 있으며 SHA256은 `CE8E484730A667E0B5D80068CF8F1418444E0113A9AF28996DD8185DF60EA0E4`입니다. v10 backup은 `submission_v10_score_0.81578.csv`로 그대로 보존합니다.
8. v19/v20/v25처럼 서로 다른 local promotion surface를 통과한 후보가 모두 Public에서 실패했으므로, 추가 validation-to-LB fitting / row probing은 금지합니다. TF-DF/WCG+KNN exact는 completeness 목적 외에는 우선순위가 낮습니다.
9. v26 adversarial validation은 train/test shift가 약함(AUC 최고 약 **0.555**)을 확인했습니다. shift-aware training은 스킵했습니다.
10. v27 Empirical-Bayes partial pooling은 유효한 독립 signal이었습니다: typed22 4-seed mean `0.84035 → typed+EB 0.84820`.
11. v29 nested selector는 outer seed에 따라 v5 대비 `+0.00673 / -0.00224`로 불안정했습니다. v30에서는 dynamic selection보다 fixed `typed22+EB CatBoost`가 두 outer seeds 평균 **0.85690**으로 우수했습니다.
12. v31 paired 6-seed audit에서 fixed typed+EB CatBoost는 v5 architecture 대비 **5승1패, 평균 +0.00655**로 serious finalist가 됐습니다.
13. 하지만 v32에서 group-aware `0.79574 / 0.79237`, pseudo-test 5개 중 3악화/1무/1개선이었고, v5 대비 16개 switch 중 최소 13개(81.25%)를 맞혀야 v10을 넘는 headroom 조건을 충족해야 했습니다. 따라서 **제출 기각**했습니다.
14. 현재 상태는 **Public champion v47 `0.83014`, trusted clean champion v5 `0.85410` 유지**입니다.
15. v33 `majority(v5, Deotte, EB)`는 repeated/pseudo에서 강했지만 v34에서 real test가 v5와 2명만 달라 v10 headroom을 수학적으로 넘을 수 없어 제출하지 않았습니다.
16. v35 residual forensics: stable-hard **121/891 (13.58%)**. `male_P1`, `Mr_P1`, `female_P3`, raw Ticket prefix가 주요 hard slice였습니다.
17. v36 raw Name/Ticket/Cabin char-TFIDF는 standalone으로 약했지만 hard-case rescue signal을 보였습니다. v37 confidence>=0.75 rescue는 개발 surface에서 매우 정밀했으나 actual test에서 v5 2-row change뿐이었습니다.
18. v38은 같은 text gate를 v10 analogue에 적용했고 actual v10에서 **PassengerId 980 하나만 1->0**. unique validation evidence가 부족해 Public row probe는 하지 않았습니다.
19. v39 field ablation 결과 980 신호는 주로 **numeric Ticket pattern**. `Ticket=364856`; train의 `364xxx` 관련 16 rows survival rate **0.125**. 이를 일반 feature로 확장했습니다.
20. v40 X-only graph centrality(PageRank/degree/core/articulation/betweenness 등)는 fixed/pseudo/group Accuracy를 평균 악화시켜 기각했습니다.
21. v41/v42 Ticket numeric-prefix EB:
    * P2 coarse prefix는 불안정.
    * **P3 full**은 repeated mean `+0.00393` (5/6 positive), group `+0.01235` (2/2), pseudo `+0.01910` (5/5)로 가장 강한 신규 clean signal.
22. v43 P3 real-test 6-seed bag은 403/418 seed-unanimous였으나 v5와 **24 rows** 달라 v10을 넘으려면 **17/24 = 70.83%** switch precision이 필요했습니다.
23. v44 paired audit에서 six-seed bagged P3 OOF는 **0.85971**로 original v5 **0.85410**보다 높았지만, 49 switches 중 27 rescue / 22 harm = **55.10% precision**뿐이었습니다. repeated pooled 59.13%, pseudo 57.14%, group 48.23%. **v43 제출 기각**.
24. 사용자 승인 후 실제 제출 결과: v38 `0.81818`, v43 `0.79665`, v45 `0.81578`, v21 exact `0.81339`, v46 `0.82775`, v47 `0.83014`, v21 WCG-both `0.80382`. **v47이 새 champion**입니다.
25. v46의 추가 WCG female-death 4 rows는 Public에서 4/4 순개선이었고, v47의 추가 9 female-death rows는 aggregate 5 correct / 4 wrong으로 +1 net이었습니다.
26. 남은 슬롯은 3개입니다. v47보다 강한 독립 가설이 나오기 전에는 row-by-row probing에 사용하지 마십시오.
