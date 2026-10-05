> HISTORICAL SESSION NOTES. Claims, scores and stopping decisions below reflect that stage, not a leakage-free certification. See ../../docs/03-validation-and-integrity.md for the final retrospective. Local-machine links may no longer resolve.

# 🚢 Titanic - Machine Learning from Disaster
> **Kaggle Competition Solution & Study Guide (v1 ~ v47)**  
> **현재 Kaggle Public Score Champion: 0.83014 (v47 v38 + broader Deotte female-death guard)**  
> **Trusted fold-safe Champion: v5 robust hard vote / Public 0.79665 / OOF Accuracy 0.85410**

---

## 📌 1. 프로젝트 개요

본 프로젝트는 타이타닉 침몰 당시의 탑승객 데이터를 활용하여 생존 여부(`Survived`: 0 또는 1)를 예측하는 Kaggle 대표 대회 솔루션입니다.  
단순한 코드 실행을 넘어, **머신러닝 핵심 원리와 도메인 지식, 통계적 평가 기법을 체계적으로 학습하고 재현할 수 있는 교육용 파이프라인**으로 설계되었습니다.

---

## 🏷️ 2. 버전별 성과 및 특징 비교 (Version Ledger)

| 버전 (Version) | 핵심 개선 내용 | CV Accuracy | CV ROC-AUC | Public Score | 상태 | 노트북 파일 |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **v1** | • 전체 데이터 분할 버그(NaN 타깃 유입) 수정<br>• WCG Leave-One-Out 기초 피처 도입<br>• 6대 트리 모델 균등 Soft Voting | **0.84848** *(fold-safe)* | **0.88914** *(fold-safe)* | **`0.79186`** | 완료 | [`Titanic_Ensemble_Pipeline_v1.ipynb`](./notebooks/Titanic_Ensemble_Pipeline_v1.ipynb) |
| **v2** | • **1인당 실제 요금(`FarePerPerson`)** 계산<br>• 티켓 접두사 및 나이-등급 상호작용 피처<br>• **ROC-AUC 비례 Weighted Blending**<br>• WCG Post-Processing<br>• 다이어그램/진단 차트/학습용 해설 | **0.84175** *(fold-safe)* | **0.89730** *(fold-safe)* | `0.78708` | 완료 | [`Titanic_Ensemble_Pipeline_v2.ipynb`](./notebooks/Titanic_Ensemble_Pipeline_v2.ipynb) |
| **v3** | • **WCG fold leakage audit**<br>• validation/test WCG를 fold-train label만으로 생성<br>• fold-safe OOF 기준선 확정<br>• 기존 audit 확률에서 로컬 제출 후보 생성<br>• 6종 진단 차트 및 Fold/OOF 해설 | **0.84175** | **`0.89730`** | 미제출 | **로컬 후보 완료** | [`Titanic_Ensemble_Pipeline_v3.ipynb`](./notebooks/Titanic_Ensemble_Pipeline_v3.ipynb) |
| **v4** | • **Diverse Model Zoo** 도입<br>• EBM / Classical / TabICLv2 비교<br>• 동일 fold-safe WCG + 동일 5-Fold manifest<br>• v1 90% + TabICLv2 10% 보수적 blend | **0.84961** *(blend)*<br>TabICLv2 단독 **0.85073** | 0.89004 *(blend)*<br>TabICLv2 단독 **0.90086** | **`0.79425`** | **★ Public Champion** | [`Titanic_Ensemble_Pipeline_v4.ipynb`](./notebooks/Titanic_Ensemble_Pipeline_v4.ipynb) |
| **v5** | • **Aggressive Multi-Family Model Zoo**<br>• RuleFit / FIGS / TabPFN v2 / TabM / RealMLP / TabR / FT-Transformer / MLP-PLR / RTDL / xRFM<br>• MLP-PLR 3-seed 안정성 감사<br>• v4b + RuleFit + 3-seed MLP 평균 **2-of-3 Robust Hard Vote** | **0.85522** *(seed42 vote)*<br>**0.85410** *(seed-robust vote)* | hard-vote AUC는 이산 vote score라 참고용 | **`0.79665`** | **★ Public Champion** | [`Titanic_Ensemble_Pipeline_v5.ipynb`](./notebooks/Titanic_Ensemble_Pipeline_v5.ipynb) |
| **v6** | • 공개 Titanic notebook/discussion 기반 전처리/FE 조사<br>• Cabin/Missing/Relational/Interaction/WCG block ablation<br>• **LastName + Fare** family-survival fold-safe 재구현<br>• selected feature / stacking probe | FamilyFare panel **0.84624**<br>FamilyFare GB **0.84961** | FamilyFare panel **0.90356** | 미제출 | **연구 완료 / 챔피언 유지** | [`Titanic_Ensemble_Pipeline_v6.ipynb`](./notebooks/Titanic_Ensemble_Pipeline_v6.ipynb) |
| **v7** | • CatBoost native categorical audit<br>• 가족/티켓 기반 Age/Deck 보간 ablation<br>• AdultMale CatBoost 제한적 HPO + alternate-fold stress<br>• champion disagreement slice audit<br>• RuleFit 제한적 HPO / seed stability | native-cat 최고 **0.84512**<br>CatBoost HPO 최고 **0.85073**<br>RuleFit vote 최고 **0.85410** | CatBoost HPO 최고 약 **0.89860** | 미제출 | **연구 완료 / 챔피언 유지** | Titanic_Ensemble_Pipeline_v7.ipynb |
| **v10** | • Gunes 원본 26-feature leaderboard pipeline 재현<br>• RF 1750 trees + 5-fold test averaging | **0.83614** *(원본식 CV; leakage 주의)* | **0.89311** | **0.81578** | **★ Public Score Champion** | - |
| **v11** | • Gunes exact 26-feature를 fold-safe target encoding으로 재작성<br>• documented RF variants 비교 | 최고 **0.83502** | 최고 **0.87771** | 미제출 | **감사 완료** | - |
| **v12** | • TabPFN v2.5/v3 정상 fit 확인<br>• FamilyFare / Gunes rates / structural features 재학습<br>• ensemble/stacking probe | TabPFN 단독 최고 **0.84848**<br>trusted ensemble 최고 **0.85410** | TabPFN FE 최고 **0.90068** | 미제출 | **후보 준비 / trusted champion 동률** | - |
| **v13** | • v10 test-aware Family/Ticket 메커니즘 분해<br>• fold-safe transductive analogue 생성<br>• v10-v5 충돌 22명 peer evidence audit<br>• conservative guard selector | transductive analogue **0.83277**<br>best guard analogue **0.83838** | analogue AUC **0.88029** | 미제출 | **2-row guard 후보 준비** | - |
| **v14-v17** | • 실제 test relation geometry 기반 pseudo-test 구축<br>• typed22 WomanChild/AdultMale relation rates<br>• FamilyFare + CatBoost/TabPFN<br>• alpha/graph/role stress | typed22 pseudo **0.85618**<br>FamilyFare+typed Cat **0.85297** | typed22 pseudo AUC **0.90260** | 미제출 | **관계 신호 확인** | - |
| **v19** | • v5 robust와 typed members를 5개 pseudo-test에서 전부 재학습<br>• strict unanimous typed consensus promotion test | fixed OOF **0.85859**<br>pseudo delta vs v5 **+0.00225** | - | **0.79425** | **Public 실패** | - |
| **v20** | • v10 analogue + typed consensus conservative guard<br>• fixed/pseudo 양쪽 non-negative rule만 승격 | pseudo `all3_weak` delta **+0.00562** | - | **0.81100** | **v10 미달 / probing 중단** | - |
| **v21** | • Chris Deotte WCG+XGBoost 공개 R notebook 메커니즘 포트<br>• precise GroupId / nanny-relative / Wilkes-Hocking link<br>• 공개 실행결과의 exact XGB 변경 승객 복구<br>• strict fold-safe + repeated stress | fixed **0.85410**<br>repeated mean **0.84623** | - | 미제출 | **historical exact 후보 + clean audit 완료** | - |
| **v22** | • FamilyFare+typed22에서 XGB/LGBM/RF/ET 제한 HPO<br>• fixed top2만 seed 123/777 confirmation | RF fixed 최고 **0.85522**<br>확인 평균 최고 RF6 **0.84998** | - | 미제출 | **fixed-fold HPO 과적합 확인 / 종료** | - |
| **v23** | • Ticket+Family 연결요소 단위 StratifiedGroupKFold stress<br>• v5 / Deotte / typed Cat / RF6 재학습 비교 | Deotte mean **0.82604**<br>v5 **0.81033** | Deotte score AUC **0.80310** | 미제출 | **Deotte group robustness 확인** | - |
| **v24** | • v10 보존 + Deotte female-death 저자유도 guard<br>• fixed + group-aware 2 seeds 비교 | 최고 rule도 한 stress seed에서 악화 | - | 미제출 | **guard 기각** | - |
| **v25** | • v5 robust + exact Deotte WCG/XGB + alternate-confirmed typed RF6 2-of-3 majority<br>• fixed/group-aware/pseudo-test 최종 promotion gate | fixed **0.85746**<br>group **0.81930 / 0.82155**<br>pseudo mean **0.86067** | - | **0.79186** | **최종 제출 실패 / v10 유지** | - |

> 💡 **Validation Audit 핵심 결과**:  
> WCG를 제거한 `no_wcg` 대비 `fold_safe`는 Accuracy **+0.00786**, ROC-AUC **+0.01352** 개선되었습니다. 반면 기존 `global_loo`가 `fold_safe`보다 높았던 폭은 Accuracy **+0.00112**, ROC-AUC **+0.00140**에 그쳤습니다. 따라서 **WCG 신호는 실제로 유효하지만 이후 평가는 반드시 fold-safe 방식으로 수행**합니다.
>
> **v4 Model Zoo 핵심 결과**: TabICLv2는 단독 OOF Accuracy **0.85073**, ROC-AUC **0.90086**으로 기존 fold-safe v1(0.84848 / 0.88914)을 넘어섰습니다. 다만 Public Score는 0.78708이었고, v1 확률 90% + TabICLv2 10%의 보수적 blend가 test 예측을 1명만 바꾸면서 Public **0.79425**를 기록해 새 챔피언이 되었습니다.
>
> **v5 핵심 결과**: MLP-PLR seed42 단독은 Accuracy **0.85297**이지만 seed 142/242에서 **0.84063 / 0.84287**로 내려가 seed 민감성이 확인되었습니다. 반면 **v4b + RuleFit + MLP-PLR** 2-of-3 hard vote는 MLP seed를 바꿔도 약 **0.85073~0.85522** 범위로 유지됐고, 3-seed 평균 MLP를 사용한 robust vote는 **0.85410**을 기록했습니다. 이 robust 후보는 Kaggle Public **0.79665**로 기존 v4b **0.79425**를 넘어 새 챔피언이 되었습니다.
>
> **v6 핵심 결과**: 공개 notebook의 일반적인 cabin/band/interaction feature를 대량 추가하면 오히려 과적합이 커졌습니다. 반면 공개 Family Survival 계열에서 쓰이는 **`LastName + Fare` 가족 키**를 fold-safe하게 재구현하자 3-model panel이 **0.84287 → 0.84624**, ROC-AUC가 **0.89786 → 0.90356**으로 개선되었습니다. 다만 v5 robust OOF Accuracy **0.85410**은 넘지 못해 제출하지 않았습니다.
>
> **v7 핵심 결과**: native categorical CatBoost, 가족/티켓 기반 Age/Deck 보간, CatBoost/RuleFit 제한적 HPO를 모두 검증했지만 v5 robust **0.85410**을 안정적으로 넘지 못했습니다. Large-family FamilyFare override는 OOF +2 correct였지만 개선이 한 fold에만 몰려 기각했습니다. 따라서 다음 최우선 축은 **TabPFN v3/v2.5**입니다.
>
> **v11/v12 핵심 결과**: Gunes exact 26-feature를 fold-safe하게 바꾸면 Accuracy가 **0.83502 / 0.82941**로 내려가 v10의 강한 Public 성능이 historical global target encoding에 상당히 의존함을 확인했습니다. TabPFN v2.5/v3는 이제 실제 fit까지 정상 작동하며 FamilyFare 또는 Gunes-rate feature로 단독 Accuracy **0.84848**까지 상승했습니다. `RuleFit + MLP seed-mean + TabPFN v3 FamilyFare` hard vote는 trusted v5와 OOF **0.85410 동률**이며 test 예측은 1명만 다릅니다.
>
> **v13 핵심 결과**: v10-v5가 다른 test 승객은 **22명**이며, 11명은 Family와 Ticket 양쪽의 train-peer 생존 신호를 모두 보유하고, 3명은 한쪽만, 8명은 관계 신호가 없습니다. v10 메커니즘을 fold-safe transductive CV로 흉내 내면 **0.83277**로 v5보다 낮아 local CV가 Public 구조를 잘 대변하지 못함을 확인했습니다. 가장 보수적인 후보는 **관계 신호가 없고 v5+TabPFN이 반대에 동의하는 2명(929, 1231)만 v10에서 되돌리는 guard**이며 제출 파일만 준비하고 아직 제출하지 않았습니다.
>
> **v14-v20 핵심 결과**: 실제 test의 Family/Ticket overlap과 covariate 분포를 맞춘 pseudo-test에서 typed22가 `legacy2` 대비 Accuracy **+0.01348**, AUC **+0.01424** 개선됐습니다. 하지만 fixed OOF **0.85859**와 pseudo 개선을 모두 통과한 strict consensus가 실제 Public에서는 **0.79425**, v10을 6명만 수정한 conservative guard도 **0.81100**에 그쳤습니다. 따라서 pseudo-test는 관계형 feature의 신호 확인에는 유효하지만 현재 actual test의 promotion surface로는 불충분하며, Public `0.81578`에 맞춘 추가 split/row probing은 중단합니다.
>
> **v21-v25 핵심 결과**: Deotte WCG+XGBoost를 공개 R source와 실행 출력 기준으로 재구성하고, repeated/group-aware/limited-HPO까지 남은 고우선순위 축을 소진했습니다. Deotte는 group-aware Accuracy **0.82604**로 v5 **0.81033**보다 강했고, 최종 `v5 + Deotte + RF6` majority는 fixed **+0.00337**, group-aware **+0.00898 / +0.01122**, matched pseudo-test **5/5 non-negative / 평균 +0.01124**를 기록했습니다. 그럼에도 실제 Public은 **0.79186**이었습니다. 따라서 이 프로젝트에서는 local validation을 더 복잡하게 Public `0.81578`에 맞추는 작업을 중단하고 **v10을 Public champion, v5를 trusted clean champion으로 유지**합니다.

---

## 📂 3. 폴더 구조

```text
Titanic - Machine Learning from Disaster/
├── data/                                         # 💾 데이터셋
│   ├── train.csv                                 # 훈련 데이터 (891 rows)
│   ├── test.csv                                  # 테스트 데이터 (418 rows)
│   └── gender_submission.csv                     # 기준 샘플 파일
│
├── notebooks/                                    # 📓 주피터 노트북
│   ├── Titanic_Ensemble_Pipeline_v7.ipynb        # [v7 최신] preprocessing audit, selective HPO, residual analysis
│   ├── Titanic_Ensemble_Pipeline_v6.ipynb        # [v6 최신] 공개 FE 조사, family-fare audit, stacking probe
│   ├── Titanic_Ensemble_Pipeline_v5.ipynb        # [v5 챔피언] 확장 Model Zoo, seed audit, hard/robust vote
│   ├── Titanic_Ensemble_Pipeline_v4.ipynb        # [v4 보관] TabICLv2, diversity, 90:10 Public champion
│   ├── Titanic_Ensemble_Pipeline_v3.ipynb        # [v3 보관] Fold-safe WCG audit, OOF 진단
│   ├── Titanic_Ensemble_Pipeline_v2.ipynb        # [v2 보관] Advanced FE + Weighted Blending
│   ├── Titanic_Ensemble_Pipeline_v1.ipynb        # [v1 보관] 0.79186 점수 달성 노트북
│   └── archive/                                  # 과거 레거시 노트북 보관소
│
├── submissions/                                  # 🚀 캐글 제출 파일
│   ├── submission_v1_score_0.79186.csv           # v1 제출본 백업 (Score 0.79186)
│   ├── submission_v2.csv                         # v2 제출본 (Score 0.78708)
│   ├── submission_v3.csv                         # v3 fold-safe 로컬 후보 (미제출)
│   ├── submission_v4.csv                         # TabICLv2 단독 제출 (Score 0.78708)
│   ├── submission_v4_blend_90_10.csv             # v1 90% + TabICLv2 10% 제출
│   ├── submission_v4_score_0.79425.csv           # v4 Public Champion 백업
│   ├── submission_v5.csv                         # MLP-PLR seed42 로컬 후보 (미제출)
│   ├── submission_v5_hard_vote.csv               # seed42 3-model hard vote (미제출)
│   ├── submission_v5_robust_vote.csv             # 3-seed MLP robust hard vote (Public 0.79665)
│   ├── submission_v5_score_0.79665.csv            # v5 Public Champion 백업
│   └── submission.csv                            # 현재 champion과 동기화
│
├── scripts/                                      # ⚙️ 파이프라인 자동화 스크립트
│   ├── generate_notebook.py                      # v1 생성 스크립트
│   ├── generate_notebook_v2.py                   # v2 생성 및 셀 자동 실행 스크립트
│   ├── audit_group_survival.py                   # v1/v2 WCG fold leakage audit
│   ├── build_v3_candidate_from_audit.py          # audit 결과에서 v3 로컬 후보 생성
│   ├── generate_notebook_v3.py                   # v3 분석 노트북 생성/실행 (재학습 없음)
│   ├── check_v4_environment.py                   # v4 모델/패키지/GPU 환경 inventory
│   ├── model_zoo_screen_v4.py                    # fold-safe Model Zoo OOF/test 생성
│   ├── build_v4_candidate.py                     # TabICLv2 threshold audit + v4 후보
│   ├── build_v4_blend_candidate.py               # v1 90% + TabICLv2 10% 보수적 blend
│   ├── generate_notebook_v4.py                   # v4 분석 노트북 생성/실행
│   ├── model_zoo_screen_v5.py                    # resumable multi-family v5 screen
│   ├── build_v5_candidate.py                     # MLP-PLR candidate + threshold audit
│   ├── build_v5_hard_vote_candidate.py           # seed42 3-model hard vote
│   ├── probe_mlp_plr_seeds_v5.py                 # MLP-PLR seed 안정성 audit
│   ├── build_v5_robust_vote_candidate.py         # 3-seed 평균 MLP robust vote
│   ├── generate_notebook_v5.py                   # v5 분석 노트북 생성/실행
│   ├── feature_ablation_v6.py                    # 공개 아이디어 feature block ablation
│   ├── targeted_feature_screen_v6.py             # 논리 feature group 단위 정밀 스크린
│   ├── group_key_audit_v6.py                     # LastName+Fare / Ticket / Cabin group-survival audit
│   ├── train_selected_v6_models.py               # 모델별 selected feature 재학습
│   ├── train_familyfare_modern_v6.py             # RuleFit / MLP에 FamilyFare 신호 주입
│   ├── stacking_probe_v6.py                      # cross-fitted meta-model probe
│   ├── generate_notebook_v6.py                   # v6 분석 노트북 생성/실행
│   ├── native_catboost_v7.py                     # native categorical representation audit
│   ├── preprocessing_audit_v7.py                 # family/ticket Age + Deck imputation audit
│   ├── catboost_hpo_v7.py                        # limited CatBoost HPO + alternate-fold stress
│   ├── disagreement_audit_v7.py                  # champion residual / slice analysis
│   ├── rulefit_hpo_v7.py                         # limited RuleFit HPO + seed stability
│   ├── generate_notebook_v7.py                   # v7 분석 노트북 생성/실행
│   ├── tabpfn_finalist_v8.py                     # TabPFN v2/v2.5/v3 + FamilyFare finalist harness
│   └── gunes_feature_audit_v9.py                 # Gunes Evitan tutorial ideas, fold-safe transfer audit
│
├── exports/                                      # 📦 audit / v3 분석 아티팩트 및 에이전트 스킬
│   ├── wcg_audit_v2/                             # no_wcg/global_loo/fold_safe CV/OOF/test 결과
│   ├── v3/                                       # v3 비교표, 확률, validation summary
│   ├── v4/                                       # Model Zoo OOF/test, diversity, threshold, blend 결과
│   ├── v5/                                       # expanded zoo, per-model cache, seed probes, vote diagnostics
│   ├── v6/                                       # public FE research, ablations, FamilyFare, stacking probes
│   ├── v7/                                       # preprocessing / HPO / residual-audit artifacts
│   ├── v8/                                       # TabPFN finalist artifacts / license-block notes
│   ├── v9/                                       # Gunes tutorial transfer audit artifacts
│   ├── context-continuity/                       # Claude Code / Codex / Antigravity 공용 스킬
│   └── context-continuity.zip                    # 스킬 배포용 압축 파일
│
├── agents.md                                     # 🤖 에이전트 지침 및 버전 관리 규약
├── discoveries.md                                # 💡 실패/성공 원인 및 머신러닝 원리 노트
├── handoff.md                                    # 🤝 세션 히스토리, 점수 레저, 인수인계
├── plan.md                                       # 🗺️ v7 결과 및 TabPFN 다음 로드맵
└── README.md                                     # 📖 프로젝트 종합 설명서
```

---

## 💡 4. 핵심 머신러닝 개념 및 도메인 원리 요약

1. **데이터 누수 (Data Leakage), Leave-One-Out, Fold-safe Target Feature**:
   * 자기 자신의 생존 여부를 제외하는 LOO만으로는 충분하지 않습니다. validation fold의 다른 동행자 라벨도 볼 수 있기 때문에, supervised 그룹 피처는 **각 fold의 train subset 라벨만 사용해 validation/test에 적용**해야 합니다.
2. **왜도(Skewness)와 Log1p 변환**:
   * 오른쪽 꼬리가 긴 요금(`Fare`)에 $y = \log(1 + x)$를 적용하여 수치 간격을 압축하고 극단치(Outlier)에 대한 민감도를 줄입니다.
3. **1인당 실제 요금 (`FarePerPerson = Fare / TicketFreq`)**:
   * 티켓 1장의 요금은 대가족 전체의 합산 금액이므로, 이를 티켓 공유자 수로 나누어 3등석 대가족의 요금 착시를 제거합니다.
4. **배깅(Bagging) vs 부스팅(Boosting)**:
   * **Bagging (Random Forest, Extra Trees)**: 여러 트리를 독립적으로 학습시켜 평균함으로써 분산(Variance)을 줄임.
   * **Boosting (GB, XGBoost, LightGBM, CatBoost)**: 이전 트리의 오차를 순차적으로 줄여나가며 편향(Bias)을 줄임.
5. **Soft Voting vs Weighted Blending**:
   * 다수결 대신 각 모델의 예측 확률(0.0~1.0)을 취하고, ROC-AUC가 뛰어난 모델에 더 큰 가중치(CatBoost 25%, GB 20%, XGB 20% 등)를 부여하여 오차를 상쇄합니다.

---

## 🛠️ 5. v6 재현 명령어

```powershell
# 공개 feature block / 개별 feature / family-key 감사
python .\scripts\feature_ablation_v6.py
python .\scripts\targeted_feature_screen_v6.py
python .\scripts\group_key_audit_v6.py

# 모델 직접 주입 및 meta-model probe
python .\scripts\train_selected_v6_models.py
python .\scripts\train_familyfare_modern_v6.py
python .\scripts\stacking_probe_v6.py

# v6 분석 노트북
python .\scripts\generate_notebook_v6.py
```
