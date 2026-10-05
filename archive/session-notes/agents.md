> HISTORICAL SESSION NOTES. Claims, scores and stopping decisions below reflect that stage, not a leakage-free certification. See ../../docs/03-validation-and-integrity.md for the final retrospective. Local-machine links may no longer resolve.

# 🤖 AGENT INSTRUCTIONS & CONTEXT GUIDE

> **To any AI coding assistant / Antigravity agent working on this repository:**  
> 이 문서는 본 프로젝트의 맥락, 작업 규약, 버전 관리 방식, 파일 역할, 주의 사항을 정의합니다. **작업을 시작하기 전 반드시 아래 문서들을 순서대로 읽고 맥락을 파악하십시오.**

---

## 📚 1. 필수 선독 문서 (Required Reading Order)

새로운 태스크나 실험을 진행하기 전, 반드시 다음 4개 문서를 읽으십시오:

1. **[discoveries.md](file:///h:/kaggle/practice/Titanic%20-%20Machine%20Learning%20from%20Disaster/discoveries.md)**:
   * **필독 이유**: 실패 원인(타깃 결측치 누수, WCG 정수 캐스팅 오류)과 성공 요인(1인당 요금, Leave-One-Out WCG, 가중 앙상블), 통계적 Shake-up 해석이 정리되어 있습니다.
2. **[handoff.md](file:///h:/kaggle/practice/Titanic%20-%20Machine%20Learning%20from%20Disaster/handoff.md)**:
   * **필독 이유**: 버전별 캐글 제출 이력과 현재 champion(v5 robust hard vote: 0.79665), 현재 시스템 상태, 파일 위치 및 환경 설정이 기록되어 있습니다.
3. **[plan.md](file:///h:/kaggle/practice/Titanic%20-%20Machine%20Learning%20from%20Disaster/plan.md)**:
   * **필독 이유**: v1/v2 fold-safe audit, v4/v5 champion, v6/v7 feature/preprocessing/HPO 감사와 v8 TabPFN v3/v2.5 finalist 계획이 정리되어 있습니다.
4. **[README.md](file:///h:/kaggle/practice/Titanic%20-%20Machine%20Learning%20from%20Disaster/README.md)**:
   * **필독 이유**: 전체 프로젝트의 공식 개요, 아키텍처, 성능 표가 요약되어 있습니다.

---

## 🏷️ 2. 버전 관리 규약 (Versioning Protocol: v1, v2, v3...)

본 프로젝트의 모든 모델링 및 제출 작업은 **v1, v2, v3...** 형식으로 체계적으로 관리합니다:

1. **노트북 파일 명명**:
   * 각 버전의 독립적인 주피터 노트북 생성: `notebooks/Titanic_Ensemble_Pipeline_v<N>.ipynb`
   * 예: `Titanic_Ensemble_Pipeline_v1.ipynb`, `Titanic_Ensemble_Pipeline_v2.ipynb`
2. **스크립트 파일 명명**:
   * 각 버전을 재현하는 생성 스크립트: `scripts/generate_notebook_v<N>.py`
3. **제출 파일 명명**:
   * 각 버전별 제출 파일: `submissions/submission_v<N>.csv`
   * 제출 점수가 확정되면 백업: `submissions/submission_v<N>_score_<SCORE>.csv`
   * 최신 버전은 `submissions/submission.csv`에도 동기화
4. **시각화 및 다이어그램 의무화**:
   * **모든 버전의 주피터 노트북에는 반드시 엔드투엔드 파이프라인 아키텍처 다이어그램(Mermaid 및 matplotlib 렌더링 플로우차트)과 4종 이상의 EDA 및 모델 진단 차트(ROC 곡선, 혼동 행렬, 피처 중요도, 확률 분포)를 포함해야 합니다.**
5. **용어 설명 및 원리 해설 의무화 (Study Guide)**:
   * 사용자의 학습 목적을 위해, 각 단계마다 사용된 머신러닝 기법(왜도와 Log변환, Bagging vs Boosting, LOO 누수 방지, Soft Voting, ROC-AUC 등)의 **수학적/도메인적 원리와 개념 해설을 마크다운으로 상세히 작성**해야 합니다.

---

## ⚖️ 3. 핵심 개발 및 모델링 원칙 (Core Principles)

### 1) 데이터 누수(Data Leakage) 엄격 금지
* 전체 데이터(`train + test`)를 합쳐 전처리할 때는 인덱스 정보 및 피처 변환에만 사용하십시오.
* **학습 직전에는 반드시 `Survived.notnull()`인 891건만 분리하여 교차 검증을 수행해야 합니다.**
* 타깃이 `NaN`인 테스트 데이터가 훈련 세트나 검증 세트에 섞여 들어가지 않도록 극도로 주의하십시오.

### 2) 5-Fold Stratified K-Fold CV (OOF) 표준 준수
* 단일 Train/Val Split(8:2 등)을 사용하지 마십시오. 데이터셋 크기가 891건으로 작아 과적합(Overfitting) 위험이 매우 큽니다.
* 모델 평가는 항상 **5-Fold Stratified K-Fold의 Out-Of-Fold(OOF) Accuracy와 ROC-AUC**를 기준으로 측정하십시오.

### 3) Leave-One-Out 기반 그룹 피처 유지
* 가족/티켓 그룹 생존 피처(WCG)를 다룰 때는 자기 자신 제외뿐 아니라 **validation/test가 오직 해당 fold-train의 라벨만 참조하도록 fold-safe하게 계산**하십시오.
* 확률값(0.5 등)을 강제로 정수형(`int`)으로 변환하여 신호를 유실시키지 마십시오.

### 4) `shepsci/kaggle-skill` 사전 검증 필수
* 캐글에 제출하기 전, 반드시 설치된 스킬의 `validate` 명령어로 무결성 검사를 수행하십시오:
  ```bash
  python C:/Users/Taeyang/.gemini/config/skills/kaggle/scripts/kaggle_skill.py validate titanic submissions/submission_v<N>.csv
  ```

---

## 🛠️ 4. 주요 명령어 치트시트 (Cheat Sheet)

```powershell
# 1. v7 representation / preprocessing audit
python scripts/native_catboost_v7.py
python scripts/preprocessing_audit_v7.py

# 2. selective HPO / residual analysis
python scripts/catboost_hpo_v7.py
python scripts/disagreement_audit_v7.py
python scripts/rulefit_hpo_v7.py

# 3. v7 analysis notebook
python scripts/generate_notebook_v7.py

# 4. TabPFN finalist harness (v2 works now; v2.5/v3 need one-time license/auth)
python scripts/tabpfn_finalist_v8.py --models TabPFN_v2 --variants all
```
