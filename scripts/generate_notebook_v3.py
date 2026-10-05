"""Generate and execute the v3 analysis notebook without retraining models.

The notebook documents the completed fold-safe WCG audit, the resulting v3
candidate, and the validation contract. Long-running model fitting remains in
`scripts/audit_group_survival.py` and is intentionally not triggered here.
"""

from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_DIR = BASE_DIR / "exports" / "wcg_audit_v2"
V3_DIR = BASE_DIR / "exports" / "v3"
REQUIRED = [
    AUDIT_DIR / "cv_group_survival_summary.csv",
    AUDIT_DIR / "cv_group_survival_paired_deltas.csv",
    AUDIT_DIR / "cv_group_survival_audit.csv",
    AUDIT_DIR / "oof_fold_safe.csv",
    AUDIT_DIR / "test_group_survival_audit.csv",
    V3_DIR / "submission_v3_comparison.csv",
    V3_DIR / "submission_v3_prediction_diff.csv",
    BASE_DIR / "submissions" / "submission_v3.csv",
]

missing = [str(p) for p in REQUIRED if not p.exists()]
if missing:
    raise FileNotFoundError(
        "v3 notebook prerequisites are missing:\n- " + "\n- ".join(missing)
    )

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# 🚢 Titanic - Machine Learning from Disaster [v3]
## Fold-Safe WCG Validation Audit & Reproducible Candidate

> **v3의 목적**은 새로운 모델을 무작정 추가하는 것이 아니라, v1/v2에서 가장 강력했던 `GroupSurvival`이 교차검증 fold 경계를 넘어서 타깃 정보를 보는 문제를 먼저 교정하는 것입니다.  
> 이 노트북은 이미 완료된 **v2 WCG audit 결과**를 읽어 시각화하고, `fold_safe` 기준선을 정식 v3 후보로 기록합니다.

### 핵심 결론
1. `WCG = Woman-Child-Group`: 여성/아동 가족·티켓 동행자의 생존 신호.
2. 기존 `global_loo`: 자기 자신은 제외하지만 validation fold의 다른 동행자 라벨을 볼 수 있어 엄밀한 OOF가 아님.
3. 신규 `fold_safe`: validation/test의 WCG는 해당 fold의 **train subset 라벨만** 사용.
4. Fold-safe WCG는 WCG를 완전히 제거한 모델보다 성능이 높아, **신호 자체는 유효**함.
5. 기존 global LOO와 fold-safe의 차이는 작아, 과거 OOF가 완전히 붕괴하는 수준의 누수는 아님.

> ⚠️ 이 노트북 자체는 장시간 모델 학습을 하지 않습니다. 원본 audit을 재실행하려면 프로젝트 루트에서  
> `python scripts\\audit_group_survival.py --profile v2` 를 실행합니다.
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 1. v3 파이프라인 아키텍처

```mermaid
flowchart TD
    A[Raw Train 891 / Test 418] --> B[Target-independent Feature Engineering]
    B --> C[Stratified 5-Fold Manifest]
    C --> D[Fold Train Labels Only]
    D --> E[Fold-safe WCG / GroupSurvival]
    E --> F[6 Base Models per Fold]
    F --> G[Weighted Soft Voting]
    G --> H[OOF 891: Validation]
    F --> I[Test 418 predictions per Fold]
    I --> J[Average over 5 folds]
    J --> K[Weighted Ensemble]
    K --> L[submission_v3.csv]
```

### Study Guide: Fold와 OOF
- **Fold**: 교차검증을 위해 train을 나눈 한 조각입니다. 5-Fold에서는 4개 조각으로 학습하고 1개 조각으로 검증하는 일을 5번 반복합니다.
- **OOF (Out-Of-Fold)**: 각 승객이 **자기 자신을 학습에 사용하지 않은 모델**에게 받은 예측을 891명 전체로 다시 합친 것입니다.
- **Test prediction**: 각 fold에서 만들어진 모델이 test 418명을 예측한 뒤, 같은 모델 family의 5개 fold 확률을 평균하고 마지막에 6개 모델을 앙상블합니다.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

BASE_DIR = Path.cwd()
AUDIT_DIR = BASE_DIR / 'exports' / 'wcg_audit_v2'
V3_DIR = BASE_DIR / 'exports' / 'v3'

summary = pd.read_csv(AUDIT_DIR / 'cv_group_survival_summary.csv')
paired = pd.read_csv(AUDIT_DIR / 'cv_group_survival_paired_deltas.csv')
fold_detail = pd.read_csv(AUDIT_DIR / 'cv_group_survival_audit.csv')
oof = pd.read_csv(AUDIT_DIR / 'oof_fold_safe.csv')
test_probs = pd.read_csv(AUDIT_DIR / 'test_group_survival_audit.csv')
comparison = pd.read_csv(V3_DIR / 'submission_v3_comparison.csv')
prediction_diff = pd.read_csv(V3_DIR / 'submission_v3_prediction_diff.csv')
submission_v3 = pd.read_csv(BASE_DIR / 'submissions' / 'submission_v3.csv')

print('Audit summary rows:', len(summary))
print('OOF rows:', len(oof), '| Test rows:', len(test_probs), '| Submission rows:', len(submission_v3))
display(summary[summary['model'] == 'Ensemble'])
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 2. 렌더링된 파이프라인 구조도"))
cells.append(
    nbf.v4.new_code_cell(
        """fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

boxes = [
    (0.4, 4.8, 2.2, 1.2, '1. Base Features\\nTarget-independent'),
    (3.0, 4.8, 2.2, 1.2, '2. 5-Fold CV\\nFixed manifest'),
    (5.6, 4.8, 2.2, 1.2, '3. Fold-safe WCG\\nTrain labels only'),
    (8.2, 4.8, 2.2, 1.2, '4. 6 Models\\nRF/ET/GB/XGB/LGB/CAT'),
    (10.8, 4.8, 2.4, 1.2, '5. Weighted Blend\\nOOF validation'),
    (5.6, 1.5, 2.2, 1.2, '6. Fold Test Avg\\n418 passengers'),
    (8.2, 1.5, 2.2, 1.2, '7. v3 Probability\\nThreshold 0.5'),
    (10.8, 1.5, 2.4, 1.2, '8. submission_v3.csv\\nLocal candidate'),
]

for x, y, w, h, label in boxes:
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', alpha=0.18, lw=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')

arrows = [
    ((2.6, 5.4), (3.0, 5.4)), ((5.2, 5.4), (5.6, 5.4)),
    ((7.8, 5.4), (8.2, 5.4)), ((10.4, 5.4), (10.8, 5.4)),
    ((9.3, 4.8), (6.7, 2.7)), ((7.8, 2.1), (8.2, 2.1)),
    ((10.4, 2.1), (10.8, 2.1)),
]
for start, end in arrows:
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.5))

plt.title('Titanic v3 - Fold-Safe WCG Validation Architecture', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. Phase 0 Audit 결과: WCG는 유효한가?

세 가지 조건을 동일한 5-Fold, 동일 모델 파라미터, 동일 v2 가중치에서 비교합니다.

- `no_wcg`: `GroupSurvival=0.5` 고정 → WCG 신호 없음.
- `global_loo`: v2 기존 방식 → 자기 자신만 제외하고 전체 train label 사용.
- `fold_safe`: 새 기준 → validation/test는 fold-train label만 사용.

따라서 `fold_safe - no_wcg`는 **WCG의 실제 기여**, `global_loo - fold_safe`는 **기존 fold 누수가 부풀린 정도**에 가깝게 해석할 수 있습니다.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """ens = summary[summary['model'] == 'Ensemble'].set_index('variant')
audit_table = pd.DataFrame({
    'Accuracy': ens['accuracy'],
    'ROC-AUC': ens['roc_auc'],
}).loc[['no_wcg', 'fold_safe', 'global_loo']]

display(audit_table.style.format('{:.5f}'))

print('Fold-safe WCG contribution:')
print(f"  Accuracy: {ens.loc['fold_safe','accuracy'] - ens.loc['no_wcg','accuracy']:+.5f}")
print(f"  ROC-AUC : {ens.loc['fold_safe','roc_auc'] - ens.loc['no_wcg','roc_auc']:+.5f}")
print('Legacy global-LOO optimism relative to fold-safe:')
print(f"  Accuracy: {ens.loc['global_loo','accuracy'] - ens.loc['fold_safe','accuracy']:+.5f}")
print(f"  ROC-AUC : {ens.loc['global_loo','roc_auc'] - ens.loc['fold_safe','roc_auc']:+.5f}")
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 📊 Chart 1 — Validation variant comparison"))
cells.append(
    nbf.v4.new_code_cell(
        """plot_df = audit_table.reset_index().rename(columns={'variant': 'Variant'})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.barplot(data=plot_df, x='Variant', y='Accuracy', ax=axes[0])
axes[0].set_ylim(0.82, 0.86)
axes[0].set_title('Ensemble OOF Accuracy')
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.4f')

sns.barplot(data=plot_df, x='Variant', y='ROC-AUC', ax=axes[1])
axes[1].set_ylim(0.87, 0.91)
axes[1].set_title('Ensemble OOF ROC-AUC')
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.4f')

plt.tight_layout()
plt.show()
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 📊 Chart 2 — Fold별 WCG 기여도와 누수 영향"))
cells.append(
    nbf.v4.new_code_cell(
        """delta_long = paired.melt(
    id_vars='fold',
    value_vars=['accuracy__fold_safe_minus_no_wcg', 'accuracy__global_loo_minus_fold_safe'],
    var_name='Comparison', value_name='AccuracyDelta'
)
plt.figure(figsize=(10, 4.5))
sns.barplot(data=delta_long, x='fold', y='AccuracyDelta', hue='Comparison')
plt.axhline(0, lw=1)
plt.title('Fold-wise Accuracy Delta')
plt.ylabel('Accuracy delta')
plt.tight_layout()
plt.show()

display(paired.style.format({c: '{:+.5f}' for c in paired.columns if c != 'fold'}))
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 📊 Chart 3 — 모델별 ROC-AUC: WCG 제거 vs Fold-safe"))
cells.append(
    nbf.v4.new_code_cell(
        """model_auc = summary[summary['model'] != 'Ensemble'].copy()
model_auc = model_auc[model_auc['variant'].isin(['no_wcg', 'fold_safe'])]
plt.figure(figsize=(11, 5))
sns.barplot(data=model_auc, x='model', y='roc_auc', hue='variant')
plt.ylim(0.86, 0.91)
plt.xticks(rotation=20)
plt.title('Model-level OOF ROC-AUC')
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. Fold-safe OOF 진단

OOF probability는 각 승객이 자신을 학습에 포함하지 않은 fold model에게 받은 예측입니다.  
v3에서는 이 OOF를 최우선 검증 표면으로 사용하며 Public LB는 보조 증거로만 취급합니다.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """y = oof['Survived'].astype(int)
p = oof['fold_safe__Ensemble']
pred = (p > 0.5).astype(int)

print(f'Fold-safe OOF Accuracy: {accuracy_score(y, pred):.5f}')
print(f'Fold-safe OOF ROC-AUC : {roc_auc_score(y, p):.5f}')
print()
print('Classification report:')
print(classification_report(y, pred, target_names=['Perished (0)', 'Survived (1)']))
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 📊 Chart 4 — OOF probability distribution"))
cells.append(
    nbf.v4.new_code_cell(
        """prob_df = pd.DataFrame({'Probability': p, 'Actual': y.map({0: 'Perished', 1: 'Survived'})})
plt.figure(figsize=(10, 4.5))
sns.histplot(data=prob_df, x='Probability', hue='Actual', bins=25, stat='density', common_norm=False, element='step')
plt.axvline(0.5, linestyle='--', lw=1.5)
plt.title('Fold-safe OOF Probability Distribution')
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 📊 Chart 5 — OOF confusion matrix"))
cells.append(
    nbf.v4.new_code_cell(
        """cm = confusion_matrix(y, pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cbar=False,
            xticklabels=['Perished', 'Survived'], yticklabels=['Perished', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Fold-safe Ensemble OOF Confusion Matrix')
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. v3 Local Candidate

현재 `submission_v3.csv`는 audit에서 이미 계산된 `fold_safe__Ensemble` test probability를 **0.5 threshold**로 이진화한 후보입니다.

중요한 점:
- 모델을 새로 튜닝해서 얻은 후보가 아니라, **검증 방법을 바로잡은 후보**입니다.
- 기존 v1/v2 Public 제출은 그대로 보존합니다.
- 이 노트북은 Kaggle 제출을 수행하지 않습니다. 외부 제출은 별도의 명시적 결정 후에만 수행합니다.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """print('v3 positive predictions:', int(submission_v3['Survived'].sum()), '/', len(submission_v3))
display(comparison)

changed = prediction_diff.copy()
mask_cols = [c for c in ['diff_vs_v1', 'diff_vs_v2'] if c in changed.columns]
if mask_cols:
    changed = changed[changed[mask_cols].any(axis=1)]
display(changed)
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 📊 Chart 6 — v3와 기존 제출의 prediction 차이"))
cells.append(
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
sns.barplot(data=comparison, x='reference', y='different_predictions')
plt.title('Number of Test Predictions Changed by v3')
plt.xlabel('Reference submission')
plt.ylabel('Changed passengers')
for container in plt.gca().containers:
    plt.gca().bar_label(container)
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 6. 해석 및 다음 실험

### 현재 판단
- `fold_safe`가 `no_wcg`보다 확실히 높으므로 **WCG는 유지**합니다.
- `global_loo`와 `fold_safe` 차이가 작으므로 과거 v2의 높은 AUC가 전부 누수 때문이라고 보기는 어렵습니다.
- 하지만 앞으로의 모든 실험은 `fold_safe` 검증을 사용해야 합니다.

### 다음 순서
1. **v1 fold-safe audit**을 실행해 v1도 동일한 신뢰 기준으로 맞춥니다.
2. v1/v2의 fold-safe OOF/test probability를 확보한 뒤 **하이브리드 blend**를 동일 검증 표면에서 비교합니다.
3. 그 이후에만 제한적 HPO와 cross-fitted stacking을 시도합니다.
4. threshold 최적화는 전체 OOF에 맞춘 뒤 같은 OOF를 재평가하지 않고, 별도 cross-fit 또는 고정 후보군으로 검증합니다.

### 재현 명령 (PowerShell)
```powershell
Set-Location "H:\\kaggle\\practice\\Titanic - Machine Learning from Disaster"

# 장시간/모델 학습: 필요할 때 사용자가 직접 실행
python .\\scripts\\audit_group_survival.py --profile v2

# 학습 없음: audit 결과에서 v3 후보 생성
python .\\scripts\\build_v3_candidate_from_audit.py

# 학습 없음: 이 분석 노트북 다시 생성/실행
python .\\scripts\\generate_notebook_v3.py
```
"""
    )
)

nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3"}

notebook_path = NOTEBOOKS_DIR / "Titanic_Ensemble_Pipeline_v3.ipynb"
with notebook_path.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook created: {notebook_path}")
print("Executing analysis-only notebook (no model training)...")
client = NotebookClient(
    nb,
    timeout=180,
    kernel_name="python3",
    resources={"metadata": {"path": str(BASE_DIR)}},
)
client.execute()

with notebook_path.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Executed notebook saved: {notebook_path}")
