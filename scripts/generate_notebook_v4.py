"""Generate and execute the analysis-only Titanic v4 Model Zoo notebook.

The notebook consumes already-computed OOF/test artifacts. It performs no model
training and no Kaggle submission.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
V4_DIR = BASE_DIR / "exports" / "v4"


def main() -> None:
    required = [
        V4_DIR / "model_zoo_summary.csv",
        V4_DIR / "model_zoo_fold_metrics.csv",
        V4_DIR / "model_zoo_oof.csv",
        V4_DIR / "model_zoo_test.csv",
        V4_DIR / "diversity_vs_v1.csv",
        V4_DIR / "pairwise_probability_correlation.csv",
        V4_DIR / "threshold_crossfit.csv",
        V4_DIR / "blend_90_10_fold_metrics.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v4 artifacts: {missing}")

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Titanic v4 — Diverse Model Zoo & Foundation Ensemble

## 목표

v3에서 fold-safe WCG 검증 기준을 확립한 뒤, v4에서는 기존 Tree 계열만 반복 튜닝하지 않고 서로 다른 학습 원리를 가진 모델을 추가합니다.

핵심 질문은 두 가지입니다.

1. 새 모델 단독 성능이 기존 v1 champion을 이기는가?
2. 단독 성능이 조금 약하더라도 v1과 다른 실수를 해서 ensemble에 도움이 되는가?

### 이번 v4에서 실제 실행한 새 모델

- Tree/Boosting: HistGradientBoosting, AdaBoost
- Classical: LogisticRegression, RBF-SVM, KNN, LDA, QDA, GaussianNB
- Interpretable: EBM
- Foundation: TabICLv2

모든 새 모델은 기존 v1/v2와 동일한 5-Fold manifest와 fold-safe WCG를 사용합니다.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Study Guide — 왜 Model Zoo인가?

Model Zoo는 서로 다른 종류의 모델을 한곳에 모아 비교하는 실험군을 뜻합니다.

- Tree 모델은 조건 분기와 비선형 상호작용에 강합니다.
- Logistic Regression은 선형 경계를 사용합니다.
- SVM은 kernel을 통해 다른 형태의 비선형 경계를 만듭니다.
- KNN은 가까운 이웃을 참고합니다.
- EBM은 feature별 함수와 소수 interaction을 더하는 해석 가능한 boosting 모델입니다.
- TabICLv2는 사전학습된 tabular foundation model로, 현재 데이터셋의 train rows를 문맥처럼 활용해 예측합니다.

앙상블에서는 개별 점수만큼 오류의 다양성이 중요합니다. 두 모델이 항상 같은 사람을 틀리면 둘을 섞어도 얻는 것이 거의 없습니다.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = Path.cwd()
V4_DIR = BASE_DIR / 'exports' / 'v4'

summary = pd.read_csv(V4_DIR / 'model_zoo_summary.csv')
fold_metrics = pd.read_csv(V4_DIR / 'model_zoo_fold_metrics.csv')
oof = pd.read_csv(V4_DIR / 'model_zoo_oof.csv')
test = pd.read_csv(V4_DIR / 'model_zoo_test.csv')
diversity = pd.read_csv(V4_DIR / 'diversity_vs_v1.csv')
corr = pd.read_csv(V4_DIR / 'pairwise_probability_correlation.csv', index_col=0)
thresholds = pd.read_csv(V4_DIR / 'threshold_crossfit.csv')
blend_folds = pd.read_csv(V4_DIR / 'blend_90_10_fold_metrics.csv')

with open(V4_DIR / 'submission_v4_metadata.json', encoding='utf-8') as f:
    v4_meta = json.load(f)
with open(V4_DIR / 'blend_90_10_metadata.json', encoding='utf-8') as f:
    blend_meta = json.load(f)

print('OOF rows:', len(oof), '| Test rows:', len(test))
display(summary)
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 1. v4 End-to-End Architecture"))
    cells.append(
        nbf.v4.new_code_cell(
            """import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

boxes = [
    (0.3, 4.8, 2.2, 1.2, '1. v2 Features\\n41 numeric features'),
    (2.9, 4.8, 2.2, 1.2, '2. Fixed 5-Fold\\nSame manifest'),
    (5.5, 4.8, 2.2, 1.2, '3. Fold-safe WCG\\nTrain labels only'),
    (8.1, 4.8, 2.4, 1.2, '4. Model Zoo\\n10 new models'),
    (10.9, 4.8, 2.5, 1.2, '5. OOF Evaluation\\nAcc/AUC/diversity'),
    (5.5, 1.4, 2.4, 1.2, '6. TabICLv2\\nBest new single'),
    (8.3, 1.4, 2.4, 1.2, '7. 90:10 Blend\\nv1 + TabICLv2'),
    (11.1, 1.4, 2.3, 1.2, '8. Kaggle\\nPublic 0.79425'),
]
for x, y, w, h, label in boxes:
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.08', alpha=0.18, lw=1.5
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=9, fontweight='bold')

arrows = [
    ((2.5, 5.4), (2.9, 5.4)),
    ((5.1, 5.4), (5.5, 5.4)),
    ((7.7, 5.4), (8.1, 5.4)),
    ((10.5, 5.4), (10.9, 5.4)),
    ((12.15, 4.8), (6.7, 2.6)),
    ((7.9, 2.0), (8.3, 2.0)),
    ((10.7, 2.0), (11.1, 2.0)),
]
for start, end in arrows:
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.5))

plt.title('Titanic v4 - Diverse Model Zoo to Conservative Ensemble', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Model Zoo 단독 성능

Accuracy는 Titanic의 실제 제출 metric과 직접 연결됩니다.
ROC-AUC는 threshold와 무관하게 생존자와 사망자의 순위를 얼마나 잘 구분하는지 보는 진단 지표입니다.

v4의 핵심 결과는 TabICLv2가 단독으로 v1 fold-safe Accuracy와 AUC를 모두 넘어섰다는 것입니다.
"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """v1_acc = accuracy_score(oof['Survived'], oof['v1__Ensemble'] > 0.5)
v1_auc = roc_auc_score(oof['Survived'], oof['v1__Ensemble'])

plot_summary = pd.concat([
    pd.DataFrame([{'model': 'v1 Champion', 'accuracy': v1_acc, 'roc_auc': v1_auc}]),
    summary[['model', 'accuracy', 'roc_auc']]
], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=plot_summary, y='model', x='accuracy', ax=axes[0])
axes[0].set_xlim(0.65, 0.87)
axes[0].set_title('Fold-safe OOF Accuracy')
sns.barplot(data=plot_summary, y='model', x='roc_auc', ax=axes[1])
axes[1].set_xlim(0.78, 0.91)
axes[1].set_title('Fold-safe OOF ROC-AUC')
plt.tight_layout()
plt.show()

display(plot_summary.sort_values('accuracy', ascending=False).style.format({
    'accuracy': '{:.5f}', 'roc_auc': '{:.5f}'
}))
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 3. Diversity — v1이 틀린 승객을 누가 구하는가?"))
    cells.append(
        nbf.v4.new_code_cell(
            """plt.figure(figsize=(9, 5))
plt.scatter(
    diversity['probability_correlation_vs_v1'],
    diversity['v1_wrong_candidate_right'],
    s=80,
)
for _, row in diversity.iterrows():
    plt.annotate(
        row['model'],
        (row['probability_correlation_vs_v1'], row['v1_wrong_candidate_right']),
        xytext=(4, 4),
        textcoords='offset points',
        fontsize=8,
    )
plt.xlabel('Probability correlation vs v1')
plt.ylabel('v1 wrong / candidate right')
plt.title('Model Diversity vs v1 Champion')
plt.tight_layout()
plt.show()

display(diversity)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """### 해석

- correlation이 낮을수록 v1과 다른 확률 패턴을 냅니다.
- 하지만 지나치게 낮은 correlation은 단순히 모델이 약해서 생길 수도 있습니다.
- 따라서 v1이 틀리고 후보가 맞힌 수와 반대 경우를 함께 봐야 합니다.
- TabICLv2는 v1과 매우 비슷하면서도 14명을 구하고 12명을 잃어 순이익 +2를 냈습니다.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 4. Pairwise Probability Correlation"))
    cells.append(
        nbf.v4.new_code_cell(
            """plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', square=True, cbar=True)
plt.title('OOF Probability Correlation')
plt.tight_layout()
plt.show()
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 5. TabICLv2의 Fold 안정성"))
    cells.append(
        nbf.v4.new_code_cell(
            """rows = []
for fold in sorted(oof['fold'].unique()):
    mask = oof['fold'] == fold
    y_fold = oof.loc[mask, 'Survived']
    rows.extend([
        {
            'fold': int(fold),
            'model': 'v1',
            'accuracy': accuracy_score(y_fold, oof.loc[mask, 'v1__Ensemble'] > 0.5),
            'roc_auc': roc_auc_score(y_fold, oof.loc[mask, 'v1__Ensemble']),
        },
        {
            'fold': int(fold),
            'model': 'TabICLv2',
            'accuracy': accuracy_score(y_fold, oof.loc[mask, 'TabICLv2'] > 0.5),
            'roc_auc': roc_auc_score(y_fold, oof.loc[mask, 'TabICLv2']),
        },
    ])
fold_compare = pd.DataFrame(rows)

plt.figure(figsize=(9, 4.5))
sns.barplot(data=fold_compare, x='fold', y='accuracy', hue='model')
plt.ylim(0.80, 0.88)
plt.title('Fold Accuracy: v1 vs TabICLv2')
plt.tight_layout()
plt.show()

display(fold_compare)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. Threshold Audit

TabICLv2는 AUC가 매우 높기 때문에 threshold를 바꿔 Accuracy를 더 올릴 가능성을 점검했습니다.

하지만 전체 OOF에서 최적 threshold를 고른 뒤 같은 OOF를 다시 평가하면 과적합입니다.
따라서 각 fold를 평가할 때 나머지 4개 fold만 사용해 threshold를 선택했습니다.
"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """display(thresholds)
print('TabICLv2 Accuracy @0.5:', f"{v4_meta['fixed_oof_accuracy']:.5f}")
print('Cross-fitted threshold Accuracy:', f"{v4_meta['crossfit_threshold_accuracy']:.5f}")
print('Selected policy:', v4_meta['threshold_policy'], 'threshold=', v4_meta['selected_threshold'])
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 7. Conservative Ensemble: v1 90% + TabICLv2 10%"))
    cells.append(
        nbf.v4.new_code_cell(
            """v1_rows = []
for fold in sorted(oof['fold'].unique()):
    mask = oof['fold'] == fold
    v1_rows.append({
        'fold': int(fold),
        'model': 'v1',
        'accuracy': accuracy_score(oof.loc[mask, 'Survived'], oof.loc[mask, 'v1__Ensemble'] > 0.5),
    })
blend_plot = pd.concat([
    pd.DataFrame(v1_rows),
    blend_folds[['fold', 'accuracy']].assign(model='90:10 blend')
], ignore_index=True)

plt.figure(figsize=(9, 4.5))
sns.barplot(data=blend_plot, x='fold', y='accuracy', hue='model')
plt.ylim(0.82, 0.87)
plt.title('Fold Accuracy: v1 vs 90:10 Blend')
plt.tight_layout()
plt.show()

print('Blend global OOF Accuracy:', f"{blend_meta['oof_accuracy']:.5f}")
print('Blend global OOF ROC-AUC :', f"{blend_meta['oof_roc_auc']:.5f}")
print('Blend test positives      :', blend_meta['test_positive_predictions'])
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """### 왜 TabICLv2 100%보다 90:10이 Public LB에서 더 좋았나?

TabICLv2 단독은 OOF에서 매우 강했지만 test 분포에서는 기존 v1이 가지고 있던 안정적인 inductive bias를 일부 잃었습니다.

90:10 blend는 v1의 기존 판단을 거의 유지하면서 TabICLv2가 경계선 승객 한 명만 수정했습니다.
OOF에서도 그 한 번의 수정이 실제로 +1 정답이었고, Public LB에서도 개선으로 이어졌습니다.

이 결과는 새 모델이 강하니 전부 교체하는 것보다 강한 기존 챔피언을 작은 비율로 보정하는 ensemble이 작은 tabular 데이터에서 더 안정적일 수 있음을 보여줍니다.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 8. Public Leaderboard 결과"))
    cells.append(
        nbf.v4.new_code_cell(
            """public = pd.DataFrame([
    {'version': 'v1', 'public_score': 0.79186, 'note': '6-tree soft voting'},
    {'version': 'v2', 'public_score': 0.78708, 'note': 'advanced FE + weighted blend'},
    {'version': 'v4 TabICLv2', 'public_score': 0.78708, 'note': 'foundation model single'},
    {'version': 'v4b 90:10', 'public_score': 0.79425, 'note': 'v1 + TabICLv2'},
])

plt.figure(figsize=(8, 4.5))
sns.barplot(data=public, x='version', y='public_score')
plt.ylim(0.77, 0.805)
plt.title('Titanic Public Leaderboard Score by Version')
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.5f')
plt.tight_layout()
plt.show()

display(public)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 9. v4 결론과 다음 계획

### 확정 결과

- 새 Public Champion: v4b 0.79425
- 기존 v1: 0.79186
- 개선폭: +0.00239
- TabICLv2 단독 OOF: Accuracy 0.85073, ROC-AUC 0.90086
- v1 90% + TabICLv2 10% OOF: Accuracy 0.84961
- 90:10 blend는 v1 test prediction 418개 중 1개만 변경

### 다음 우선순위

1. 현재 v4b를 champion으로 동결합니다.
2. TabPFN/TabM/RealMLP 같은 새로운 family는 한 번에 하나씩 추가해 OOF 다양성을 측정합니다.
3. 45개 모델 전체를 무조건 ensemble에 넣지 않습니다.
4. 새 모델은 단독 성능 + v1/v4b와 오류 다양성 + fold 안정성을 통과해야 ensemble 후보가 됩니다.
5. Public LB는 확인용 증거로만 사용하고, 반복적인 LB probing은 하지 않습니다.
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

    notebook_path = NOTEBOOKS_DIR / "Titanic_Ensemble_Pipeline_v4.ipynb"
    with notebook_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook created: {notebook_path}")
    print("Executing analysis-only notebook (no model training / no Kaggle submission)...")
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


if __name__ == "__main__":
    main()
