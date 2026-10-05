"""Generate and execute the analysis-only Titanic v5 Model Zoo notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
V5_DIR = BASE_DIR / "exports" / "v5"


def main() -> None:
    required = [
        V5_DIR / "model_zoo_summary.csv",
        V5_DIR / "diversity_vs_v4b.csv",
        V5_DIR / "pairwise_probability_correlation.csv",
        V5_DIR / "threshold_crossfit.csv",
        V5_DIR / "hard_vote_fold_metrics.csv",
        V5_DIR / "robust_vote_fold_metrics.csv",
        V5_DIR / "seed_probes" / "seed_summary.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v5 artifacts: {missing}")

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Titanic v5 — Aggressive Multi-Family Model Zoo

v4에서 TabICLv2를 추가해 Public Score 0.79425를 만든 뒤, v5에서는 모델 family를 과감하게 넓혔습니다.

이번 핵심은 모델 수 자체가 아니라 서로 다른 학습 원리를 가진 모델의 OOF 오류 패턴을 비교하는 것입니다.

추가한 모델:
- Foundation: TabPFN v2
- Rule / Interpretable: RuleFit, FIGS
- Modern DL: TabM, RealMLP, MLP-PLR, RTDL-MLP, RTDL-ResNet
- Retrieval DL: TabR
- Transformer: FT-Transformer
- Classical NN: sklearn-style MLP
- Kernel / Feature: xRFM

TabPFN v3 / v2.5는 패키지까지 설치됐지만 Prior Labs의 1회 라이선스 승인/인증이 필요해 현재 자동 실행이 보류되어 있습니다.

모든 성공 모델은 동일한 5-Fold manifest + fold-safe WCG를 사용했습니다.
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
V5_DIR = BASE_DIR / 'exports' / 'v5'

summary = pd.read_csv(V5_DIR / 'model_zoo_summary.csv')
diversity = pd.read_csv(V5_DIR / 'diversity_vs_v4b.csv')
corr = pd.read_csv(V5_DIR / 'pairwise_probability_correlation.csv', index_col=0)
oof = pd.read_csv(V5_DIR / 'model_zoo_oof.csv')
seed_summary = pd.read_csv(V5_DIR / 'seed_probes' / 'seed_summary.csv')
thresholds = pd.read_csv(V5_DIR / 'threshold_crossfit.csv')
hard_folds = pd.read_csv(V5_DIR / 'hard_vote_fold_metrics.csv')
robust_folds = pd.read_csv(V5_DIR / 'robust_vote_fold_metrics.csv')

display(summary)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Pipeline Mermaid

```mermaid
flowchart LR
    A[Train 891 / Test 418] --> B[v2 Base Features]
    B --> C[Fixed 5-Fold Manifest]
    C --> D[Fold-safe WCG]
    D --> E[Expanded Model Zoo]
    E --> F[OOF Accuracy / AUC / Diversity]
    F --> G[MLP-PLR Seed Audit]
    G --> H[v4b + RuleFit + MLP Vote]
    H --> I[Local v5 Candidates]
```
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 1. v5 Architecture"))
    cells.append(
        nbf.v4.new_code_cell(
            """import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

boxes = [
    (0.2, 4.8, 2.2, 1.2, '1. v2 Features\\n41 features'),
    (2.7, 4.8, 2.2, 1.2, '2. Fixed 5-Fold\\nSame manifest'),
    (5.2, 4.8, 2.2, 1.2, '3. Fold-safe WCG\\nTrain labels only'),
    (7.7, 4.8, 2.5, 1.2, '4. Model Zoo\\n12 new models'),
    (10.5, 4.8, 3.0, 1.2, '5. OOF Screen\\nAcc/AUC/diversity'),
    (4.4, 1.4, 2.4, 1.2, '6. MLP-PLR\\nBest single @ seed42'),
    (7.1, 1.4, 2.4, 1.2, '7. Seed Probe\\n42/142/242'),
    (9.8, 1.4, 3.0, 1.2, '8. Hard Vote\\nv4b + RuleFit + MLP'),
]
for x, y, w, h, label in boxes:
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.08', alpha=0.18, lw=1.5
    )
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')

arrows = [
    ((2.4, 5.4),(2.7,5.4)), ((4.9,5.4),(5.2,5.4)),
    ((7.4,5.4),(7.7,5.4)), ((10.2,5.4),(10.5,5.4)),
    ((12.0,4.8),(5.6,2.6)), ((6.8,2.0),(7.1,2.0)), ((9.5,2.0),(9.8,2.0)),
]
for start, end in arrows:
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.5))

plt.title('Titanic v5 - Aggressive Multi-Family Model Zoo', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 2. Model Zoo OOF 성능"))
    cells.append(
        nbf.v4.new_code_cell(
            """plot_df = summary.copy().sort_values('accuracy', ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(plot_df['model'], plot_df['accuracy'])
plt.xlim(0.80, 0.86)
plt.xlabel('OOF Accuracy')
plt.title('v5 Model Zoo - Fold-safe OOF Accuracy')
plt.tight_layout()
plt.show()

display(summary.style.format({
    'accuracy': '{:.5f}',
    'roc_auc': '{:.5f}',
    'fold_accuracy_std': '{:.5f}',
    'fold_auc_std': '{:.5f}',
}, na_rep='-'))
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """핵심 결과:
- MLP-PLR seed42: Accuracy 0.85297 — 단일 모델 로컬 1위
- RuleFit: Accuracy 0.84961, AUC 0.89598
- TabPFN v2: Accuracy 0.84287, AUC 0.89663
- TabR, RealMLP, FTTransformer, TabM 등은 이번 설정에서 champion을 넘지 못함
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 3. Diversity vs v4b Champion"))
    cells.append(
        nbf.v4.new_code_cell(
            """plt.figure(figsize=(9, 5))
plt.scatter(
    diversity['probability_correlation_vs_v4b'],
    diversity['v4b_wrong_candidate_right'],
    s=80,
)
for _, row in diversity.iterrows():
    plt.annotate(
        row['model'],
        (row['probability_correlation_vs_v4b'], row['v4b_wrong_candidate_right']),
        xytext=(4,4), textcoords='offset points', fontsize=8,
    )
plt.xlabel('Probability correlation vs v4b')
plt.ylabel('v4b wrong / candidate right')
plt.title('Diversity vs Current Champion')
plt.tight_layout()
plt.show()

display(diversity)
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 4. MLP-PLR Seed Stability"))
    cells.append(
        nbf.v4.new_code_cell(
            """display(seed_summary.style.format({
    'accuracy': '{:.5f}',
    'roc_auc': '{:.5f}',
}))

plt.figure(figsize=(7,4))
sns.barplot(data=seed_summary, x='seed', y='accuracy')
plt.ylim(0.83, 0.86)
plt.title('MLP-PLR Accuracy by Seed')
plt.tight_layout()
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """seed42의 0.85297은 매우 좋지만 seed142/242는 약 0.84063 / 0.84287입니다.

따라서 MLP-PLR 단독 최고점은 seed 민감성이 큽니다. 이 때문에 v5에서는 MLP-PLR 하나만 champion으로 올리기보다, 기존 v4b와 RuleFit을 함께 사용하는 hard vote를 검사했습니다.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 5. Hard Vote와 Robust Vote"))
    cells.append(
        nbf.v4.new_code_cell(
            """hard_acc = (hard_folds['hard_vote_accuracy'] * [179,178,178,178,178]).sum() / 891
robust_acc = (robust_folds['accuracy'] * [179,178,178,178,178]).sum() / 891
vote_summary = pd.DataFrame([
    {
        'candidate': 'seed42 hard vote',
        'accuracy': hard_acc,
        'fold_std': hard_folds['hard_vote_accuracy'].std(ddof=0),
    },
    {
        'candidate': 'seed-mean robust vote',
        'accuracy': robust_acc,
        'fold_std': robust_folds['accuracy'].std(ddof=0),
    },
])
display(vote_summary.style.format({'accuracy':'{:.5f}','fold_std':'{:.5f}'}))

fold_plot = hard_folds[['fold','hard_vote_accuracy']].rename(
    columns={'hard_vote_accuracy':'accuracy'}
).assign(candidate='seed42 hard vote')
rob_plot = robust_folds[['fold','accuracy']].copy().assign(candidate='robust vote')
fold_plot = pd.concat([fold_plot, rob_plot], ignore_index=True)

plt.figure(figsize=(9,4.5))
sns.barplot(data=fold_plot, x='fold', y='accuracy', hue='candidate')
plt.ylim(0.82, 0.89)
plt.title('Hard Vote Fold Accuracy')
plt.tight_layout()
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """seed42 hard vote 구성:
1. v4b champion
2. RuleFit
3. MLP-PLR seed42

각 모델을 먼저 0/1로 바꾼 뒤 3개 중 2개 이상이 생존이라고 판단하면 최종 1입니다.
OOF Accuracy는 0.85522로 현재 로컬 최고입니다.

하지만 MLP seed 민감성을 고려해 MLP-PLR 3개 seed의 평균 확률을 세 번째 투표자로 사용하는 robust vote도 만들었습니다. 이 후보는 OOF Accuracy 0.85410이며 seed 노이즈에 덜 의존합니다.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6. Threshold Audit"))
    cells.append(
        nbf.v4.new_code_cell(
            """display(thresholds)
print('MLP-PLR fixed threshold 0.5 OOF Accuracy: 0.85297')
print('Cross-fitted threshold OOF Accuracy       : 0.85073')
print('Conclusion: keep threshold 0.5')
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 7. 결론

현재 로컬 후보 순위:
1. seed42 hard vote: 0.85522
2. seed-robust hard vote: 0.85410
3. MLP-PLR seed42: 0.85297
4. v4b champion: 0.84961

중요한 판단:
- MLP-PLR는 강하지만 seed 민감성이 큼
- RuleFit은 다른 family이면서 높은 AUC와 유용한 오류 다양성을 제공
- hard vote는 MLP seed가 바뀌어도 약 0.8507~0.8552 범위에서 강함
- 따라서 다음 Kaggle 확인 후보로는 seed-robust hard vote가 더 보수적
- TabPFN v3 / v2.5는 1회 라이선스 승인 후 후속 실험 대상으로 남아 있음

이 노트북은 모델 재학습이나 Kaggle 제출을 수행하지 않습니다.
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

    path = NOTEBOOKS_DIR / "Titanic_Ensemble_Pipeline_v5.ipynb"
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook created: {path}")
    print("Executing analysis-only notebook...")
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(BASE_DIR)}},
    )
    client.execute()
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed notebook saved: {path}")


if __name__ == "__main__":
    main()
