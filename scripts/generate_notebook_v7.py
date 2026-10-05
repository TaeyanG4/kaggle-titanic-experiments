"""Generate and execute the analysis-only Titanic v7 notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


BASE_DIR = Path(__file__).resolve().parents[1]
V7_DIR = BASE_DIR / "exports" / "v7"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


def main() -> None:
    required = [
        V7_DIR / "native_catboost_summary.csv",
        V7_DIR / "catboost_hpo_screen.csv",
        V7_DIR / "catboost_hpo_stability.csv",
        V7_DIR / "preprocessing_audit_summary.csv",
        V7_DIR / "disagreement_candidate_summary.csv",
        V7_DIR / "disagreement_slice_summary.csv",
        V7_DIR / "rulefit_hpo_screen.csv",
        V7_DIR / "rulefit_hpo_stability.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v7 artifacts: {missing}")

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Titanic v7 — Preprocessing Bottleneck Audit & Selective HPO

## 목표

v6에서 공개 Titanic 노하우를 바탕으로 feature engineering을 넓게 조사한 뒤,
v7에서는 다음 질문만 집중적으로 검증합니다.

1. one-hot 대신 CatBoost native categorical 표현이 더 나은가?
2. 가족/티켓 기반 Age 및 Cabin/Deck 보간이 실제로 도움이 되는가?
3. v6에서 강했던 AdultMale CatBoost를 소규모 HPO하면 안정적으로 좋아지는가?
4. champion의 오류를 FamilyFare / Ticket / TabPFN v2가 반복적으로 구하는 slice가 있는가?
5. champion 구성원인 RuleFit을 튜닝하면 hard vote가 좋아지는가?

현재 Public Champion은 **v5 robust hard vote = 0.79665**이며 v7은 이를 덮어쓰지 않습니다.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path.cwd()
V7 = BASE_DIR / 'exports' / 'v7'

native = pd.read_csv(V7 / 'native_catboost_summary.csv')
cat_screen = pd.read_csv(V7 / 'catboost_hpo_screen.csv')
cat_stability = pd.read_csv(V7 / 'catboost_hpo_stability.csv')
pre = pd.read_csv(V7 / 'preprocessing_audit_summary.csv')
rescue = pd.read_csv(V7 / 'disagreement_candidate_summary.csv')
slices = pd.read_csv(V7 / 'disagreement_slice_summary.csv')
rule_screen = pd.read_csv(V7 / 'rulefit_hpo_screen.csv')
rule_stability = pd.read_csv(V7 / 'rulefit_hpo_stability.csv')
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Public-notebook hypotheses

공개 자료에서 반복되는 핵심은 Title, family/ticket groups, cabin/deck,
smart Age imputation, woman-child group survival입니다.

현재 파이프라인은 이미 Title × Pclass Age median과 fold-safe WCG를 사용하므로,
v7에서는 **아직 충분히 검증하지 않은 표현 방식**만 분리해 테스트했습니다.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 2. Native categorical CatBoost"))
    cells.append(
        nbf.v4.new_code_cell(
            """display(native.style.format({
    'accuracy':'{:.5f}', 'roc_auc':'{:.5f}',
    'fold_accuracy_std':'{:.5f}', 'fold_auc_std':'{:.5f}'
}))

plt.figure(figsize=(9,4.5))
d = native.sort_values('accuracy')
plt.barh(d['variant'], d['accuracy'])
plt.xlim(0.835, 0.852)
plt.title('Native categorical CatBoost OOF Accuracy')
plt.tight_layout()
plt.show()
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """결론: native categorical은 이번 데이터/설정에서 기존 one-hot 표현을 이기지 못했습니다.
최고 variant도 Accuracy 0.84512 수준이므로 이 branch는 종료합니다.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 3. Relational Age / Deck preprocessing"))
    cells.append(
        nbf.v4.new_code_cell(
            """panel = pre[pre['model']=='Panel'].sort_values('accuracy', ascending=False)
display(panel.style.format({
    'accuracy':'{:.5f}', 'roc_auc':'{:.5f}',
    'delta_accuracy':'{:+.5f}', 'delta_auc':'{:+.5f}'
}))

plt.figure(figsize=(10,4.5))
plt.barh(panel.sort_values('accuracy')['variant'], panel.sort_values('accuracy')['accuracy'])
plt.xlim(0.835, 0.85)
plt.title('Preprocessing variants - 3-model panel Accuracy')
plt.tight_layout()
plt.show()
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """관계형 보간은 일부 모델에서 작은 개선이 있었지만 panel 기준 안정적 향상은 없었습니다.
특히 Age를 가족/티켓으로 먼저 채우는 방식은 AUC가 조금 오르더라도 Accuracy가 떨어지는 경우가 많았습니다.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 4. Limited CatBoost HPO"))
    cells.append(
        nbf.v4.new_code_cell(
            """display(cat_screen.style.format({
    'accuracy':'{:.5f}', 'roc_auc':'{:.5f}',
    'fold_accuracy_std':'{:.5f}', 'fold_auc_std':'{:.5f}'
}))
display(cat_stability.style.format({
    'mean_accuracy_across_manifests':'{:.5f}',
    'min_accuracy_across_manifests':'{:.5f}',
    'std_accuracy_across_manifests':'{:.5f}',
    'mean_auc_across_manifests':'{:.5f}',
}))
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """가장 안정적인 설정은 slow_d3_l2_6:
- trusted fold Accuracy: 0.85073
- alternate split seeds 123 / 777: 둘 다 0.84624

baseline보다 안정적이지만 v5 robust 0.85410을 넘지 못하므로 champion 승격 대상은 아닙니다.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 5. Champion residual / disagreement audit"))
    cells.append(
        nbf.v4.new_code_cell(
            """display(rescue)
cols = ['slice_feature','slice_value','n','champion_accuracy','champion_errors','best_alt_delta']
display(slices[cols].head(20).style.format({
    'champion_accuracy':'{:.4f}', 'best_alt_delta':'{:+.4f}'
}))
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """FamilyFareGB는 champion이 틀린 17명을 구했지만 champion이 맞힌 21명을 잃어 전체로는 -4입니다.

Large family slice에서는 +2 correct가 관측됐지만 **두 개선 모두 fold 4에만 발생**했습니다.
따라서 대가족 override rule은 validation artifact일 가능성이 있어 승격하지 않았습니다.

공개 woman-child deterministic override도 테스트했지만 v5 robust 예측을 한 명도 바꾸지 않았습니다.
즉 그 고신뢰 규칙은 현재 ensemble이 이미 흡수하고 있습니다.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6. RuleFit limited HPO"))
    cells.append(
        nbf.v4.new_code_cell(
            """display(rule_screen.style.format({
    'rulefit_accuracy':'{:.5f}', 'rulefit_auc':'{:.5f}',
    'hard_vote_accuracy':'{:.5f}',
    'fold_rulefit_std':'{:.5f}', 'fold_hard_vote_std':'{:.5f}'
}))
display(rule_stability.style.format({
    'mean_rulefit_accuracy':'{:.5f}', 'min_rulefit_accuracy':'{:.5f}',
    'std_rulefit_accuracy':'{:.5f}', 'mean_hard_vote_accuracy':'{:.5f}',
    'min_hard_vote_accuracy':'{:.5f}', 'std_hard_vote_accuracy':'{:.5f}',
    'mean_rulefit_auc':'{:.5f}',
}))
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """no_linear RuleFit은 seed42 단독 Accuracy 0.85073까지 올라갔지만,
hard vote는 0.85410 그대로였습니다. 다른 RuleFit seed에서는 vote가 0.84961까지 내려갑니다.

따라서 original RuleFit을 유지합니다.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 7. v7 결론과 다음 계획

### 이번에 기각된 가설

- CatBoost native categorical이 one-hot보다 낫다 → 기각
- 가족/티켓 Age/Deck 보간이 전체 모델을 안정적으로 높인다 → 기각
- AdultMale CatBoost HPO가 champion을 이긴다 → 기각
- 대가족 FamilyFare override가 fold 전반에 안정적이다 → 기각
- RuleFit HPO가 robust hard vote를 높인다 → 기각

### 남은 가장 높은 정보가치

**TabPFN v3 / v2.5**입니다.

현재 tabpfn 패키지와 실행 adapter는 준비되어 있지만 Prior Labs의 일회성
license/auth 승인이 필요합니다. 승인 후에는 다음 네 실험을 동일 fold에서 수행합니다.

1. TabPFN v2.5 — 기존 trusted v2 representation
2. TabPFN v3 — 기존 trusted v2 representation
3. TabPFN v2.5 — v2 + selected FamilyFare representation
4. TabPFN v3 — v2 + selected FamilyFare representation

승격 기준:
- OOF Accuracy가 v5 robust 0.85410을 직접 넘거나,
- 비슷한 Accuracy에서 champion이 틀리는 row를 유의미하게 구하고,
- fold/model-version 안정성이 확인될 것.

그 전에는 추가적인 작은 HPO나 generic feature accumulation을 중단합니다.
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

    path = NOTEBOOKS_DIR / "Titanic_Ensemble_Pipeline_v7.ipynb"
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook created: {path}")
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
