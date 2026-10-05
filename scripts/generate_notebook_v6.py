"""Generate and execute the analysis-only Titanic v6 feature-engineering notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


BASE_DIR = Path(__file__).resolve().parents[1]
V6_DIR = BASE_DIR / "exports" / "v6"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


def main() -> None:
    required = [
        V6_DIR / "feature_ablation_summary.csv",
        V6_DIR / "targeted_feature_panel_rank.csv",
        V6_DIR / "group_key_audit_summary.csv",
        V6_DIR / "familyfare_modern_summary.csv",
        V6_DIR / "selected_models_summary.csv",
        V6_DIR / "stacking_probe.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v6 artifacts: {missing}")

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            """# Titanic v6 — Public Feature-Engineering Research & Fold-Safe Ablation

## Why v6?

v5 improved the Public score mainly through model diversity. v6 tests whether
the next bottleneck is representation/preprocessing instead.

Public Titanic notebooks and discussions were used as hypothesis generators,
not copied blindly. Repeated ideas include:

- Title / family size / is-alone
- richer cabin structure
- ticket/family counts
- Sex × Pclass interactions
- smarter missing-value handling
- surname/fare family groups and family-survival features

Many of these already exist in our v2-v5 pipeline. v6 therefore focuses on the
missing or materially different representations.

Public references:
- Manav Sehgal, Titanic Data Science Solutions
- Chris Deotte, Titanic using Name only [0.81818]
- Caleb Castleberry, Titanic Cabin Features
- Family Survival implementations using LastName + Fare
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path.cwd()
V6_DIR = BASE_DIR / 'exports' / 'v6'

ablation = pd.read_csv(V6_DIR / 'feature_ablation_summary.csv')
targeted = pd.read_csv(V6_DIR / 'targeted_feature_panel_rank.csv')
groupkey = pd.read_csv(V6_DIR / 'group_key_audit_summary.csv')
family_modern = pd.read_csv(V6_DIR / 'familyfare_modern_summary.csv')
selected = pd.read_csv(V6_DIR / 'selected_models_summary.csv')
stack = pd.read_csv(V6_DIR / 'stacking_probe.csv')
"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Broad feature blocks

The first test added four logical blocks independently:

1. cabin/missingness
2. relational counts
3. explicit interactions/bands
4. detailed fold-safe WCG peer statistics

Large structural blocks generally hurt. This is a useful negative result:
Titanic is too small for indiscriminately adding dozens of correlated features.
"""
        ),
        nbf.v4.new_code_cell(
            """panel = ablation[ablation['model'] == 'Panel'].sort_values('accuracy', ascending=False)
display(panel)
plt.figure(figsize=(9,4))
plt.bar(panel['variant'], panel['accuracy'])
plt.xticks(rotation=35, ha='right')
plt.ylim(0.82, 0.855)
plt.title('v6 broad feature-block ablation')
plt.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Targeted one-group screen

After broad blocks failed, each logical feature group was tested separately.

Best panel Accuracy gains:
- FarePerFamily: +0.00224
- AdultMale: +0.00224
- IsMother: +0.00112

Ticket-WCG detail did not improve fixed-threshold Accuracy but materially raised
AUC, especially for CatBoost.
"""
        ),
        nbf.v4.new_code_cell(
            """display(targeted.head(15))
plt.figure(figsize=(10,5))
top = targeted.head(12).iloc[::-1]
plt.barh(top['feature_group'], top['delta_accuracy'])
plt.title('Targeted feature groups: Accuracy delta vs baseline')
plt.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Public Family Survival idea: LastName + Fare

A materially different public approach groups passengers by surname and fare,
then derives peer survival information. We rebuilt this idea under strict
fold-safe validation.

This was the strongest v6 representation finding:
- familyfare_all panel: Accuracy 0.84624 vs baseline 0.84287
- ROC-AUC 0.90356 vs 0.89786
- GradientBoosting: 0.84961 / 0.90125

So the public idea transfers, but not enough by itself to beat the v5 robust
champion at 0.85410 local OOF Accuracy.
"""
        ),
        nbf.v4.new_code_cell(
            """gpanel = groupkey[groupkey['model'] == 'Panel'].sort_values(
    ['accuracy','roc_auc'], ascending=False
)
display(gpanel)
"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Why not just feed every new feature into RuleFit / MLP?

We tested that too. LastName+Fare peer features improved ranking/AUC for RuleFit
and MLP-PLR, but reduced 0.5-threshold Accuracy. The same pattern appeared when
model-specific selected feature sets were added broadly.

This tells us the new representation contains signal, but the current decision
boundary/calibration does not convert all of that signal into more correct
binary classifications.
"""
        ),
        nbf.v4.new_code_cell(
            """display(family_modern)
display(selected)
"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Cross-fitted stacking probe

The new Family-Fare probabilities were added as inputs to a cross-fitted
LogisticRegression meta model. AUC rose strongly in some stacks, but Accuracy
did not beat the v5 robust hard vote.

Therefore no v6 submission is promoted yet.
"""
        ),
        nbf.v4.new_code_cell(
            """display(stack)
"""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. v6 conclusion

What worked:
- LastName + Fare family grouping is real signal.
- FarePerFamily and AdultMale are small useful single features.
- richer ticket/WCG statistics improve ranking for selected models.

What did not work:
- adding every public feature at once;
- large interaction/band blocks;
- replacing the robust v5 vote with a simple larger vote;
- naive cross-fitted logistic stacking for Accuracy.

Current Public Champion remains:
- v5 robust hard vote: 0.79665 Public
- local OOF Accuracy: 0.85410

## Next plan

1. **TabPFN v3 / v2.5**
   - local package/API is ready;
   - one-time Prior Labs license/authentication is the blocker;
   - after approval, run exactly the same trusted 5 folds with v2 and v6
     Family-Fare representations.

2. **Native-categorical CatBoost lane**
   - preserve TicketPrefix, Title, Deck, FamilyType and selected group keys as
     categoricals instead of only one-hot encoding;
   - compare raw/native CatBoost against current numeric CatBoost.

3. **Disagreement-slice feature engineering**
   - inspect rows where v5 robust and Family-Fare models disagree;
   - search for repeatable rules in Sex/Pclass/Title/family/ticket/cabin slices;
   - only encode rules that survive fold-level checks.

4. **Feature selection, not feature accumulation**
   - test FamilyFare peer feature + one extra structural feature at a time;
   - retain only features that improve at least several folds or multiple model
     families.

5. **Submission discipline**
   - do not spend another slot until a v6/v7 candidate beats the v5 robust
     0.85410 OOF baseline with acceptable fold/seed stability.
"""
        ),
    ]

    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    path = NOTEBOOKS_DIR / "Titanic_Ensemble_Pipeline_v6.ipynb"
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
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
