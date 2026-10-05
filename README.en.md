# Kaggle Titanic Experiments

**Titanic — Machine Learning from Disaster**

A documented semi-automated experiment in feature engineering, ensembles, validation and failure analysis.

[한국어](README.md) · [Experiment journey](docs/02-experiment-journey.md) · [Integrity statement](docs/03-validation-and-integrity.md) · [Reproduction](docs/05-reproduction.md)

![Kaggle Titanic Experiments: competition experiments and the recorded Public best](docs/assets/hero.svg)

## Purpose, not just a score

The research question is **how far a human-guided GPT web session can take the Titanic experiment using owner-configured skills and prompts**. The repository name identifies the competition and its experiment record; the execution tools and human roles are described separately below.

This is an observational case study of semi-automated data-science work on Kaggle Titanic. According to the owner, Antigravity was used for initial setup; subsequent experimentation was directed through a GPT web session using owner-configured skills and prompts. The human supplied goals, challenged explanations, requested further investigation and authorized submissions. Connected local tools executed Python and managed files. This was not a fully autonomous agent benchmark, nor a controlled comparison against manual work or a no-skills baseline.

The intended constraint was to improve without cheating, answer lookup or deliberate leakage. **The final audit does not support an unqualified claim that every historical branch, or the final best Public artifact, is leakage-free.** Some branches built target-derived features before CV; others reused passenger-level predictions from a public notebook. Later candidate selection also used leaderboard feedback. Those facts are disclosed rather than hidden. They do not establish a competition-rule violation by themselves, but they limit what the result proves.

## Recorded outcomes

| Evidence | Result | Interpretation |
|---|---:|---|
| First submission in this 2026 campaign | 0.79186 | v1 |
| Best recorded Public submission | **0.83014** | v47, submission **56841675** |
| Displayed-score improvement | **+0.03828** | Approximately 3.83 percentage points |
| Retained v5 benchmark | OOF 0.85410; Public 0.79665 | Historically audited group-target boundary; not a blanket certificate |
| P3 bagged research candidate | Reported OOF 0.85971; submitted Public 0.79665 | Local improvement did not transfer |
| Actual campaign submissions | **17** | October 4–5, 2026 UTC; older practice excluded |

Source: [read-only Kaggle receipts](docs/evidence/kaggle-submissions.csv), [submission hashes](docs/evidence/submission-manifest.json), and retained experiment exports. Version labels v1–v47 are identifiers, not 47 equal or independent trials. No live rank or percentile is claimed.

![All submissions, including regressions, and best-so-far](docs/assets/submission-history.png)

## What was tried

The campaign progressed from tree ensembles and fold-aware family/ticket features through TabICL, RuleFit, MLP, TabPFN, blending and stacking. It then tested public-feature recipes, typed relational rates, matched pseudo-tests, group-held-out validation, small hyperparameter searches, adversarial validation, empirical-Bayes shrinkage, nested selection, raw character n-grams, graph centrality and numeric ticket-prefix features. Both successful and unsuccessful branches are retained.

The final Public artifact combines a historical Gunes-style RF prediction with a text-based correction and a broader Deotte-derived female-death guard. It is **a leaderboard-selected historical artifact**, not an independently evaluated end-to-end model with a certified OOF score. The detailed Korean reports explain the underlying techniques and evidence boundaries.

## Verify without training or credentials

```bash
git clone https://github.com/TaeyanG4/kaggle-titanic-experiments.git
cd kaggle-titanic-experiments
python tools/verify_publication.py
python tools/replay_final_artifact.py --output replayed_v47.csv
```

The replay reconstructs the frozen submission from archived predictions. It does not retrain the models. See [reproduction](docs/05-reproduction.md) for data acquisition, optional dependencies and limitations.

## Reading map

The [design](docs/01-experiment-design.md), [journey](docs/02-experiment-journey.md), [validation audit](docs/03-validation-and-integrity.md), [workflow and prompts](docs/04-workflow-and-prompts.md), [reproduction](docs/05-reproduction.md), [lessons](docs/06-results-and-lessons.md), and [references](docs/07-references.md) form the final report. Earlier session notes remain in `archive/session-notes/`; original 2023 work is separated in `archive/legacy-2023/` without rewriting Git history.

Raw competition files, credentials, checkpoints and downloaded third-party notebooks are not included in the new campaign snapshot. Notebook outputs have been stripped. Derived OOF tables may contain official training labels; submission `Survived` columns are predictions, **not hidden test answers**. Existing historical source is preserved for audit and may contain defects: it is not a production library.

**Conclusion:** the session produced a substantial, inspectable experiment record and a higher observed Public score. It did not prove that custom skills caused the improvement, eliminate validation-selection bias, or establish leakage-free generalization at 83.014%. There is no no-skills control, no sealed final holdout, and no verified complete model/prompt provenance.

Campaign closed. No scheduled training or submission automation is enabled. See [NOTICE](NOTICE.md) before reusing third-party-derived material.

Repository naming history: `Kaggle_Titanic_practice` → `titanic-gpt-web-experiment` → **`kaggle-titanic-experiments`**. Git history and experiment evidence are retained.
