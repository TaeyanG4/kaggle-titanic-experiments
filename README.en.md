# Kaggle Titanic Experiments

Titanic - Machine Learning from Disaster

![Kaggle Titanic Experiments](docs/assets/hero.svg)

Kaggle Notebook: [Titanic 2026 Ensemble Techniques - Public 0.83014](https://www.kaggle.com/code/taeyangg4/titanic-2026-ensemble-techniques-public-0-83014)

[한국어](README.md) / [Experiment log](docs/02-experiment-journey.md) / [Validation](docs/03-validation-and-integrity.md) / [Reproduction](docs/05-reproduction.md)

## About the experiment

I started this project to see how far custom skills and prompts could take a semi-automated machine-learning workflow. The task was Kaggle Titanic: build features, compare models, inspect failures and decide which predictions to submit.

Antigravity handled the initial environment setup. Subsequent planning, coding, execution requests, analysis and documentation were done through a GPT web session connected to local Python tools. I set the direction, questioned the results and approved submissions. This was a guided workflow, not a fully autonomous benchmark.

The first submission scored 0.79186. The selected final artifact, v47, scored 0.83014. This repository includes the failed submissions and rejected experiments as well as the improvements.

## Recorded results

| Item | Result |
|---|---|
| First submission | v1, Public 0.79186 |
| Best submission | v47, Public 0.83014 |
| Displayed-score increase | 0.03828, about 3.83 percentage points |
| v5 reference ensemble | OOF 0.85410, Public 0.79665 |
| P3 bagged candidate | Reported OOF 0.85971, submitted Public 0.79665 |
| Actual submissions | 17, October 4-5, 2026 UTC |
| Final submission ID | 56841675 |

Sources are the [submission receipts](docs/evidence/kaggle-submissions.csv), [evaluation exports](exports/v44/summary.csv) and [artifact hashes](docs/evidence/submission-manifest.json). The v1-v47 labels identify work stages, not 47 independent experiments.

![Every submission and the best score so far](docs/assets/submission-history.png)

The goal was to improve without answer lookup or data leakage. Some historical reproduction branches nevertheless calculated target-derived features before CV, reused public notebook predictions or used leaderboard feedback in later selection. The final Public score should not be read as an independently verified leakage-free generalization result. These limitations are documented in the [validation report](docs/03-validation-and-integrity.md).

## External public-notebook comparators

For context, the repository also tracks public Kaggle notebooks scoring at least 0.80 that are useful leakage-aware comparators. These are the Public/Best Scores displayed on the notebook pages, not scores reproduced in this repository. `Clean candidate` means a priority for comparison under the currently visible information boundary; it is not an independent certification of the full notebook code.

| Public notebook | Displayed score | Current status | Note |
|---|---:|---|---|
| [Yoni Krichevsky - Top 3% with only 4 features - no data leakage](https://www.kaggle.com/code/yoni2k/top-3-with-only-4-features-no-data-leakage) | 0.81818 | clean-priority candidate | The notebook explicitly says `no data leakage`, and Kaggle shows one input file. A full independent code audit has not been completed. |
| [Jonathan Oheix - Titanic survivors prediction - TOP 5%](https://www.kaggle.com/code/jonathanoheix/titanic-survivors-prediction-top-5) | 0.82296 | audit pending | Kaggle shows one input file and the score, but the full feature pipeline has not yet been audited for family/ticket target-derived features. |
| [Chris Deotte - Titanic Deep Net [0.82296]](https://www.kaggle.com/code/cdeotte/titanic-deep-net-0-82296) | 0.82296 | audit pending | An R competition notebook with a verified displayed score; it is not classified as clean until the full code is audited under the strict leakage boundary. |
| [Titanic competition w/ TensorFlow Decision Forests](https://www.kaggle.com/code/gusthema/titanic-competition-w-tensorflow-decision-forests) | 0.80143 | conservative baseline | A pinned competition notebook using the Titanic competition inputs. It is kept as a reproducible external baseline rather than a high-score ceiling candidate. |

Under that convention, 0.81818 is the strongest current public score treated as a clean-priority comparator, while the two 0.82296 notebooks remain audit-pending candidates. This table does not reclassify v47's 0.83014 as leakage-free; it is meant to keep different evidence boundaries visibly separate. Source and audit notes are in [references](docs/07-references.md).

## Workflow

<p align="center">
  <img src="docs/assets/workflow.png" width="660" alt="Initial setup and human direction feed a GPT web session, local execution, evaluation, approved submission and experiment records">
</p>

[Diagram source](docs/diagrams/workflow.dot) / [Larger image](docs/assets/workflow.svg)

Skills defined the order of checks and the records to keep. The prompts did not serve as model input features. The session wrote and ran Python experiments, while I reviewed the direction and decided when to continue or submit.

## What changed

### Baseline and validation boundaries

The first model averaged six tree-model probabilities. Features included title, family size and ticket frequency. v2 added fare per ticket holder and further interactions, but its Public score fell to 0.78708. Group-survival features then received a separate audit: validation labels needed to stay out of the group statistics used to predict that fold.

### Diverse models and a stable ensemble

v4b blended the tree average with TabICLv2 at 90:10 and scored 0.79425. v5 used a majority vote across v4b, RuleFit and a three-seed MLP-PLR average, reaching 0.79665. Logistic stacking was also tested, but a higher AUC did not consistently produce higher classification accuracy in those comparisons.

### Public recipes and relational features

The Gunes-style v10 reproduction used age/fare quantile bins, grouped decks and family/ticket target statistics. It scored 0.81578, but its original CV reused statistics built from all training labels. Later branches compared fold-aware encodings, role-specific rates, empirical-Bayes shrinkage and Deotte WCG rules rather than treating that CV score as a clean baseline.

### Error analysis and new representations

Repeated errors led to a character n-gram model of Name, Ticket and Cabin. It was weak on its own but supplied some different predictions. Graph centrality did not help the recorded accuracy comparisons. Numeric three-digit ticket prefixes were more promising locally: the P3 bag reached reported OOF 0.85971, yet its submitted Public score was 0.79665.

### Final submitted combination

The final branch preserved v10 and applied selected corrections instead of replacing every prediction. A high-confidence text correction produced v38 at 0.81818. Adding the narrow WCG female-death guard produced v46 at 0.82775. The broader Deotte-derived guard produced v47 at 0.83014.

![The frozen v10, v38, v46 and v47 prediction lineage](docs/assets/final-lineage.png)

[Diagram source](docs/diagrams/final-lineage.dot) / [Larger image](docs/assets/final-lineage.svg)

Both v46 and v47 start from v38. The 13 changes in the broader guard include the four in the narrow guard. Public feedback informed the selection, and the broader guard includes reused public predictions. There is no independently evaluated OOF score for the final v47 artifact.

## Reproduction

```bash
git clone https://github.com/TaeyanG4/kaggle-titanic-experiments.git
cd kaggle-titanic-experiments
python tools/verify_publication.py
python tools/replay_final_artifact.py --output replayed_v47.csv
```

These commands check the frozen files and reconstruct the final prediction CSV. They do not retrain models or submit to Kaggle. Training requirements and limitations are described in [reproduction](docs/05-reproduction.md).

## Further reading

The Korean reports cover the [design](docs/01-experiment-design.md), [step-by-step changes](docs/02-experiment-journey.md), [validation](docs/03-validation-and-integrity.md), [skills and prompts](docs/04-workflow-and-prompts.md), [lessons](docs/06-results-and-lessons.md) and [sources](docs/07-references.md). Earlier working notes remain in `archive/`.

No raw competition CSVs, credentials or model checkpoints are included in the new experiment snapshot. Submission labels are predictions, not hidden test answers. No no-skills control or sealed final holdout was used, so the score increase cannot be attributed to skills alone.

The experiment is closed. There is no scheduled training or submission process. See [NOTICE](NOTICE.md) for reuse information.
