# Final handoff

Repository: `TaeyanG4/titanic-gpt-web-experiment` (formerly `Kaggle_Titanic_practice`). Original Git history is retained.

Start with [README](README.md), [integrity](docs/03-validation-and-integrity.md), and [reproduction](docs/05-reproduction.md). Historical working notes are in [archive/session-notes](archive/session-notes/).

Frozen Public artifact: `submissions/submission_v47_score_0.83014.csv`; active alias `submissions/submission.csv`; both SHA-256:

`ce8e484730a667e0b5d80068cf8f1418444e0113a9af28996dd8185df60ea0e4`

Do not conflate v5's historical OOF 0.85410, P3's reported bagged OOF 0.85971 and v47's Public 0.83014. They are different evaluations. The v47 lineage reuses public predictions and was selected adaptively from Public scores. No sealed final holdout was available.

Run `python tools/verify_publication.py` for artifact/document checks. Run `python tools/replay_final_artifact.py --output replayed_v47.csv` for frozen-file reconstruction, not model retraining. Never infer hidden labels from the submission artifacts and call them training truth.

No active experiment or scheduled submission is part of the final repository. Credentials and model caches remain outside it.
