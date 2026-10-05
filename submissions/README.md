# Frozen prediction artifacts

These CSVs contain predicted `Survived` values, not hidden test answers. Some files were submitted; many were not. Filenames with scores are backups, not additional submissions. The actual 2026 campaign consists of the 17 receipts in [kaggle-submissions.csv](../docs/evidence/kaggle-submissions.csv).

`submission.csv` and `submission_v47_score_0.83014.csv` preserve the selected Public artifact. Its construction includes historical preprocessing, public prediction reuse and leaderboard-adaptive selection; see the [integrity report](../docs/03-validation-and-integrity.md).

All included files are listed in the [hash manifest](../docs/evidence/submission-manifest.json). The [replay tool](../tools/replay_final_artifact.py) checks ID alignment and reconstructs the final artifact from its frozen parents without training or submitting anything.
