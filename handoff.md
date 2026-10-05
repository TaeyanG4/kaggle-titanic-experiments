# Final handoff

Repository: `TaeyanG4/kaggle-titanic-experiments`.

Read [README](README.md), [validation](docs/03-validation-and-integrity.md), and [reproduction](docs/05-reproduction.md) before changing the experiment record. The [step-by-step log](docs/02-experiment-journey.md) links to the implementation and results for each stage.

The selected Public artifact is `submissions/submission_v47_score_0.83014.csv`, also preserved as `submissions/submission.csv`. Its SHA-256 is:

```text
ce8e484730a667e0b5d80068cf8f1418444e0113a9af28996dd8185df60ea0e4
```

v5 OOF 0.85410, P3 bagged OOF 0.85971 and v47 Public 0.83014 are different evaluations. v47 includes public prediction reuse and leaderboard-adaptive selection. It has no independently evaluated OOF score.

Run `python tools/verify_publication.py` to check the frozen files and documentation. Run `python tools/replay_final_artifact.py --output replayed_v47.csv` to reconstruct the final CSV from its stored parents. Neither command trains models or submits to Kaggle.

The flowchart sources are in `docs/diagrams/`; the renderer is in `tools/diagram-renderer/`. Preserve image and source hashes when editing them. Do not change historical scripts and attribute their old scores to the changed code.

The experiment is closed. Credentials and model caches remain outside the repository.
