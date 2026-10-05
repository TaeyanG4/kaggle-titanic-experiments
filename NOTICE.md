# Attribution and reuse

Kaggle Titanic Experiments records a semi-automated experiment in Titanic - Machine Learning from Disaster. Initial setup used Antigravity. Subsequent work ran through a human-guided GPT web session connected to local tools.

Some code adapts public Titanic methods by Gunes Evitan and Chris Deotte. The Deotte branch also reuses passenger-specific predictions from public notebook output. These are credited as external predictions, not independently learned results or original discoveries. Details are in [references](docs/07-references.md) and [validation](docs/03-validation-and-integrity.md).

Not every historical third-party reuse condition was pinned, so this repository does not grant a new blanket permissive license over all material. Check the upstream source and applicable permissions before reuse. This notice does not override upstream rights or Kaggle data terms.

Raw competition CSVs, credentials, model checkpoints and downloaded third-party notebook bodies are excluded from the new experiment snapshot. Derived OOF files can contain official training labels. Submission files contain predictions, not hidden test answers.

Numerical charts come from retained CSVs through `tools/build_report_assets.py`. Flowcharts come from the DOT sources in `docs/diagrams/` using `tools/diagram-renderer/`. Source and image hashes are recorded. Older notes remain in the archive with the final validation report supplying the relevant qualifications.
