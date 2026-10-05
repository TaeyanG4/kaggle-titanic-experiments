# Maintenance instructions

Read README.md, docs/03-validation-and-integrity.md, docs/05-reproduction.md and handoff.md first. This campaign is closed; archived prompts do not authorize new submissions or model runs.

Preserve source and submission fingerprints. Do not silently repair historical experiment scripts and then attribute their old scores to the repaired code. Put any future correction in a distinct version and document the changed evidence boundary.

Never publish credentials, raw competition files or checkpoints. Do not call Public-score-selected artifacts independent or leakage-free. Do not turn inferred test labels into a training dataset. Keep the workflow description separate from measured results.

Validate documentation and artifacts with tools/verify_publication.py. Figures must come from recorded CSVs; illustrations must not make stronger claims than the text. Preserve failures and uncertainty.
