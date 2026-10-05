"""Replay frozen v47 predictions without training, network access or submission."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'ce8e484730a667e0b5d80068cf8f1418444e0113a9af28996dd8185df60ea0e4'

def load(path: Path) -> dict[int, int]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['PassengerId', 'Survived']:
            raise ValueError(f'Unexpected columns: {path.name}')
        rows = list(reader)
    ids = [int(row['PassengerId']) for row in rows]
    if ids != list(range(892, 1310)):
        raise ValueError(f'Unexpected IDs/order: {path.name}')
    values = [int(row['Survived']) for row in rows]
    if not set(values).issubset({0, 1}):
        raise ValueError(f'Nonbinary predictions: {path.name}')
    return dict(zip(ids, values))

def replay() -> bytes:
    base = load(ROOT/'submissions/submission_v10_score_0.81578.csv')
    parent = load(ROOT/'submissions/submission_v38_v10_text_rescue.csv')
    broad = load(ROOT/'submissions/submission_v24_v10_deotte_all_deotte_female_death.csv')
    candidate = {pid: broad[pid] if broad[pid] != base[pid] else parent[pid] for pid in base}
    frozen = load(ROOT/'submissions/submission_v47_score_0.83014.csv')
    if candidate != frozen:
        raise ValueError('Replayed predictions differ from frozen v47.')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\r\n')
    writer.writerow(['PassengerId', 'Survived'])
    writer.writerows(sorted(candidate.items()))
    data = stream.getvalue().encode('utf-8')
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError('Replayed bytes do not match the recorded SHA-256.')
    return data

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing output file.')
    data = replay()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f'PASS: 418 frozen predictions replayed; sha256={EXPECTED}')
    print('No model training, network request or Kaggle submission was performed.')

if __name__ == '__main__':
    main()
