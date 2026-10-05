"""Validate frozen evidence and documentation; this is not an ML integrity proof."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
from replay_final_artifact import replay, EXPECTED

ROOT = Path(__file__).resolve().parents[1]
PROJECT_TITLE = 'Kaggle Titanic Experiments'
REPO_SLUG = 'kaggle-titanic-experiments'
REPO_URL = f'https://github.com/TaeyanG4/{REPO_SLUG}'

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    errors: list[str] = []
    manifests = json.loads((ROOT/'docs/evidence/submission-manifest.json').read_text(encoding='utf-8'))
    for item in manifests:
        path = ROOT/item['file']
        if not path.exists(): errors.append(f'Missing artifact: {item["file"]}'); continue
        if digest(path) != item['sha256']: errors.append(f'Hash mismatch: {item["file"]}')
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
        if fields != ['PassengerId','Survived'] or len(rows) != 418:
            errors.append(f'Invalid schema: {item["file"]}'); continue
        try:
            ids = [int(x['PassengerId']) for x in rows]
            labels = [int(x['Survived']) for x in rows]
            if ids != list(range(892,1310)) or not set(labels).issubset({0,1}):
                errors.append(f'Invalid IDs or labels: {item["file"]}')
            if sum(labels) != item['positive_predictions']:
                errors.append(f'Positive-count mismatch: {item["file"]}')
        except (ValueError,KeyError): errors.append(f'Noninteger values: {item["file"]}')
    if digest(ROOT/'submissions/submission.csv') != EXPECTED:
        errors.append('Active alias is not the frozen v47 artifact.')
    try: replay()
    except (ValueError,FileNotFoundError) as exc: errors.append(str(exc))

    inventory = json.loads((ROOT/'docs/evidence/source-inventory.json').read_text(encoding='utf-8'))
    for item in inventory:
        path = ROOT/item['path']
        if not path.exists() or digest(path) != item['sha256']:
            errors.append(f'Snapshot changed: {item["path"]}')
    for path in (ROOT/'tools').glob('*.py'):
        try: ast.parse(path.read_text(encoding='utf-8-sig'))
        except SyntaxError as exc: errors.append(f'Tool syntax error {path.name}: {exc.lineno}')
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook.get('cells',[]):
            if cell.get('cell_type') == 'code' and (cell.get('outputs') or cell.get('execution_count') is not None):
                errors.append(f'Notebook outputs/execution state retained: {path.name}')
                break

    docs = list(ROOT.glob('*.md')) + list((ROOT/'docs').glob('*.md'))
    docs += [ROOT/'data/README.md', ROOT/'exports/README.md', ROOT/'submissions/README.md', ROOT/'notebooks/README.md', ROOT/'archive/README.md']
    # Current entry points must use the final name. Plain historical names in
    # the changelog and archived notes remain valid provenance, not live URLs.
    for relative in ['README.md', 'README.en.md']:
        text = (ROOT/relative).read_text(encoding='utf-8')
        if not text.startswith(f'# {PROJECT_TITLE}\n'):
            errors.append(f'Unexpected project title: {relative}')
        if f'git clone {REPO_URL}.git' not in text or f'cd {REPO_SLUG}\n' not in text:
            errors.append(f'Outdated clone instructions: {relative}')
    previous_url = re.compile(r'https://github\.com/TaeyanG4/(?:titanic-gpt-web-experiment|Kaggle_Titanic_practice)(?=[/\s.#`)]|$)')
    try:
        hero = ET.parse(ROOT/'docs/assets/hero.svg').getroot()
        hero_title = hero.find('{http://www.w3.org/2000/svg}title')
        if hero_title is None or hero_title.text != PROJECT_TITLE:
            errors.append('Cover title does not match the current project name.')
        if PROJECT_TITLE not in (ROOT/'tools/build_report_assets.py').read_text(encoding='utf-8'):
            errors.append('Cover generator does not contain the current project name.')
    except (ET.ParseError, FileNotFoundError) as exc:
        errors.append(f'Invalid cover SVG: {exc}')
    links = 0
    for path in docs:
        text = path.read_text(encoding='utf-8')
        if previous_url.search(text):
            errors.append(f'Outdated repository URL in {path.relative_to(ROOT)}')
        for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)',text):
            if target.startswith(('http://','https://','mailto:','#')): continue
            target = unquote(target.split('#',1)[0])
            if not target: continue
            links += 1
            if not (path.parent/target).resolve().exists():
                errors.append(f'Broken local link in {path.relative_to(ROOT)}: {target}')

    # A narrow, explicit scan is helpful but cannot prove the absence of secrets.
    patterns = [re.compile(r'gh[pousr]_[A-Za-z0-9]{36,}'), re.compile(r'github_pat_[A-Za-z0-9_]{50,}'),
                re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')]
    scanned = 0
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.parts: continue
        if path.name in {'kaggle.json','.env','access_token'}:
            errors.append(f'Forbidden credential file: {path.relative_to(ROOT)}')
        if path.suffix.lower() not in {'.py','.md','.json','.ipynb','.yml','.yaml','.txt'}: continue
        if path.stat().st_size > 5_000_000: continue
        text=path.read_text(encoding='utf-8-sig',errors='replace');scanned+=1
        if any(p.search(text) for p in patterns):
            errors.append(f'Potential credential pattern in {path.relative_to(ROOT)} (value not printed)')
    if (ROOT/'data/train.csv').exists() or (ROOT/'data/test.csv').exists():
        errors.append('Raw competition CSV found in publication checkout.')
    with (ROOT/'docs/evidence/kaggle-submissions.csv').open(encoding='utf-8',newline='') as f:
        receipts = list(csv.DictReader(f))
    if len(receipts)!=17: errors.append(f'Unexpected campaign receipt count: {len(receipts)}')
    if not any(r['ref']=='56841675' and r['publicScore']=='0.83014' for r in receipts):
        errors.append('Final Kaggle receipt is missing or changed.')
    if errors:
        print('\n'.join('FAIL: '+e for e in errors));raise SystemExit(1)
    print(f'PASS: {len(manifests)} submission artifacts, {len(inventory)} snapshot files, {len(receipts)} campaign receipts.')
    print(f'PASS: frozen v47 replay, notebook output policy, {links} local links, {scanned} text-file credential scans.')
    print(f'PASS: project titles, cover and current repository URLs use {REPO_SLUG}.')
    print('Scope: artifact/document checks only; not statistical significance, code safety or leakage certification.')

if __name__=='__main__': main()
