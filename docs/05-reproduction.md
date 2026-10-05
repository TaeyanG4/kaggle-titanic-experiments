# 05. 재현 방법

[프로젝트 홈](../README.md) / [검증과 한계](03-validation-and-integrity.md)

## 파일 검증과 재학습을 구분하기

이 저장소에서는 보존된 파일의 해시를 확인하고, 기존 예측을 결합해 최종 CSV를 다시 만들 수 있다. 이것은 원시 데이터에서 모델을 다시 학습하는 것과 다르다.

| 작업 | 실행 내용 |
|---|---|
| 자료 검증 | 해시, 행 수, ID, 문서 링크와 흐름도 이미지 확인 |
| 최종 파일 재조립 | 보존한 v10, v38과 broad guard의 예측 결합 |
| 모델 재학습 | 공식 데이터와 해당 모델 의존성을 준비해 실험 코드 실행 |

빠른 검증과 최종 파일 재조립은 외부 인증이나 GPU 없이 실행한다. 모든 과거 실험을 처음부터 한 번에 재학습하는 runner는 제공하지 않는다.

## 저장소 받기

```bash
git clone https://github.com/TaeyanG4/kaggle-titanic-experiments.git
cd kaggle-titanic-experiments
python tools/verify_publication.py
```

Python 3.12 환경을 기준으로 기록했다. 위 검증 도구는 표준 라이브러리만 사용한다. 보존된 스냅샷을 검사하므로 모델을 재학습해 exports를 바꾼 작업 폴더가 아니라 새 clone에서 먼저 실행하는 편이 좋다.

## 최종 CSV 다시 만들기

```bash
python tools/replay_final_artifact.py --output replayed_v47.csv
```

입력으로 읽는 파일은 다음과 같다.

```text
submissions/submission_v10_score_0.81578.csv
submissions/submission_v38_v10_text_rescue.csv
submissions/submission_v24_v10_deotte_all_deotte_female_death.csv
```

v38에서 시작해 broad guard와 v10의 예측이 다른 행만 broad guard로 바꾼다. 결과를 저장하기 전에 PassengerId, 이진 label과 최종 예측의 동일성을 확인한다. 이미 같은 이름의 출력이 있으면 덮어쓰지 않는다.

동결한 v47 파일의 SHA-256은 다음과 같다.

```text
ce8e484730a667e0b5d80068cf8f1418444e0113a9af28996dd8185df60ea0e4
```

CSV는 줄바꿈만 달라도 byte hash가 달라질 수 있다. 제출 파일에는 Git 줄바꿈 정규화를 적용하지 않았고, 재조립 도구는 예측과 직렬화된 byte hash를 함께 확인한다.

## 원시 데이터 준비

재학습은 별도 작업 clone에서 진행하는 것이 좋다. [공식 Titanic 대회](https://www.kaggle.com/competitions/titanic)의 이용 조건을 확인하고 자신의 계정으로 데이터를 받는다. 인증 파일은 저장소에 넣지 않는다.

```bash
kaggle competitions download titanic -p data
python -m zipfile -e data/titanic.zip data
```

설치한 CLI의 구문이 다르면 `kaggle competitions download --help`를 확인한다. `data/train.csv`, `data/test.csv`, `data/gender_submission.csv`가 필요하며 [데이터 지문](evidence/data-fingerprints.json)으로 파일을 비교할 수 있다.

## 환경과 선택 의존성

[환경 스냅샷](evidence/environment-snapshot.json)은 종료 때 확인한 라이브러리 버전이다. 모든 과거 실행의 lockfile은 아니다. 핵심 tabular 분기는 `requirements-core.txt`, 차트 재생성은 `requirements-report.txt`에 나눴다.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
```

TabPFN, TabICL, pytabkit, RuleFit 계열과 PyTorch는 실행할 분기에 따라 별도 환경이 필요하다. 모델 접근 승인과 라이선스도 확인해야 한다. 체크포인트나 폰트 파일은 저장소에서 배포하지 않는다.

## 주요 실행 파일

| 목적 | 파일 | 주의할 점 |
|---|---|---|
| 그룹 타깃 감사 | `scripts/audit_group_survival.py` | 공식 데이터 필요 |
| Gunes 재현 | `scripts/reproduce_gunes_original_v10.py` | 원본 방식의 OOF에는 누출 위험이 있음 |
| Deotte 포트 | `scripts/deotte_wcg_xgb_v21.py` | Python 근사와 외부 공개 예측을 구분 |
| EB 피처 | `scripts/partial_pooling_v27.py` | 선행 표현과 export 의존성 확인 |
| Nested 선택 | `scripts/nested_selection_audit_v29.py` | 여러 모델을 재학습하므로 실행 비용이 큼 |
| P3 비교 | `scripts/ticket_prefix_ablation_v42.py` | parent와 변형의 model seed 차이 확인 |
| 최종 파일 재조립 | `tools/replay_final_artifact.py` | 재학습과 제출은 하지 않음 |

역사적 스크립트는 다른 버전의 exports를 입력으로 읽을 수 있다. 필요한 선행 실험은 코드에서 확인해야 한다. 원시 승객 정보를 포함해 공개에서 제외한 파생 파일은 공식 데이터로 다시 생성해야 한다. 재학습 시 기존 exports를 덮어쓸 수 있으므로 원래 보존본과 분리해 작업한다.

## 차트와 흐름도 재생성

점수 차트는 기존 CSV에서 생성한다.

```bash
python -m pip install -r requirements-report.txt
python tools/build_report_assets.py
```

흐름도는 Node.js 도구로 DOT 소스를 렌더링한다. 본문은 생성한 PNG를 표시하며 SVG와 소스도 함께 보관한다. GitHub의 Mermaid 실행 여부에 의존하지 않는다.

```bash
npm ci --prefix tools/diagram-renderer
npm run render --prefix tools/diagram-renderer
python tools/verify_publication.py
```

[흐름도 폴더](diagrams/README.md)의 manifest에는 소스와 이미지의 해시, 해상도, 렌더러 버전이 있다. 운영체제의 폰트에 따라 렌더링 bytes는 달라질 수 있어 재생성할 때 manifest도 갱신된다. 어떤 그림 생성 명령도 모델을 학습하거나 Kaggle에 제출하지 않는다.

## 검사의 범위

검증 도구는 파일 지문, 최종 예측 재조립, 노트북 출력 제거, 문서 링크, PNG와 SVG, 흔한 인증 문자열 패턴을 검사한다. 이 검사만으로 모든 비밀 정보나 모델 결함을 찾을 수 있는 것은 아니다. 원래 실험의 통계적 가정과 무누출성은 [검증 문서](03-validation-and-integrity.md)와 코드를 별도로 검토해야 한다.

이전 연습에 있던 pickle은 이력 보관용이며 검증 도구가 로드하지 않는다. 출처를 신뢰할 수 없는 pickle은 역직렬화하지 않는 것이 안전하다.
