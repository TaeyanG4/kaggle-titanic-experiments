# 05. 재현 방법과 한계

[프로젝트 홈](../README.md) · [검증·무결성](03-validation-and-integrity.md)

## 1. 세 가지 재현 수준을 구분하기

| 수준 | 가능한 작업 | 보장하지 않는 것 |
|---|---|---|
| 자료 검증 | 해시·행 수·ID·지표 파일·문서 링크 검사 | 누출 없는 모델 인증 |
| 산출물 재조립 | 보존된 v10/v38/Deotte-derived 예측을 결합해 최종 v47 재구성 | 원시 데이터에서 재학습한 성능 |
| 모델 재학습 | 원본 데이터와 해당 의존성을 준비해 역사적 스크립트 실행 | 모든 실행 환경에서 bit-identical 결과, 독립 일반화 성능 |

빠른 검증과 최종 산출물 재조립은 외부 계정·GPU·Kaggle 인증 없이 수행하도록 새 도구를 추가했습니다. 원래 실험 스크립트의 동작은 문서화 과정에서 조용히 고치지 않았습니다.

## 2. 저장소 받기

```bash
git clone https://github.com/TaeyanG4/kaggle-titanic-experiments.git
cd kaggle-titanic-experiments
python tools/verify_publication.py
```

Python 3.12를 기록 환경으로 삼았습니다. 위 검증 도구 자체는 표준 라이브러리만 사용합니다. 최종 저장소명은 **`kaggle-titanic-experiments`**입니다. 기존 `Kaggle_Titanic_practice`와 정리 중 사용한 `titanic-gpt-web-experiment`의 커밋 이력은 유지했습니다.

이미 이전 이름으로 clone한 경우에는 기존 작업 폴더 안에서 remote만 갱신하면 됩니다. 로컬 폴더명 자체를 바꿀 필요는 없습니다.

```bash
git remote set-url origin https://github.com/TaeyanG4/kaggle-titanic-experiments.git
git remote -v
```

## 3. 최종 파일을 다시 조립하기

```bash
python tools/replay_final_artifact.py --output replayed_v47.csv
```

이 명령은 다음 세 보존 파일을 읽습니다.

- `submissions/submission_v10_score_0.81578.csv`
- `submissions/submission_v38_v10_text_rescue.csv`
- `submissions/submission_v24_v10_deotte_all_deotte_female_death.csv`

v38을 시작점으로 삼고, broad guard가 v10과 다른 행에만 broad guard의 예측을 적용합니다. PassengerId 정렬과 이진 label을 확인한 뒤 결과를 저장합니다. 입력 데이터가 변형되어 있으면 실패하도록 합니다.

최종 동결 artifact의 SHA-256은 다음과 같습니다.

```text
ce8e484730a667e0b5d80068cf8f1418444e0113a9af28996dd8185df60ea0e4
```

CSV의 줄바꿈만 달라도 byte hash가 달라질 수 있으므로 저장소는 제출 CSV를 Git에서 줄바꿈 정규화하지 않도록 설정했습니다. 재조립 도구는 예측 동일성과 직렬화된 byte hash를 함께 검사합니다.

## 4. 원시 데이터 준비

Kaggle 계정으로 [공식 Titanic 대회](https://www.kaggle.com/competitions/titanic)에 접근하고, 해당 이용 조건을 확인한 뒤 데이터를 내려받으세요. CLI 인증은 자신의 계정 설정으로 처리해야 하며 인증 파일을 저장소에 커밋하면 안 됩니다.

```bash
kaggle competitions download titanic -p data
python -m zipfile -e data/titanic.zip data
```

CLI 구문은 버전에 따라 달라질 수 있으므로 설치된 `kaggle competitions download --help`가 우선합니다. 압축 해제 후 `data/train.csv`, `data/test.csv`, `data/gender_submission.csv`가 필요합니다. [기록된 파일 지문](evidence/data-fingerprints.json)과 비교할 수 있습니다. 이 문서에는 토큰·쿠키·인증 파일을 포함하지 않습니다.

## 5. 환경과 선택적 모델

[environment-snapshot.json](evidence/environment-snapshot.json)은 종료 때 읽은 라이브러리 버전입니다. 모든 과거 실행 시점의 lockfile을 완전히 복원한 것이 아닙니다. 핵심 tabular 분기를 위한 버전 목록은 `requirements-core.txt`, 차트 생성을 위한 목록은 `requirements-report.txt`에 분리했습니다.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
```

TabPFN·TabICL·pytabkit·RuleFit 계열, PyTorch와 GPU 환경은 별도 선택 의존성입니다. 접근 승인·모델 파일·라이선스가 필요한 분기는 해당 제공자의 조건을 확인해야 합니다. 승인된 환경이 없으면 해당 실험을 건너뛰고 보고된 수치를 새로 재현했다고 주장하지 마세요. 체크포인트나 폰트 파일은 이 저장소에서 배포하지 않습니다.

## 6. 주요 분기의 실행 입구

| 목적 | 입구 | 주의 |
|---|---|---|
| 초기 그룹 타깃 감사 | `scripts/audit_group_survival.py` | 공식 데이터 필요 |
| Gunes 역사적 재현 | `scripts/reproduce_gunes_original_v10.py` | 누출 위험이 있는 OOF; 권장 pipeline 아님 |
| Deotte 재현·포트 | `scripts/deotte_wcg_xgb_v21.py` | 외부 공개 예측 상수와 Python 근사 구분 |
| EB 관계 피처 | `scripts/partial_pooling_v27.py` | 선행 representation·export 의존성 확인 |
| nested 선택 감사 | `scripts/nested_selection_audit_v29.py` | 비용이 큰 여러 모델을 사용; 독립 실험 아님 |
| P3 실험 | `scripts/ticket_prefix_ablation_v42.py` | parent와 model seed 차이에 주의 |
| 최종 제출 재조립 | `tools/replay_final_artifact.py` | 재학습 없음, 인증 없음, 제출 없음 |

각 역사적 스크립트는 다른 버전의 `exports/`를 읽을 수 있습니다. 경로 의존성은 소스에서 확인하고 필요한 선행 실험을 실행해야 합니다. 원시 승객 필드를 포함한 일부 파생 CSV는 출판 스냅샷에서 제외되어 있으므로 그런 파일은 원래 데이터에서 재생성해야 합니다. **모든 v1~v47을 한 명령으로 처음부터 재학습하는 검증된 runner는 제공하지 않습니다.**

기존 스크립트 실행은 해당 `exports/`를 덮어쓸 수 있으므로 새 clone이나 별도 브랜치에서 작업하세요. 저장된 `submission.csv`를 학습 결과로 무심코 덮어쓰지 마세요.

## 7. 차트 재생성

```bash
python -m pip install -r requirements-report.txt
python tools/build_report_assets.py
python tools/verify_publication.py
```

차트 입력은 저장된 Kaggle 제출 영수증, v42·v44 지표입니다. 새로운 실험·새로운 제출·새로운 점수 조회는 하지 않습니다. 그림에 숨겨진 test 정답을 넣지 않습니다. 표지 SVG는 설명용이며 통계적 근거가 아닙니다.

이전 대화에서 생성한 래스터 인포그래픽에는 무누출을 단정하는 문구와 실제 처리 범위를 오해하게 하는 표현이 있어 최종 보고서의 증거 그림으로 사용하지 않았습니다. 여기서는 수치 차트를 코드와 CSV에서 생성하고, 흐름도는 텍스트 기반 Mermaid로 유지합니다.

## 8. 검사 범위

`verify_publication.py`는 제출 스키마와 해시, 최종 파일 동일성, 원래 source snapshot 해시, 새 노트북 출력 제거, 주요 문서 링크를 확인합니다. 추가로 흔한 인증 문자열 패턴을 스캔하되, 모든 비밀 정보·코드 결함을 탐지한다고 보장하지 않습니다. ML 실험 전체를 다시 fit하거나 통계적 가정을 검증하는 도구는 아닙니다.

원래 2023년의 pickle 파일은 이력 보존용입니다. 이를 재현 도구가 로드하지 않으며, 출처를 신뢰할 수 없는 pickle을 임의로 역직렬화해서는 안 됩니다.
