# 흐름도 소스

본문에는 PNG를 넣었다. 브라우저에서 Mermaid 코드를 실행하지 않아도 그림이 표시되며, 확대용 SVG와 수정용 Graphviz DOT 파일을 함께 보관한다.

| 파일 | 내용 |
|---|---|
| `workflow.dot` | 초기 설정부터 실험, 승인, 제출과 기록까지의 흐름 |
| `v5-ensemble.dot` | v4b 블렌딩과 v5의 세 멤버 다수결 |
| `validation-boundary.dot` | 그룹 타깃 통계를 만들 때 필요한 학습과 검증 라벨의 분리 |
| `final-lineage.dot` | v10, v38, v46, v47의 예측 파일 구성 관계 |

Node.js 환경에서 저장소 루트를 기준으로 다음 명령을 실행한다.

```bash
npm ci --prefix tools/diagram-renderer
npm run render --prefix tools/diagram-renderer
python tools/verify_publication.py
```

렌더링은 Viz.js의 Graphviz 구현과 sharp를 사용한다. 모델 학습이나 Kaggle 접속은 하지 않는다. `manifest.json`에는 소스와 이미지의 해시, 해상도와 렌더러 버전을 기록한다. 폰트 파일은 배포하지 않으며 운영체제에 따라 글꼴 모양에 차이가 날 수 있다.
