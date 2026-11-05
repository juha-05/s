# 클로드 전달용 10페이지 슬라이드 구성안

1. **제목/목적** — 전력망 안정성: 회귀와 분류 비교, 12회 안의 optimizer 선택. 핵심 질문 두 개를 제시.
2. **문제 정의와 사전 가설** — 회귀 `stab>0`, 분류 `p>=0.5`; H1=not_supported, H2=not_supported, H3=untested. `pre_registered_plan.md` 문구 유지.
3. **데이터/전처리** — 10,000행, 14열, stable 3,620, unstable 6,380; 항상 unstable 기준선 63.8%; 누출 `stab`과 중복 `p1` 제외, 11개 입력, 8:2 층화 분할. 그림: `figures/data_distribution.png`.
4. **공통 모델과 조건** — 11→64 ReLU→32 ReLU→1, Adam 0.001, batch 64, 50 epoch, 동일 초기 은닉 가중치/배치 순서, scaler는 학습에만 fit.
5. **회귀 vs 분류** — 테스트 정확도 회귀 0.9695, 분류 0.9600. 그림: `figures/goal2_learning_curves.png`, 두 confusion PNG. 표: `test_metrics.csv`.
6. **오판정/불확실성** — 경계 거리별 오류, 회귀 산점도. 분류-회귀 정확도 차이 -0.0095, 95% CI [-0.0185,-0.0005]. 그림: `figures/misclassification_analysis.png`; H1 회고.
7. **4+3+4+1 계획** — 4 optimizer 대표점→우승 optimizer 3 LR→상위2×2-fold→최종. 폭/신뢰도/비용 trade-off와 전역 최적 아님을 명시.
8. **12회 로그와 fold** — 실제 12/12회. 표: `goal3_run_log.csv`, `candidate_ranking.csv`, `cross_validation_results.csv`, `cross_validation_summary.csv`.
9. **최종 결과/비용/한계** — RMSprop lr=0.003, 42 epoch; 최종 정확도 0.9625. 실제 unstable 1,276개 중 59개(4.62%)를 stable로 오판정. 61회 전수 탐색과 비교, 2-fold와 학습량 차이 한계.
10. **H2/H3·AI·참고문헌** — H2=not_supported, H3=untested. AI 활용: 계획을 실행 코드로 변환; 직접 검증/수정 사례: 과제의 12회 제한에 최종 재학습 포함, scaler fold별 재fit, 회귀 accuracy 직접 계산, 목표 3 카운터 강제. 과제 PDF와 전달된 4개 연구 링크 기재.
