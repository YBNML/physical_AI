# 09 · F/T 재검증 원시 findings (국문 요약본)

*(국문 요약본. 영문 원문 전문은 [09-recheck-findings-en.md](09-recheck-findings-en.md). 이 findings의 종합은 07-REVISION-01에 반영되어 있습니다.)*

근거 태그: **[M]** measured(직접 측정) · **[PR]** peer-reviewed · **[PP]** preprint · **[V]** vendor · **[I]** inference

---

## Q1. 손목 F/T 입력이 "학습된 저수준 모델"을 정당화하는가 — 학습된 IK 지름길 퇴화를 깨는가

### VERDICT — 부분적으로 그렇다
이전 판정은 메커니즘은 맞았지만 결론이 틀렸다. "삭제"가 아니라 **"residual로 강등 + 이미지 입력 삭제"**로 수정해야 한다. 손목 F/T는 진짜 비기구학적 부가 정보다: 접촉 상태·물체 stiffness/무게·슬립을 담는 시스템 유일의 입력이며 (목표 포즈, 관절)에서 유도 불가능하다. 접촉 중심 태스크에서 "위치 출력 정책의 **입력**으로서의 힘"이 변혁적이라는 실증도 명확하다 — **FACTR 21.3% → 61.2%(naive) → 87.5%(curriculum)**(미학습 물체, 관절 위치 출력+PID), FILIC peg-in-hole 68→90%(sim)·실기체 +33~+53pp, sensorless-bilateral ACT 너트 돌리기 0/5→5/5(힘은 입력만). 즉 "IK가 쓸 수 없는 정보를 갖는가"의 답은 그렇다. 그러나 셋이 설계를 제한한다. **(1) 정보이론적 퇴화는 깨졌지만 최적화 퇴화는 안 깨졌다.** next-joint-angle MSE의 ~99%는 IK(목표 포즈)로 설명되고 wrench 기여는 소수 timestep에 집중된 작은 residual이며(FIRST: 힘 제어 활성 = 접촉 스텝의 84%, 자유공간 스텝의 2.9%), 측정한 모든 논문에서 아키텍처가 강제하지 않으면 정책은 힘을 무시했다 — **ForceVLA naive concat +2.9pp(37.3→40.2) vs 구조화 fusion +23.2pp**, FoAR의 naive concat은 한 태스크에서 vision-only보다 악화. 확정 다이어그램의 flat concat은 ~+3pp가 나올 확률이 가장 높은 구성이고, Model 2는 FACTR 레짐(소형 ACT)이 아니라 ForceVLA 레짐(pretrained vision encoder)이다. **(2) 출력이 stiffness·wrench 목표·selection mask 없는 맨 관절 위치다.** ALOHA 메커니즘(F ≈ Kp·(q_cmd − q_meas))은 실재하고 위치 출력 정책의 작동은 입증됐지만, 그것은 힘 인지 **이산 행동**(접촉 감지·모드 전환·재시도·그립 적응)이지 연속 힘 **조절**이 아니다. FACTR 스스로 "recognizes contact events and switches strategies, rather than continuously regulating applied force magnitude" (접촉 이벤트를 인식해 전략을 바꿀 뿐, 힘 크기를 연속 조절하지 않는다). 진짜 조절을 이룬 시스템은 전부 Model 2에 없는 것을 추가했다: impedance 내부 루프(FILIC 2kHz), reference wrench+축별 selection mask(Force Policy), hybrid 위치-힘 제어기(Tactile-VLA), 수작업 reactive 규칙(FoAR ε=0.006m). **(3) 속도.** 측정된 M4에서 이미지 포함 Model 2는 ViT 전체 forward — 단독 62.75ms, 경합 시 p50/p99 214/289ms — 즉 3~5Hz 힘 루프이고, DLS IK 5.31µs·Robotiq FT300 출력 100Hz와 대비된다. 3Hz 루프는 힘 제어기가 아니다. baseline도 가정보다 강하다: Minimalist Compliance Control(2026)은 학습·F/T 센서 없이 모터 전류만으로 sensorless admittance를 구현해 egg-on-bread 40→80%; 고전 힘 제어는 잘 정의된 삽입을 이미 푼다(0.1mm 틈새 100%, 20µm ~4s). 순결론: wrench는 학습 컴포넌트를 정당화하지만, 그 정당한 형태는 analytic IK + admittance 위의 **작고 빠른, 이미지 없는 residual**이지 IK를 재구현하는 이미지 조건 monolith가 아니다.

*(참고: 위 3~5Hz는 이후 Q5 재측정에서 MPS placement 아티팩트로 판명 — CPU 배치 시 59.5Hz p50. 또한 REV.2에서 v1에서는 학습된 Model 2 자체를 제외하는 것으로 권고가 갱신됨.)*

### 핵심 findings
| 핵심 주장 | 수치 | 태그 |
|---|---|---|
| 힘은 입력만(출력은 위치)으로도 작동 — 설계의 최강 근거 | FACTR 21.3→61.2→87.5%; sensorless-bilateral: 너트 0/5→5/5→5/5, 오이 0/5→3/5→2/5(힘 출력은 해로움) | PP |
| 위치 출력 정책은 SWITCHING만, 연속 REGULATION 불가 | FoAR reactive step ε=0.006m; FACTR 회복 13.3→90% | PP |
| naive concat(다이어그램 그대로)은 이득 ~0일 확률 최대 | ForceVLA 37.3→40.2(+2.9pp) vs MoE 60.5(+23.2pp); pre-VLM 주입 0%; FoAR wiping 0.500→0.475 | PR |
| 힘 무시는 최적화 문제 — 자유공간에서 힘≈0이라 gradient가 주변화 | FIRST 접촉 84±10.8% vs 자유공간 2.9±2.1%; curriculum 없으면 vision attention 80–90% | PP |
| 지름길 깨는 3법(curriculum·contact gating·reweighting) — 다이어그램에 전무 | FACTR curriculum +26.3pp; FIRST pre-contact 0.818 vs contact-only 0.670 | PP |
| (포즈,관절)→관절 = IK; wrench 기여 중 접촉 상태·물체 속성 추론만 학습 정당, 힘 조절은 admittance가 해결 | DLS IK 5.31µs vs Model 2(이미지) 62.75ms / 214/289ms ≈ 4만 배 | M |
| 물체 속성 일반화가 가장 설득력 있는 실제 이득 | Bi-ACT 미학습 물체: 아이크림 50→100%, 글루 자 50→80% | PP |
| 최강 baseline = sensorless admittance(센서·학습 불요) | egg-on-bread 80% vs 40%; 힘 추정 오차 0.69±0.73N | PP |
| residual 구조가 정답: q_out = IK_DLS + Δθ(q, wrench, target) | TER-DAgger 77.2% vs BC(ACT) 27.6%, demo 100+50 | PP |
| Model 2의 헤드 이미지가 설계 최악 요소 — 삭제 시 38차원 MLP/GRU 수백 Hz | 12+14+12=38차원; FT300 상한 100Hz | M |
| bilateral(힘 피드백) 텔레옵이 전제조건일 가능성 | ALPHA-α rig $8,951 vs ALOHA $20,485 — 힘 피드백 포함이 더 저렴 | PP |
| 센서 실무: 100Hz 상한, 동적 보상 필수 | FT300-S: 보상 후 정지 <0.5N, 이동 중 ~2N residual, ~$1,950 | V |
| 전류 기반 외력 추정이 경쟁력 — 구매 재검토 | NEXT: Franka 접촉 L1 0.547±0.348Nm(DOB 대비 62.8% 우위), Piper 0.018±0.012Nm, $0 | PP |
| 프레임 불일치(wrench=손목, 목표=헤드-이미지) + 그리퍼 부재(문헌 힘 이득의 큰 몫이 그립력) | 헤드 extrinsics 6-DoF 누락; ALPHA-α 이득이 그립 조절에 집중 | I/PP |
| 반례 실재 | naive concat 효과 범위 −2.5pp~+39.9pp; ForceVLA early fusion 0% | PP |

### WHAT CHANGES 핵심
- 'Model 2 삭제'는 철회, 'residual 강등'은 강화: q_out = IK_DLS(target, q) + Δ(q, wrench, target) — 지름길을 구성상 제거.
- **Model 2 입력에서 헤드 이미지 삭제** (최고 가치 변경).
- 접촉 phase gate 또는 modality curriculum을 **필수**로 (flat concat = +2.9pp 구성).
- 축별 stiffness/reference-wrench 출력 추가, 또는 "switching만 한다"를 명시적 수용.
- **텔레옵 rig 힘 피드백 여부를 F/T 투자 전에 확인** — 없으면 데이터에 학습 가능한 힘 구조가 없음.
- zero-wrench ablation(zero/shuffle)을 go/no-go 게이트로 + 미학습 물체 평가(학습 물체에선 전 방법 동률).
- 센서 구매 vs 전류 추정 재검토. *(참고: REV.2에서 Galbot G1 손목 F/T 표준 탑재 확인 — 논점 소멸)*
- baseline은 "IK + sensorless admittance"로, delta≈0일 때의 대응 규칙을 사전 정의.

### 실행에 영향 주는 UNCERTAINTIES
- 팔 제어 인터페이스 미확정 — torque/impedance 가능이면 FILIC식이 정답, 위치 전용이면 switching만 가능. *(참고: 이후 REV.2에서 위치 전용으로 확정)*
- 텔레옵 rig 힘 피드백 여부가 사실상 결정적 — 최우선 확인.
- "학습 모듈 vs IK+admittance+탐색 primitive" 동일 조건 비교는 문헌에 없음 — 사내 실험 필요.
- wrench는 sim 전이 최난이도 신호 — 검토한 힘 정책은 전부 실데이터 학습. sim 계획은 자유공간 phase로 국한될 수 있음.

---

## Q2. 7-DoF×2 중복성 — 8/94/96 검증, arm-angle 자동 라벨, QP-IK, L2 평균 붕괴의 기하학

### VERDICT — 8/94/96은 정확히 확인, 단 적용 위치가 틀렸다
**8% / 94% / 96%는 정확히 확인** — Mazzaglia et al., "Redundancy-aware Action Spaces for Robot Learning"(RA-L 2024, arXiv:2406.04144) Table I 행 평균: Task 0/40/0/0/0=8.0%, ERJ=94.0%, Joint(oracle)=96.0%; 실기체 Franka 5태스크, demo 40개·trial 10회, ACT(chunking+temporal ensembling). 단 두 조건 없이 인용하면 정당하게 기각된다: (i) 5태스크는 **전체 구성 제어가 필요하도록 의도 선정**됐고(캐비닛 0.3m 진입, 팔꿈치로 버튼), 같은 논문의 표준 RLBench 8태스크에서는 task space가 완전히 경쟁력 있으며 ERJ는 "completely in line with task space"; (ii) 단일 팔이고, task-space baseline은 posture task 없는 기본 파라미터 MoveIt+pick_ik — 무규제 IK의 최악 버전. 결정적으로, 확정 I/O상 **Model 2는 14관절 입출력의 joint-space 정책이므로 8%는 Model 2에 적용되지 않는다.** 중복성 결함은 전적으로 Model 1에 있다: 상태·액션이 포즈 전용이라 대뇌는 자기 팔꿈치를 관측·명령할 수 없고, 자기 계획의 실행 불가능성을 감지할 수 없으며, 포즈 전용 상태는 14-DoF 플랜트에 대해 Markov가 아니다. 장애물 인지 posture 선택은 전부 Model 2에 암묵 위임되는데, 헤드 이미지에 팔꿈치 관련 장애물이 안 보이는 경우가 잦다. 수정은 싸고 무손실: **팔당 스칼라 1개(base joint j1 또는 stereographic SEW 각 ψ)를 Model 1 입출력에 추가**(14→16개). 기록된 관절에서 FK로 자동 라벨(~34µs/샘플, M4 측정 — 1M step ~35초, 인력 0)이고 ERJ 논문상 비구속 태스크에서 비용 0. 별도로 중복성보다 크다고 보는 결함 2개: 포즈가 헤드-이미지 프레임인데 헤드 관절 상태가 I/O에 없음(헤드가 움직이면 Model 1의 autoregressive rollout이 ill-posed), 그리고 손목 F/T는 팔꿈치/전완 충돌을 감지 못함 — 확공간 7-DoF 운용이 만드는 바로 그 접촉 모드.

*(참고: 이후 REV.2에서 대상 팔이 canonical S-R-S 7-DoF(closed-form IK 존재)로 확정 — SEW/arm-angle parameterization이 그대로 적용됨.)*

### 핵심 findings
| 핵심 주장 | 수치 | 태그 |
|---|---|---|
| 8/94/96 = Table I 행 평균, 정확 재현 | Task 8.0 / ERJ 94.0 / Joint 96.0%; 10 trial/태스크 | PR |
| 8%는 worst-case 선정 레짐 — 표준 스위트에서 스칼라 추가 비용 0 | RLBench 8태스크: ERJ ≈ task space; per-step 계산 task 12 / ERA 13 / ERJ 19ms | PR |
| 8%는 Model 2에 부적용(joint-space, 96% 열) — 결함은 Model 1 | Model 1: 12 in/out vs 플랜트 14 DoF — 팔당 1 DoF 관측·명령 불가 | I |
| SEW/arm angle은 기록 관절에서 FK closed-form 추출 — 라벨링 무비용 | M4: FK 16.9µs, ψ 라벨 34.4µs/샘플 → 1M step ~35s | M |
| 알고리즘적 특이점은 정리상 불가피(Hairy Ball) — stereographic SEW는 half-line으로 축소, 워크스페이스 밖 배치 가능 | conventional = full line vs stereographic = half-line | PR |
| **같은 self-motion manifold 위 두 유효 해의 L2 평균은 manifold 밖** | Δψ→EE 오차: 10°→1.9mm / 20°→8.0 / 30°→17.6 / 45°→38.5 / 60°→50.3 / 90°→96.2 / 120°→145.7mm; 고정 EE 포즈에서 **팔꿈치 스윕 박스 0.467×0.438×0.240m**, Δq1=316°, elbow 원 반경 0.239m | M |
| 붕괴 리스크는 유계 — Model 2가 현재 q 조건이므로 관건은 demo의 ψ 잔여 스프레드 | 스프레드 20°→~8mm(생존) / 60°→~50mm(치명) | I |
| QP-IK는 14-DoF bimanual도 실시간 여유 — 단 tail이 p50의 >10배 | iKinQP: 단일팔 median 0.26ms(std 3.3–4.6ms), 양팔 상호회피 1.47–1.64ms | PR |
| 데이터 파이프라인의 redundancy 해소 방식이 정책 품질로 전파 | SEW-Mimic: self-collision ~50%→~1.3%; 그 데이터로 학습 시 mink-IK 데이터 대비 ~3배 성공률 | PP |
| 주요 bimanual IL 중 최종 액션이 EE 포즈인 시스템 전무 — 대부분 6-DoF라 문제 미직면 | ACT/ALOHA 14차원 관절; π0 전 플랫폼 joint space(7-DoF Franka 포함); RDT-1B 팔당 관절 슬롯 10; ManiDP posture guidance 56.33→85.00% | PR |
| 6-DoF(무중복)에서는 EEF와 joint 동률 — ERJ 격차는 순수 redundancy 효과 | 13,000+ 실측 rollout: EEF-delta 89.6 vs joint-delta 88.0%(1.6pt=노이즈); chunk-wise delta가 step-wise 대비 최대 10pt, O(1) vs O(k) | PP |
| 스칼라는 base joint(j1)로 — 짝수(bending) 관절은 성능 붕괴; ERJ > ERA | j1 "by a large margin"; 8-DoF 확장에서도 유지 | PR |
| redundancy보다 큰 결함 2개: 헤드-이미지 프레임 + 손목 F/T의 팔꿈치 접촉 무감지 | elbow 이동 ~0.47m at 고정 EE 포즈; wrist F/T의 elbow 감도 = 구조상 0 | I |

### WHAT CHANGES 핵심
- **'8% vs 94%'를 이 아키텍처 비판으로 인용 중지** — Model 2는 96% oracle 열. 오인용 시 리뷰 전체가 기각됨.
- 진단을 Model 2 → **Model 1로 이전**; 권고는 "joint space 전환"이 아니라 "**팔당 스칼라 1개 추가(12→14)**", ERA보다 ERJ(j1).
- no-downside 논거 명시(표준 스위트 비용 0 = 쉬운 yes).
- **pre-build 진단**: 기존 데이터에서 ψ 히스토그램(~35s/1M step); 조건부 스프레드 <20°→8mm 생존, >60°→스칼라 필수.
- IK 비용 논거 폐기(양팔 QP ~1.5ms median), 단 p99 예산·CPU 유지 경고.
- SEW-Mimic "데이터 파이프라인 전파" 증거 + "아무도 포즈를 출력하지 않음" 컨센서스 체크 추가.
- 헤드-이미지 프레임·팔꿈치 접촉을 동등 이상 우선순위로 승격.
- ψ를 쓴다면 **stereographic SEW**(Elias & Wen Eq. 26) 필수 — conventional 정의는 워크스페이스 일부에서 라벨 불연속.

### 실행에 영향 주는 UNCERTAINTIES
- bimanual ERJ 비교는 문헌에 없음 — 격차 방향 단정 금지. 통계력도 얇음(10 trial): 8/94/96을 정밀 효과크기로 취급 금지.
- manifold 수치는 Franka 1개 포즈 기준 — 실제 URDF로 재계산 후 인용.
- **헤드 카메라가 팔꿈치 관련 장애물을 실제로 보는가**가 논거의 경첩 — 기존 녹화 검토로 반나절에 답 가능.
- 현 파이프라인의 IK seeding/posture task 여부 미확인 — 이미 있다면 실질 격차는 논문 baseline보다 작을 수 있음.

---

## Q3. Model 2는 헤드 관절 상태 입력이 구조적으로 누락인가 — 이미지에서 head→base 변환 복구 가능성과 취약성

### VERDICT — 방향은 맞지만 강한 주장("식별 불가")은 부정확하고, 그 차이가 처방을 바꾼다
설계는 명세대로라면 결함이다 — 그러나 Model 2는 정보 기아 상태가 아니다. 14관절과 헤드 이미지를 받고 자기 손이 대개 보이므로 B_T_H = FK_B(q_arm)·(H_T_hand)⁻¹로 변환이 원리상 복구 가능하다(markerless eye-hand self-calibration, iCub/Vicente 계열). 즉 변환은 **약하게 관측가능**하다. 결함의 본질: 이 채널은 (i) 파지 물체에 손이 가려지거나 FOV를 벗어날 때 — 헤드가 움직이는 바로 그때 — 실패하고, (ii) 광축(깊이) 방향에서 조건이 최악이며, (iii) 어떤 손실 항도 요구하지 않고, (iv) 학습 중 훨씬 싼 지름길 — **상수 B_T_H 암기** — 에 밀린다. 이 지름길은 실증 문서화됐다: "Do You Know Where Your Camera Is?"(arXiv:2510.02268, ICRA 2026) — 정책은 정적 배경 단서로 카메라 포즈를 추론하고 "this shortcut collapses when workspace geometry or camera placement shifts" (배치가 바뀌면 붕괴). 결과는 최악의 실패 유형: 학습·검증 우수, 현장에서 조용한 드리프트. 정량적으로 측정 p50/p99 214/289ms에서 30°/s 헤드 팬은 0.5m 리치에서 **56–76mm** 횡오차 — DLR이 조작 신뢰성에 요구한 ~3mm와 대비. 별도 발견: **Model 1도 오염된다** — 미래의 헤드-프레임 포즈를 예측하는데 헤드가 움직이면 "t+k의 헤드 프레임"은 "t의 헤드 프레임"과 다른 프레임이므로 재투영 없이는 학습 라벨 자체가 비일관(EgoVLA가 정확히 이 문제를 world-frame 카메라 포즈로 해결). 조건부 결론: 에피소드 동안 헤드가 고정이면(학습=배포 동일) 결함이 아니며 헤드-프레임은 오히려 실증적으로 **우월**(OC-VLA). 목이 살아있는 순간 B_T_H는 잠재 외생 변수가 되어 "옳고 더 나음"에서 "조용히 편향됨"으로 격하된다. 올바른 수정은 Model 2에 헤드 관절을 넣고 학습을 기대하는 것이 아니라 — **두 모델 사이에서 변환을 analytic하게 수행**하는 것이다.

### 핵심 findings
| 핵심 주장 | 수치 | 태그 |
|---|---|---|
| 변환은 보이는 손으로 약하게 관측가능 — 단 필요할 때 정확히 실패(가림·FOV 이탈·깊이 조건·손실 무요구) | 손 가시율 데이터 없음 | I |
| 배경 단서 지름길 실증 — 처방은 Plücker ray 명시 주입 | ACT/DP/SmolVLA 전반 개선(세부 수치 비공개) | PR |
| OC-VLA 이득 확인, **'14% vs 29%' 열화 주장은 오류** | +13.8/+8.0/+10.0pt 확인; novel view 하락 OC-VLA 14.0 vs OpenVLA-OFT 21.3 vs Robot Base 16.7 vs π0 16.0pt; '29'는 논문에 없음 → 이점은 ~2.7–7pt | PR |
| OC-VLA는 정적 카메라+기지 extrinsics에서만 성립 — 현 설계는 위험한 절반만 채택 | 정적 D435i 3대, 배치마다 재캘리브레이션; 이동 카메라 분석 전무 | PR |
| EgoVLA(실제 움직이는 헤드)는 world-frame 카메라 포즈로 미래 라벨 재투영 필수; 변환은 네트워크 밖 analytic | ~500k 쌍 | PP |
| 능동 인지 시스템 2/2가 헤드 포즈를 관측에 넣고 팔 액션은 world/base frame | ActiveUMI HMD 6-DoF 입력; ViA proprio 23차원(목 포즈 포함), ~45% 개선(능동 vs 고정 비교임) | PP |
| Helix는 헤드를 정책 출력으로 만들어 해소; 토르소 이동 시 체인이 목+토르소에 걸침 | S1 200Hz, 상체 35 DoF | V |
| **latency×헤드 속도 → 횡오차 표: sync 항이 오차 예산 지배** | r=0.5m, dt=60/100/214/289/500ms: 10°/s→5.2/8.7/18.7/25.2/43.6mm; 30°/s→15.7/26.2/56.0/75.7/130.9mm; 60°/s→31.4/52.4/112.1/151.3/261.8mm; 기준선 DLR 3.1mm | M |
| 정적 캘리브레이션 오차가 두 번째 구성 의존 bias — 목 sub-degree는 어려움 | DLR Agile Justin 21→3.1mm; 목 1° = 8.7mm @0.5m | PR |
| 지배 하이브리드: Model 1은 헤드 프레임 유지(캡처 시점 재투영) + 모델 사이 analytic B_T_H + Model 2 입력은 base frame(관절 출력과 일치); 중력정렬 프레임이면 F/T 중력보상도 상수화 | analytic SE(3) 합성 µs급 vs 추론 214ms; base 직접 예측 비용 8–13.8pt는 포즈 출력 모델에만 해당 | I |

### WHAT CHANGES 핵심
- 프레이밍 격하: "입력 누락"이 아니라 "**암기된 상수가 이기는 비지도 self-calibration**" — q_head 추가만으로는 불충분.
- **두 모델 사이 analytic 프레임 변환 삽입**: Model 1은 헤드 프레임 유지, Model 2 포즈 입력은 base_link.
- 헤드/목(+가동 시 토르소) 관절 상태를 **이미지 캡처 타임스탬프와 함께** 기록 — 타임스탬프 자체가 현재 부재.
- **Model 1 라벨 수정**: EgoVLA 방식으로 미래 손 위치를 캡처 시점 헤드 프레임으로 재투영.
- OC-VLA 인용 수정: '+8.0/+13.8/+10.0' 유지, '14 vs 29' 폐기 → 14.0 vs 21.3/16.7pt.
- v1: **freeze-the-head를 runtime assertion 있는 시스템 invariant로** (헤드 속도 초과 프레임 거부/플래그).
- latency 예산에 헤드 모션 오차 항 명시 — 3.1mm 목표 대비 지배 오차원.
- 능동 인지가 필요해지면 ActiveUMI/ViA/Helix 방식: 헤드 포즈를 정책 출력+입력으로, 팔 액션은 중력정렬 world/base frame.

### 실행에 영향 주는 UNCERTAINTIES
- "이동 카메라 + 카메라 포즈 유무"의 정확한 ablation은 문헌에 없음 — 페널티 방향만 문서화, 크기 미상.
- 태스크 중 손의 이미지 내 가시율이 self-calibration 채널의 생사 결정 — 기존 로그에서 FK 투영으로 측정 가능.
- 헤드 actuation·토르소 이동 여부에 판정 전체가 조건부. *(참고: 이후 REV.2 확정 — head→arm_base는 목 관절 2개에만 의존하나, 목 틸트 40.6°뿐이라 조작 중 토르소 상시 이동 → "헤드 프레임 정적" 가정은 성립하지 않음)*
- 텔레옵 데이터의 실제 헤드 각속도 분포 미측정.

---

## Q4. 누락 필드들(그리퍼·타이밍·stiffness·양손 상대 포즈·chunking)의 실측 비용과 최소 추가 집합

### VERDICT — 셋은 대규모 실측 열화, 하나는 구조적 불가능
그리퍼 명령은 나올 곳이 없다: Model 1은 6-DoF 포즈, Model 2는 팔 관절 14개를 출력하며 개발자의 "7-DoF×2=14"는 전부 팔이다 — 겉보기에 같은 ACT/ALOHA의 14는 "joint positions for two robot arms (7+7=14 DoF)"이되 6-DoF ViperX 팔이므로 **12 팔 관절 + 2 그리퍼 관절**이다. 현 설계는 손을 여닫으라고 물리적으로 명령할 수 없으므로 파지 성공률은 저하가 아니라 **0%**다. 같은 형태의 모든 발표 next-best-pose 아키텍처(HDP의 a_high=(a_pose, a_grip), PerAct, RVT)는 그리퍼를 정식 필드로 유지한다 — Model 1은 자기 계열에서 유일하게 이를 버렸다. 둘째, per-step 인터페이스는 문헌 최대의 실측 열화: ACT 자체 ablation이 "performance improves drastically from **1% at k = 1 to 44% at k = 100**" — 다이어그램이 함의하는 바로 그 k=1 형태에서 44배이며, 두 vision backbone이 경합해 루프가 ~2–3Hz인 M4에서는 치명적으로 더 나쁘다. 셋째, 타이밍/duration: 시간 없는 waypoint는 궤적이 아니고, 저수준은 속도를 Δpose ÷ (jitter 섞인 도착 간격)으로만 추론하는데, 접근 속도는 충돌력에 선형 비례함이 실측됐다(0.02–0.16m/s, 각 4회) — 즉 214/289ms의 추론 jitter가 동일 명령 포즈에서 접촉력을 회차마다 **~35%** 변조한다. 결정적으로 같은 논문이 stiffness는 충돌력에 "no significant effect"임을 측정했으므로 compliance는 누락된 duration을 구제하지 못한다. 넷째, 양손 상대 포즈는 UMI bimanual 셔츠 접기에서 **70% vs 30%**의 실측 비용이고, 절대 포즈 2개의 뺄셈으로 복구 불가 — 학습 손실이 무해한 common-mode 오차와 물체를 부수는 differential 오차를 구별하지 못하기 때문. 예상 비용 순위: **그리퍼(하드 0) > chunking+타이밍(44배, 같은 수정) > 상대 포즈(2.3배, bimanual의 전제) > stiffness/모드(현재 손목 F/T를 dead input으로 만듦)**. 최소 추가 = 4개 필드: 손별 연속 aperture, H개 미래 waypoint chunk, waypoint별 duration, 명시적 상대 포즈.

### 핵심 findings
| 핵심 주장 | 수치 | 태그 |
|---|---|---|
| 개발자의 14 ≠ ACT의 14 (12 팔 + 2 그리퍼) | 부족분 = 어떤 모델도 쓰지 않는 액추에이터 채널 2개 | PR |
| 그리퍼 명령의 원천이 다이어그램에 없음 — 구조적 불가능 | 파지 0%; HDP a_high 8차원 중 빠진 것이 정확히 a_grip | PR |
| 그리퍼 오차는 latch — 실패를 관측 못 하고 조용히 진행 | pseudo-tactile 피드백: 교란 내성 10–30%→100% (상태 **입력**도 필요) | PP |
| 그리퍼/팔 타이밍 어긋남 ≈ 1스텝 = 30pt 하락 | UMI tossing: latency matching 유 87.5% → 무 57.5% | PR |
| 연속 aperture는 암묵적 그립력 채널 — 손목 F/T로 대체 불가 | soft finger 변형 조절 = 힘 제어(UMI HD5) | PR |
| 다지 핸드면 누락이 2 스칼라 → 24–44차원 + pre-shaping 결정 | Inspire 12 / Allegro·LEAP 16 / Shadow 22 DoF | PR |
| k=1 per-step은 문헌 최대 실측 열화 — 다이어그램은 정확히 k=1 | **ACT 1%@k=1 → 44%@k=100**; DP Ta=8 최적, Ta=1 유의미 열화 | PR |
| 이 M4에선 더 나쁨 — 블라인드 간격 ~21배 | 2 backbone 직렬 경합 ~430ms p50/578ms p99 = 1.7–2.3Hz vs ACT 50Hz | I |
| duration 없는 waypoint → 접근 속도가 GPU 스케줄링의 함수; 속도 ∝ 충돌력(선형 실측) | 0.02–0.16m/s 선형, 목표 힘 10–20N ±~3N; jitter → ~35% 충돌력 산포 | PR |
| stiffness는 duration 대체 불가(실측 null) — 충돌력 불변, 접촉 후 jerk만 완화 | "no significant effect" | PR |
| UMI 70% vs 30% inter-gripper 확인 — 타이밍 실패를 공간 필드가 치유; "a critical ingredient" | 14/20 vs 6/20 | PR |
| 상대 포즈는 뺄셈 복구 불가 — 결정적 이유는 손실 대칭성(common-mode=differential 동일 페널티 → gradient 압력 부재) | UMI 등록 정확도 10.1mm/0.8° | I |
| 최소 추가 4필드 + Model 2에 2단 stiffness/모드 | H=8 기준 ~192 float, <0.1ms vs 인코더 60.6ms — 성능 반론 불성립 | I |

### WHAT CHANGES 핵심
- "ACT의 14차원과 같다" 서술 중지 — ACT는 현 스펙의 반증.
- 그리퍼를 "누락 필드"가 아닌 "**누락 액추에이터 채널**"(blocking 결함)로 재분류.
- aperture는 **연속**, 그리퍼 **상태**(또는 파지력)를 입력으로도 — 명령만 추가하면 절반의 수정.
- chunking과 timing을 단일 그룹으로 병합: "**dt를 갖는 H개 waypoint**".
- "상대 포즈는 하류에서 뺄셈" 주장 폐기 — 손실 대칭성 논거를 선두에.
- **손목 F/T는 현재 dead input**: 감지한 접촉에 대응할 출력 채널이 없음 — stiffness/모드 출력을 추가하거나 센서를 빼라.
- "compliance로 타이밍 대체" 탈출구를 실측 null로 봉쇄.
- 1.7–2.3Hz 플랫폼에서 chunking은 최적화가 아니라 동작 연속성의 전제. *(참고: 이 rate는 이후 Q5에서 CPU 배치 기준으로 갱신됨)*
- '양손'이 parallel-jaw인지 다지인지 확정 전에 필드 수 인용 금지(2 vs 24–44차원).

### 실행에 영향 주는 UNCERTAINTIES
- '양손' = parallel-jaw vs 다지 미해결 — 수정 규모를 자릿수로 좌우.
- ACT k=1 ablation은 sim 4세팅 평균 — 방향 확실, 크기는 외삽.
- Model 2 아래 별도 time parameterizer가 있으면 비결정성 일부 흡수 — 단 "특정 접근 속도를 명령 못 함"은 잔존.
- F/T가 다이어그램 밖 별도 admittance 루프에서 소비된다면 stiffness 누락 심각도는 하락.

---

## Q5. 확정 스펙에서 M4 Mac mini 32GB 컴퓨트 재측정 — Model 2 Hz, CPU 한계선, 인코더 공유, 고전 IK

### VERDICT — 확정 스펙은 여유 있게 실행 가능; 이전의 비관은 방향만 맞고 처방이 틀렸다
결정적 측정 사실: **Model 2는 GPU에 살면 안 되지만, vision encoder를 빼앗길 필요도 없다.** Model 1(263.8M, ViT-B/16@224 + 16L d=960 trunk)이 MPS를 포화시키는 동안, 자체 ViT-S/16@224 + (14관절+6 wrench+14포즈) MLP의 Model 2는 **CPU에서 59.5Hz p50 / 50.4Hz p99 / 최악 28Hz**로 돌고 Model 1은 4.45→4.2Hz로만 하락한다. 같은 Model 2를 MPS에 두면 15.5Hz p50 / 11.5Hz p99로 붕괴하고 Model 1도 3.4Hz로 끌려간다 — 상호 파괴. 원인은 M4의 batch-1 추론이 FLOP-bound가 아니라 **dispatch-bound**라는 것: batch-1 ViT-B forward는 GPU 지속 3.78 TFLOPS의 0.565(15%)만 달성하고, 고립 MPS dispatch 왕복 ~179µs vs CPU ~1µs. 따라서 CPU(AMX/Accelerate 1.90 TFLOPS)는 **테스트한 모든 Model 2 크기에서 MPS를 이긴다** — GPU에 fp16, CPU에 fp32를 주고도(ViT-B 44.7ms CPU vs 64.2ms MPS). 인코더 공유의 절감은 14.6ms/cycle(MPS 6.0%, CPU 9.5%)뿐이라 컴퓨트 최적화로 채택하면 안 되고 — Model 2의 vision rate가 Model 1의 4–5Hz에 묶이는 대가 — 진짜 가치는 **cached token 위의 Model 2 head가 0.197ms(5076Hz p50 / 4149Hz p99)로 >1kHz F/T reflex 경로를 사실상 공짜로 준다**는 것. 고전 경로는 논점이 아니다: bimanual DLS+null-space IK 16–21µs p50, p99 <28µs. 새로 확정된 F/T 입력은 공짜(6 스칼라 → MLP, 측정 불가 수준). 열 상태는 내내 깨끗(~15분 지속 부하 후 경고 0) — 전 수치 un-throttled 정상상태.

*(참고: 이후 REV.2에서 GalbotSDK가 Linux 전용이라 Mac은 로봇과 통신 불가하고 실기체 추론은 로봇 탑재 AGX Orin에서 수행됨이, REV.3에서 학습·시뮬은 RTX 3090/64GB/Ubuntu 박스로 확정됨. 본 M4 수치는 배포 수치가 아니라 "batch-1 추론은 dispatch-bound"라는 구조 결론과 모델 사이징 참고자료로 읽을 것.)*

### 핵심 findings (전부 [M])
| 핵심 주장 | 수치 |
|---|---|
| **헤드라인**: 자체 vision 인코더 포함 Model 2가 CPU 50–110Hz, cerebrum GPU 포화 중 | ViT-S(22.7M) 59.5/50.4/28Hz(p50/p99/최악); d384-L6(12.0M) 110.5/100.9/69.7Hz; d192@128(3.1M) 499/428/205Hz; M1 동시 4.5Hz 유지 |
| Model 2를 GPU에 같이 올리면 상호 파괴 | 둘 다 MPS: M1 3.4Hz, M2 15.5/11.5Hz; M2→CPU 시 M2 3.8배 + M1 24% 동시 개선 |
| CPU는 GPU 경합에 면역 — **기존 214/289ms는 MPS 한정** | CPU 경합 페널티 3–9% vs MPS 2.0–5.9배(ViT-S 18.35→61.36ms) |
| batch-1에서 CPU(fp32)가 모든 크기에서 MPS(fp16)를 이김 | ViT-B: CPU 44.67 vs MPS 64.23ms; 스레드 수 영향 미미 |
| 원인: dispatch-bound | MPS 피크 3.78 TFLOPS, batch-1 ViT-B 달성률 15%; dispatch 왕복 179µs(MPS) vs 1.0µs(CPU) |
| CPU 한계선: ≥50Hz p99 기준 인코더 ~22–25M @ ~200 tokens | ViT-S@224 55.7Hz p99(경계); ViT-B는 @224px 18.8Hz p99로 탈락 |
| 인코더 공유 절감 6–10%뿐 — 컴퓨트 목적 채택 금지 | 14.6ms/cycle; Model 2 vision이 Model 1의 4–7Hz에 종속되는 대가 |
| 공유의 진짜 가치: cached token reflex 경로 >1kHz | M2 head 0.197ms p50 → 5076/4149/1965Hz, GPU 풀로드 중 CPU 1스레드 |
| bimanual 7-DoF 고전 IK는 µs 스케일(부하 중 측정) | 단일팔 10.71µs p50; stacked 12×14 15.96µs; coupled 18×14 19.21µs; 관측 최악 172µs = 5800Hz |
| F/T 입력 비용 0 — 사이징 무영향 | vision이 Model 2 비용의 ~96%, proprio+F/T+fusion+head 전체 ~4% |
| cerebrum도 CPU가 더 빠름 — GPU 가치는 offload 병렬성 | M1: MPS 229.9ms(4.3Hz) vs CPU 138.6ms(7.2Hz); both-CPU: M1 5.2Hz + M2 41.3Hz p50/최악 32.6Hz |
| 열 스로틀링 없음 | ~15분 부하 후 경고 0건, 피크 matmul 3.78 TFLOPS 재현 |

### WHAT CHANGES 핵심
- 두 모델의 MPS 동시 배치 계획 **폐기** (최고 가치 수정).
- Model 2 CPU행의 이유 교체: "작아서"가 아니라 **CPU가 batch-1에서 더 빠르고 경합 면역이라서** — vision encoder 유지한 채 성립.
- **214/289ms 경합 페널티를 계획 입력에서 은퇴** — CPU에선 3–9%, 기존 latency 예산은 자릿수 오류.
- 인코더 공유는 컴퓨트 최적화가 아닌 **cached-token reflex 경로**로 재정의해 채택 결정.
- Model 2 인코더 상한 ~22–25M; d192–d256 6층 선호(120–190Hz p99); ViT-B급 배제.
- GPU는 fast path가 아님 — Model 1을 GPU에 두는 실제 근거는 "CPU를 제어 모델용으로 비우기"이며 이를 기록.
- Model 1 예산 4.2–4.5Hz; 더 빠른 갱신은 trunk 축소가 유일한 레버(230ms 중 163ms = 71%).
- 고전 IK는 컴퓨트 예산에서 제거(free); F/T 입력도 zero-cost 취급.
- 배치 2안 중 의도적 선택: M1-GPU+M2-CPU(중앙값 최대, 최악 28Hz) vs both-CPU(M1 5.2Hz, M2 tail 우수 32.6Hz) — 최악 jitter가 안전 제약이면 both-CPU 유리.

### 실행에 영향 주는 UNCERTAINTIES
- 센서/I/O latency(캡처·전송·전처리·F/T 읽기) 전부 미포함 — 실제 end-to-end는 유의미하게 더 길고, 전처리가 Model 2와 CPU 코어 경합.
- 모델 간 IPC(ROS2/공유메모리) 미측정 — 배포 시스템과의 최대 격차.
- 전 모델 랜덤 가중치 — latency는 유효하나 해당 크기의 **정확도 충분성**은 별개.
- 진짜 부등식 제약 QP 미측정(추정 50–500µs는 inference); ANE(CoreML)·CPU fp16/int8 미검 — placement 판도를 바꿀 수 있음.
- 열 검증 ~15분 — 상용화라면 장시간 soak test 필요; non-realtime OS라 worst-case는 유한 샘플 관측치.
