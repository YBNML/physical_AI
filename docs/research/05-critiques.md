# 05 — 적대적 비판: 원시 objection 덤프 (3개 렌즈)

*(국문 요약본. 영문 원문 전문은 [05-critiques-en.md](05-critiques-en.md). 이 비판들은 F/T 입력 확정 이전에 작성되었고, 일부 FATAL 판정 — 특히 'Model B는 IK' 계열 — 은 이후 07-REVISION-01에서 전제가 철회되었습니다.)*

---

## LENS 1: "IK 재발명" — 저수준 '소뇌'는 고전 기구학의 중복 재구현인가

analytic IK + servo가 이미 sub-millisecond에 보증과 함께 푸는 문제를 Model B가 다시 풀고 있는 것 아닌가 — RGB 입력은 명시된 역할과 실제 역할이 다르다는 신호 아닌가.

| 심각도 | 주장 (한 줄) | 핵심 근거 수치 | 이후 개정에서의 지위 |
|---|---|---|---|
| [FATAL] | Model B의 I/O(목표 EE 포즈+현재 관절→관절 명령)는 문자 그대로 IK+servo = MoveIt Servo. 학습 데이터 0으로 이미 존재 | IKFast closed-form ~5–50µs, TRAC-IK >99.8% sub-ms(KDL ~96%); 학습형 ~15–40ms로 500–8000배 느림; IKFlow mm급·~1.5° 오차; 발표된 계층형 VLA 중 학습 저수준에 Cartesian→관절 변환을 시키는 사례 전무(GR00T N1.6은 절대 관절각 포기, π0.5는 "simple PD controllers") | **철회** — REV.1에서 F/T 입력 확정으로 'Model B는 IK' 전제 철회 |
| [FATAL] | IK는 다중해(6-DoF 최대 8 branch, 7-DoF는 1-D null space manifold) — 학습 회귀기는 branch를 일관되게 못 골라 mode 평균(무효해) 또는 timestep 간 branch flip(33ms에 ~180° 손목 슬루). 데이터가 늘수록 악화 | HDP: 포즈 출력 diffuser의 IK 오류율 24.55%; step-wise 적분 노이즈 증폭 O(k); offline MSE에서 비가시 | **수정** — 'IK 대체' 프레임은 철회, 관절 출력 head의 branch 일관성 문제 자체는 잔존 |
| [MAJOR] | B의 RGB 입력이 최대의 tell — 직무가 기구학이면 이미지 불필요, 이미지가 load-bearing이면 IK가 아님. 정당화 후보 (a)충돌 회피 null space(→depth가 맞음) (b)visual servoing(유일한 정당 용도, 단 목표는 이미지 공간) (c)접촉 phase(위치 전용 하드웨어면 표현 불가) 모두 붕괴 | VLA-Perf: vision+VLM이 latency 지배(π0 on Thor: 6.06+20.30+26.20ms); M4 Pro ~7 vs 4090 165 FP16 TFLOPS; 카메라 30–60Hz가 상한 → '200Hz reflex' 서사 붕괴; OpenHelix: latent이 시각 변화에 "largely insensitive" | **철회** — REV.1에서 'RGB 삭제' 권고 철회; REV.2에서 추론이 AGX Orin으로 이동해 Mac 경합 전제 소멸 |
| [MAJOR] | 학습망이 유일한 명령 경로 — 관절 한계·속도/토크·도달성·특이점 보호 전무. 실제 소뇌는 motoneuron 직결 출력이 없고 ablation은 마비가 아닌 ataxia — ablation이 마비를 낳는 모듈은 corticospinal tract | QP-IK(iKinQP, TSID, pink)는 하드 제약 + 가능 집합 내 재최적화; damped-least-squares는 특이점에서 설계상 graceful degradation; 한계·특이점 근방은 시연 데이터가 구조적으로 과소 커버 | **유지** — 비학습 reflex layer 권고는 전 개정에서 핵심 유지 |
| [MAJOR] | 절대 SE(3) 포즈는 4개 canonical action 추상 중 최하 실측치, 7-DoF에선 미결정 — 둘 다 신경망 아닌 산수로 해결 가능 | EE absolute 69.0 / joint absolute 77.3 / joint delta 88.0 / EE delta 89.6%; chunk-wise가 step-wise 대비 ~10점; ER/ERJ 실기체: joint 96% / ERJ 94% / task-space 8%('take cup out of cabinet' 0%); hand-eye ~2.1mm RMS / 3.2° bias 고착 | **수정** — REV.1: 8%/94%/96%는 오인용(관절 출력이므로 96% 열). delta·6D 회전 권고는 유효 |
| [MAJOR] | 이미 포화 근방인 절반에 용량 배분 — oracle ablation상 오차는 거의 전부 고수준에 있고, 파라미터 0의 고전 저수준으로 이미 충분 | GHOST: 고수준만 교체로 36.7%→90%; RT-Affordance 예측 68% vs oracle 76%; keypose 계열(PerAct, RVT 등)은 학습 저수준 0으로 ~10 demo/task; 분할비 Helix 7B/80M, RoboDual 7B/20M, π0 3B/300M — 50/50 사례 없음 | **유지** |
| [MODERATE] | 시연 유래 정답 포즈로 학습한 B를 A의 오차 분포에 배포 = covariate shift. IK는 구조상 이 문제가 없음(전 작업공간에서 정의상 정확) | GHIL-Glue: handoff 수정만으로 CALVIN +25%, 실기체 54%→70%; OpenHelix: projector pre-alignment 없으면 전 구성 0/0/0/0/0 | **유지** |
| [MODERATE] | B의 직무(IK/모션 플래닝/visual servoing) 미규정 — 셋은 baseline·입출력·학습 신호·벤치마크가 전부 다른 문제이며, 이 모호성이 설계를 방어 가능해 보이게 함 | 플래닝이면 반증 존재: HDP 80.18% vs PerAct+Planner 57.72%; RK-Diffuser 94.55% vs RRT 26.82%(oracle goal) — 단 승인은 kinematics-aware(미분 가능 FK 증류)이지 신경망이어서가 아님 | **수정** — REV.1(F/T)·REV.2(위치 전용 확정)로 직무가 'F/T 기반 학습 admittance' 쪽으로 구체화 |
| [MODERATE] | '용량 분할로 성능 향상'은 문헌이 검증한 바 없음 — 직접 검증한 유일 랩의 결론은 이득이 런타임 분할이 아니라 학습 데이터에 있다는 것 | π0.5 ablation: 'implicit HL'이 2위 — "a significant portion of that benefit is already obtained simply by including subtask prediction data in the training mixture." (이득의 상당 부분은 subtask 예측 데이터를 학습 혼합에 넣는 것만으로 얻어진다); HiRT 실기체 quasi-static 70.0 vs 71.3(분할 열세); OpenHelix 비동기 sweep 94/97/95/95/95/95/95 | **유지** |

### 성립 조건 (WHAT WOULD MAKE IT WORK) — 핵심 압축

- **zero-learning baseline 선구축·실측**: Model A → TRAC-IK(joint-limit 회피 + null space posture bias) → ros2_control 500–1000Hz(토크 지원 시 Cartesian impedance). 이걸 못 이기면 분할은 아무것도 입증 못 함.
- **하드웨어 명령 인터페이스 선확인**: 위치 전용이면 학습 B의 표현력 = IK + 위치 servo. 학습 저수준 논거는 토크/admittance/가변 stiffness 인터페이스(Franka FCI, KUKA FRI 등)를 요구. *(참고: 이후 REV.2에서 위치 전용으로 확정 — 단 손목 F/T 표준 탑재도 확정되어 'F/T 입력 + 위치 출력' admittance 경로가 별도로 열림)*
- **B를 clipped RESIDUAL로**: `q_cmd = IK(pose, seed=q_prev) + clip(f_θ(·), ±δ)` — Kawato feedback-error-learning, residual-RL(Johannink et al. ICRA 2019: 수천 샘플로 접촉 삽입 ~80–100%). day-zero 동작, gain 스윕 ablation, dysmetria형 graceful degradation.
- **pose→joint map은 학습 금지** — 학습은 정확해 중의 이산 선택(branch/swivel angle)만. 7-DoF면 Model A가 포즈+swivel angle을 방출(ERJ)하고 정확히 풀 것.
- **인터페이스 선수정**: chunk-wise SE(3) DELTA + 6D 연속 회전(Euler·raw quaternion 금지) + 그리퍼 + duration/속도 + swivel 스칼라 + stiffness 채널(최소 3단 compliance 플래그). EE delta 89.6 vs absolute 69.0; ERJ는 숫자 1개로 8%→94%.
- **두 번째 vision encoder 제거**: A의 캐시 특징 공유(corticopontine 패턴 — 소뇌는 피질 추상화 입력을 받지 raw retina를 받지 않음) 또는 proprioception + wrench + 저해상 wrist crop만.
- **비학습 reflex layer를 두 모델 아래에**: 관절/속도/토크 clamp, 작업공간 경계, A 포즈에 대한 도달성+충돌 필터, 힘 임계 후퇴 — 이것이 '계층'을 실제로 정당화.
- **B는 A의 실제 출력 분포로 학습**: DAgger 라운드 + 목표 포즈 노이즈 증강 + 첫날부터 stale 목표 학습(HiRT 방식).
- **저렴한 판별 실험 2개 먼저**: (1) π0.5식 'implicit HL'(단일 모델 + 보조 목표 포즈 head; OpenHelix 보조 head는 단일 최대 이득 3.45→4.01); (2) 단일 모델 + action chunking/Real-Time Chunking(SmolVLA: 동일 성공률, ~30% 단축, 2배 처리량, ~50줄). 하나라도 격차를 닫으면 분할 불필요.
- **파라미터·데이터 우위는 Model A로**: RT-Affordance의 leverage는 저렴한 비-텔레옵 감독 — ~750장(수집 1h/라벨 2h) 없으면 77%→24%, web co-training 없으면 →11%. 평가는 이점이 실제로 나타나는 동적 장면(HiRT 75% vs 48%)에서, 충분한 검정력으로.
- **이름을 바꾸거나 forward model 추가**: efference copy + 상태로 다음 관절 상태·시각 특징 예측(실제 소뇌 기능인 ~100–150ms latency 보상; 예측 오차 = 이상 신호 + A 재질의 트리거).

---

## LENS 2: "정보 병목" — 수제 6-DoF EE 포즈 인터페이스는 손실 병목이자 복합 오차원인가

포즈 목표가 표현 못 하는 것, ground-truth 포즈로 학습한 B가 A의 noisy 포즈를 받을 때의 붕괴, seam의 비미분성, 프레임·캘리브레이션 오차 전파, 고정 데이터 예산에서 소형 2개 vs 대형 1개를 공격.

| 심각도 | 주장 (한 줄) | 핵심 근거 수치 | 이후 개정에서의 지위 |
|---|---|---|---|
| [FATAL] | 맨 SE(3) 포즈는 action이 아님 — 그리퍼 명령·duration·task-phase 부재로, 같은 포즈에서 접근/파지/들기/놓기가 겹치는 순간마다 B의 학습 문제가 ill-posed(반쯤 닫힌 그리퍼 mode 평균) | 발표된 포즈 인터페이스는 전부 그리퍼가 first-class: HDP는 (a_trans, a_rot, a_grip); RT-Affordance keyframe은 그리퍼 flip 시점 추출; HAMSTER는 open/close 주석. duration 없는 waypoint는 궤적이 아니고 접근 속도가 접촉 충격력을 결정 | **유지·강화** — REV.1에서 그리퍼 명령 부재가 blocking 결함으로 정식 제기 |
| [FATAL] | hindsight 정답 학습→예측 포즈 배포 covariate shift 캐스케이드 + 양방향 보정 채널(gradient·보상·신호) 전무. 계통 오차면 50회 반복 50회 동일 실패, 아무도 관측 못 함 | GHOST: 36.7–40%→90%(~50점이 handoff에); GHIL-Glue: handoff만 고쳐 CALVIN +25%, 실기체 54%→70%; OpenHelix Table 6: pre-alignment 없으면 0/0/0/0/0 | **유지** |
| [FATAL] | '고정 예산 분할로 성능·일반화 향상'은 문헌이 능동 반박하는 유일한 주장 — 실측 이득은 전부 latency 분리/저렴한 고수준 데이터/인터넷 pretraining으로 분해. 로컬 예산 반분은 최대 실측 효과를 포기 | HiRT 70.0 vs 71.3; orchestration 단기 69.57 vs 69.63%(동률); pretrained VLM: LIBERO 77.5→97.8%(~20점, 어느 계층 프리미엄보다 큼); 실전 분할은 극단 편중 + frozen pretrained 상단(87:1, 350:1, ~10:1); RT-Affordance 77→24→11% — 분할의 가치는 데이터 접근 | **유지** |
| [MAJOR] | '절대 SE(3) waypoint'는 최악 실측 action 추상 + hand-eye bias(계통적·비관측·B가 보정 불가)를 최대 고착 | 69.0 vs 89.6%; chunk-wise +~10점(O(1) vs O(k)); TRI LBM은 relative SE(3)+6D 회전, GR00T N1.6은 절대 관절각→상태 상대 chunk; hand-eye ~2.1mm RMS/~3.2° + 자세 의존 처짐 — delta는 양 끝점이 bias 공유로 대부분 상쇄 | **유지** — 최고 가치·최저 비용 단일 수정으로 재확인 |
| [MAJOR] | 프레임 미규정('head/카메라 프레임 또는 base_link') — 움직이는 head에서 카메라 프레임 포즈는 캡처 순간의 head extrinsic과 함께만 유의미, 추론 latency가 head 운동을 그대로 목표 오차로 변환(간헐 실패 = 최악의 디버깅 시그니처) | OC-VLA: +8.0/+13.8/+10.0, novel viewpoint 열화 절반(14% vs 29%) *(참고: REV.1 확인 결과 이 14%/29%는 논문에 없는 수치)*; head 20°/s × 300ms = 0.6m에서 ~63mm 오차; 스탬프 없는 1°도 ~10mm | **수정** — 수치 일부 오인용. 프레임 계약 요구는 유지, REV.2에서 head→arm_base가 목 관절 2개 의존 + 목 틸트 40.6° 한계로 토르소 상시 이동 확정 — 우려가 오히려 구체화 |
| [MAJOR] | 7-DoF에서 6-DoF 포즈는 미결정 명령(B가 elbow를 발명, 그 선택은 A의 장면 지식과 무관) + pose→joints는 이미 정확·고속으로 풀린 IK | joint 96 / ERJ 94 / task-space 8%(0% on 'take cup out of cabinet'); IKFast 5–50µs; TRAC-IK >99.8%; HDP 24.55% 실현 불가 포즈 | **수정/부분 철회** — REV.1: IK 전제 철회 + 8/94/96 오인용(96% 열); REV.2: G1 팔은 canonical S-R-S 7-DoF, closed-form IK 존재 확정 |
| [MAJOR] | 힘·stiffness·compliance 채널 부재 → 접촉 밀집 과제 구조적 불가 — 팔이 위치 전용이면 B도 compliance 표현 불가, 양 수준 모두 수정 불가 | VICES: 과제별 최적 제어 공간 상이; peg-in-hole: "tolerance and the robot's precision are at the same level" (공차와 정밀도가 같은 수준)이면 오차가 jamming으로 증폭; ForceVLA +23.2%(삽입 80%); FAVLA 80.8%, +38.0점; 6-DoF 포즈는 1987년 이래 (pose, stiffness, damping) 튜플보다 엄격히 덜 표현적 | **수정** — REV.2에서 위치 전용 + 손목 F/T 표준 동시 확정: '양 수준 수정 불가' 전제가 'F/T 입력 + 위치 출력 admittance 가능'으로 변경 |
| [MAJOR] | 자체 RGB 스트림 = 지배 비용 2회 지불 + 단일 Apple silicon에서 직렬화 → 총 latency = L(A)+L(B), 동급 monolith보다 엄격 열세 — 제어율 정당화 무효 | VLA-Perf(π0/4090: vision 4.02+VLM 19.79 vs action 7.25ms); MLX 단일 Metal dispatch queue(#3078, 미수정 close); SmolVLA: 단일 모델 비동기 chunk로 동일 성공률·~30% 단축·2배 처리량; OpenVLA-OFT: 26배 가속 후 LIBERO 97.1% | **철회/무효화** — REV.1 RGB 삭제 철회; REV.2 추론은 AGX Orin(Mac 전제 소멸) |
| [MODERATE] | 점 추정 전달 — 고수준의 다봉성(handle vs rim)과 불확실성 폐기. 회귀 head는 bimodal 목표의 평균(공중 포즈)을 출력, B는 '확신'과 '50/50'을 구분 불가. offline MSE에서 비가시 | GHOST: per-patch GMM → dense heatmap(단일 픽셀 마스크보다 우수); RT-Trajectory: 2.5D 38% vs 2D 0%; RT-H offline MSE ~20% 개선이 온라인 성공률로 비례 전이 안 됨 | **유지** |
| [MODERATE] | 단일 EE 포즈는 양팔 협조 제약(핵심은 두 손의 상대 변환)·전신 자세·재시도 의도 표현 불가 — 종료/전환 정책 부재로 첫 실패 파지에서 livelock(장면 불변→같은 포즈→같은 명령 무한 반복) | 독립 5mm 오차 2개 → 최대 10mm 상대 오차(squeeze 또는 drop); Helix S1은 손목 포즈+손가락+토르소+head를 200Hz 출력; orchestration: 전환/종료가 결정 변수(최적 계층 67.1% vs naive 40.6%, ablation 시 ~95%→0 근방) | **유지** — 본 설계가 양손인 만큼 직접 유관 |

### 성립 조건 (WHAT WOULD MAKE IT WORK) — 핵심 압축

- **절대 → chunk-wise SE(3) DELTA 궤적**(8–16 waypoint, chunk 시작 기준, 6D 연속 회전): 단일 수정으로 ~20점(69.0→89.6) + ~2mm/~3° bias 대부분 상쇄.
- **인터페이스를 최소 실행 단위로**: waypoint당 delta pose + 연속 그리퍼 aperture + dt/속도; 전역 compliance 채널·redundancy 스칼라·phase/attempt 필드. HDP의 (a_trans, a_rot, a_grip)은 바닥이지 천장이 아님.
- **학습 latent 사이드채널 병행** — RoboDual 패턴(이산 action + latent, 실기체 +26.7%, 시연 5%로도 강함): 포즈의 해석성·단위 테스트 가능성 유지 + phase·의도·'부드럽게'의 대역폭 복원.
- **점이 아닌 분포를 시각적으로 전달**: GMM을 B가 소비하는 RGB 위에 heatmap/오버레이로 렌더링(GHOST·RT-Affordance·RT-Trajectory 전부 렌더링; 2.5D 38 vs 2D 0%). mode 가중치/엔트로피로 하류 gating.
- **프레임 계약을 코드로**: A는 카메라 프레임 예측 → 캡처 시점 timestamp의 head 상태로 즉시 base_link 변환 → B는 base_link만. staleness assert. 카메라 프레임 목표가 추론 지연을 살아남게 하지 말 것.
- **B를 A의 실제 출력 분포로 학습**: 실측 오차 공분산 노이즈 증강 → DAgger 라운드 → feasibility/progress 필터(GHIL-Glue +25%, 54→70%).
- **독립 학습 후 결합 금지** — 3개 recipe: (1) 연속 latent로 end-to-end(Helix); (2) 고수준 freeze + projector pre-align(OpenHelix — 없으면 전 구성 0%); (3) VLM freeze + adapter/action만 학습(GR00T N1.5: 43.3→83.0%, 최대 보고 이득; 로컬에서 가장 현실적).
- **상향 폐루프**: B→A 스칼라(tracking 오차/도달성/신뢰도) + event-trigger 재계획(그리퍼 상태 변화·오차 임계·subtask 성공·인터럽트). OpenHelix상 호출 빈도는 무관 — 노력은 trigger에. 사이에 작업공간+도달성 하드 게이트.
- **v1에서 Model B 삭제 후 필요성 입증**: A → TRAC-IK → 팔 컨트롤러 500–1000Hz. 학습 B는 additive clipped residual로만 재도입. IK를 하고 있다면 µs로 풀린 문제의 24.55% 오류율 근사. *(참고: 이후 REV.2에서 실제로 'v1에서 학습된 Model 2 제외' 권고로 수렴)*
- **분할은 편중 + pretrained 상단**(20:1 이상): pretrained VLM ~20점(77.5→97.8)이 최대 lever. 50/50 from-scratch 로컬 분할은 지지 증거 0. 진짜 이득은 고수준의 저렴한 비-텔레옵 데이터(RT-Affordance ~750장; HAMSTER 고수준 1.2M off-domain + 저수준 320 에피소드).
- **제어율은 싼 방법 먼저**: chunking + real-time chunking + 비동기(chunk 5→250은 ~11% 비용, denoising 10→50은 5배, autoregressive 토큰 head는 102배 페널티 — 긴 chunk + 적은 flow-matching step).
- **분할 전 3개 control + 검정력**: (a) 동급 monolith, (b) A + 고전 IK/impedance, (c) 보조 목표 포즈 head 단일 모델(π0.5 'implicit HL'). TRI 기준 ≥50 rollout/조건 — 10-트라이얼 A/B는 노이즈 측정. 평가 regime도 사전 명시: 이점은 동적·장기/추론 밀집(+41.8/+30.0점)에서만, 단기 quasi-static은 동률(69.57 vs 69.63).

---

## LENS 3: "시스템·안전" — 체인된 두 신경망의 latency·jitter·실패 격리·귀속

end-to-end latency 예산·jitter, 비RT OS(macOS) 스케줄링, 가속기 경합, 실패 격리(안전 envelope·staleness·NaN·desync), 실패 귀속·디버깅. latency 수치는 실제 대상 머신(Mac mini M4 base — 10코어 CPU 4P/6E, 10코어 GPU, 32GB unified, kern.sched=edge; 사전 연구가 가정한 M4 Pro가 아니라 GPU 절반·120 vs 273GB/s)에서 실측한 pre-build 설계 비판. *(참고: 이 렌즈의 Mac 추론 전제는 이후 REV.2 — 추론은 로봇 탑재 AGX Orin, GalbotSDK는 Linux 전용으로 Mac은 로봇과 통신 불가 — 및 REV.3(학습·시뮬은 RTX 3090/Ubuntu)으로 대부분 소멸. 단 안전 envelope·인터페이스 계약·귀속 하니스 요구는 플랫폼 무관하게 유효.)*

| 심각도 | 주장 (한 줄) | 핵심 근거 수치 | 이후 개정에서의 지위 |
|---|---|---|---|
| [FATAL] | 이 머신에서 분할은 rate를 분리하지 않고 역전시킴 — RGB 소비 B가 A보다 느려짐. vision encoder 비용을 compute-bound에 최악인 가속기에서 2회 지불 | 실측(M4, MPS, fp16): ViT-B/16급 62.75ms p50 solo(15.9Hz) → 805M cerebrum(470ms solo) 동시 실행 시 214.29ms p50/288.91ms p99 = 4.7Hz; cerebrum도 467→500ms p50, 628ms p99; 총 GPU 처리량 보존(두 모델 비용 = latency의 합); 30fps 카메라 33ms 상한 → 'RGB 위 200Hz 소뇌'는 존재 불가; MPS는 kernel-dispatch 단위 interleave로 경합 6.3x p50/9.2x p99(사전 연구의 MLX 예측보다는 완화) | **철회** — REV.1: 4.7Hz 캡은 MPS placement 아티팩트(올바른 배치에서 CPU 59.5Hz p50); REV.2: 추론은 AGX Orin |
| [FATAL] | 안전 envelope 전무 — 관절 clamp·속도/가속 한계·힘/토크 임계·작업공간 경계·도달성 검사·watchdog·e-stop 없음, B가 유일한 명령 생성기라 모든 네트워크 실패가 전권 팔 운동으로 표출. degrade할 baseline 컨트롤러 자체가 없음 | 고수준은 무방비 신뢰 수준이 아님: RT-Affordance 77→24→11%; GHOST 배포 36.7% vs oracle 90% — 출력 절반가량이 틀리는 것이 정상 regime; task-space 8% vs joint 96% redundancy 결과 인용 *(참고: 8%/94%/96%는 REV.1에서 오인용 판정 — 관절 출력 설계이므로 96% 열)* | **유지** — reflex layer는 이후 전 개정에서 최우선 요구 |
| [MAJOR] | 인터페이스가 timestamp·head/base 스탬프·validity horizon·sequence 번호·지연 시 동작 없이 'A의 출력'으로만 명세 — 실측 500ms p50/628ms p99 A 주기에서 staleness가 지배 오차항 | 카메라 프레임 포즈는 캡처 순간 head 상태 없이 무의미(목표가 무음으로 미끄러짐); p50↔p99 128ms swing = 학습으로 보상 불가한 jitter; HiRT는 cached stale latent로 학습; GHIL-Glue +25% / 54→70% | **유지** — REV.2에서 목 관절 2개 의존 + 토르소 상시 이동 확정으로 캡처 시점 스탬프 요구가 더 절실해짐 |
| [MAJOR] | 이 박스의 macOS는 제어 호스트가 될 수 없음 — '고속 소뇌' 절반이 실제로 돌아갈 곳이 없음 | kern.sched=edge, RT 스케줄링 클래스·PREEMPT_RT·isolcpus 없음; XNU에 SocketCAN·EtherCAT master 없음; ROS2 macOS는 REP-2000 Tier 3; 실측 p99/p50 비 1.28(solo)→1.89(경합), I/O 부하 없이도; 접촉 문헌상 setpoint <~20Hz면 힘 진동 — 설계는 4.7Hz p50 | **무효화(결론은 채택)** — REV.2: GalbotSDK Linux 전용(Mac은 로봇과 통신 불가), 추론은 AGX Orin — '제어를 Mac에서 빼라'가 사실상 실현 |
| [MAJOR] | Model B 명세 = 학습된 IK+servo — 낭비를 넘어 능동적으로 위험: 고전 solver의 infeasibility 신호를 파괴 | solver는 typed failure 반환("no solution", "at joint limit", "near singularity") → 재계획/후퇴/중단/A 재질의 분기 가능; 회귀망은 항상 그럴듯한 숫자(도달 범위 밖·특이점에서 신호 0); TRAC-IK ~0.8ms에 failure 반환 vs B는 완전 신전 + 손목 한계 + 전류 상승, 로그에 오류 0건 | **철회** — REV.1에서 IK 전제 철회. typed failure 논거는 reflex layer의 도달성 필터로 흡수·존속 |
| [MAJOR] | 실패 귀속 메커니즘 부재 — A/B/인터페이스/staleness/캘리브레이션 5개 가설이 구분 불가(각각 ~1주 비용, 상호작용), 추적 효과 크기는 현실적 트라이얼 수의 노이즈 바닥 이하 | 유일한 국소화법은 frozen-one-side oracle swap(GHOST 40→90%); HiRT 정적 1.3점(monolith 우세); orchestration 69.57 vs 69.63 동률; TRI 1,800 트라이얼 + Bayesian + Bonferroni로 겨우 분리 | **유지** |
| [MODERATE] | 데이터 비용은 2배가 아니라 순차·재진입적 — B 데이터셋은 A가 있어야 수집 가능, A 재학습(데이터 추가·프레임 변경·재캘리브레이션)마다 무효화. 공유 CI 부재로 회귀가 무음 | 강제 스케줄 train-A → rollout → relabel → train-B; handoff shift 실측 16–25점(GHIL-Glue), oracle 격차 50점(GHOST); A-v3/B-v2 비호환이 어디에도 기록되지 않고 로그상 양쪽 다 개별 정상으로 보임 | **유지** — 단 REV.2의 'v1에서 학습된 Model 2 제외' 권고로 v1에서는 유예 |
| [MODERATE] | 유효성·유한성·liveness 계약 부재 — fp16+Metal에서 deadline miss가 실측된 norm인데 '늦은 메시지'의 동작이 미정의 | fp16 overflow는 attention/normalization에서 일상적; A의 NaN 포즈가 B의 LayerNorm을 거치면 유한한 쓰레기 값(NaN보다 나쁨 — 소박한 검사를 통과); 실측 p99 628ms/289ms; 20분 무인 구동 중 overflow 1회 → 수 라디안 점프 → 최대 속도 슬루, 로그엔 ~2,000 프레임 중 이상 1개·예외 0건 | **유지** — ~50줄 가드(assert+limiter+deadman)는 플랫폼 무관 |

### 성립 조건 (WHAT WOULD MAKE IT WORK) — 핵심 압축

- **servo 루프를 Mac 밖으로**: ≥500Hz 루프는 팔 컨트롤러나 소형 Linux 호스트에서, Mac은 유선 Ethernet으로 20–50Hz setpoint만(~0–4ms, 사실상 공짜). 이 하나로 모든 macOS 실시간 objection 해소. *(참고: REV.2에서 GalbotSDK Linux 전용 + 추론 AGX Orin 확정으로 사실상 그대로 실현)*
- **비학습 reflex layer를 두 모델보다 먼저**: 위치/속도/가속/jerk clamp, 토크·전류 임계 + 자동 후퇴, 작업공간 box, 자기충돌·도달성 검사, finite/in-range assert, step-magnitude limiter, deadman watchdog(hold-last-good 후 속도 0 ramp). 수백 줄, 학습 데이터 0.
- **B = TRAC-IK + impedance/PD 위의 CLAMPED RESIDUAL**: residual→0이면 고전 스택 완전 복구, fault는 폭주 아닌 dysmetria, ablation은 스칼라 gain 1개, solver의 typed infeasibility 신호 상속.
- **B에서 RGB 제거**: ViT-B/16급만으로 경합 시 4.7Hz 상한 *(참고: REV.1에서 MPS placement 아티팩트로 판정 — 올바른 배치에서 CPU 59.5Hz p50; RGB 삭제 권고 자체도 철회)*; proprioception 스케일 net은 실측 2.18ms p50/2.80ms p99(460Hz). vision이 정말 필요하면(= A의 bias를 보정하는 visual servo) 명시 선언 + 저해상 wrist crop + 고전 uncalibrated IBVS 대비 벤치마크 + 15–30Hz 상한 수용.
- **인터페이스를 typed·timestamped·versioned 메시지로**: SE(3) 목표(6D 회전) + 그리퍼 + duration/속도 + 캡처 timestamp + 캡처 시점 head/base 관절 상태 + frame id + sequence 번호 + validity horizon. base_link 변환은 A 출력 직후. stale 정책 명시: horizon 초과 시 hold + 속도 감쇠, 외삽 금지. 절대 포즈보다 chunk-wise DELTA(89.6 vs 69.0%).
- **DAgger relabeling + 버전 고정 + 오프라인 회귀 테스트**: (A-version, B-version, calibration-version)을 단일 배포 아티팩트로; 로그된 인터페이스 trace를 B에 재생해 명령 경계·smoothness assert — A 재학습이 B를 무음으로 깨지 못하게.
- **귀속 하니스를 모델보다 먼저**: Oracle-A(정답 포즈→B, B 단독 상한), Oracle-B(B를 TRAC-IK+impedance로 교체, A 라이브 포즈, A 단독 상한), 전체 timestamp 로깅 + 오프라인 재생. 포즈 인터페이스의 진짜 강점(RViz 인스펙션·6-DoF 마우스 보정·단위 테스트)은 부수 효과가 아닌 산출물로 예산 배정.
- **저렴한 control 실험 먼저**: (a) monolith + chunking + real-time chunking 실측 — rate 격차가 닫히면 분할 불필요; (b) 보조 목표 포즈 head 단일 모델(π0.5 'implicit HL'). 평가는 프리미엄이 존재하는 장기·추론 밀집·동적 과제(+30~+42점)에서 ≥50 rollouts/조건으로 — 아니면 '검정력 미달로 결론 불가'를 명시.
- **Mac에서 동시 실행해야 한다면 유일한 진짜 병렬 구성**: A는 MLX/Metal GPU, B는 Core ML을 cpuAndNeuralEngine 고정(GPU 명시 배제) — ANE 상주를 powermetrics로 실증(무음 GPU fallback이 6.3x/9.2x 열화를 재현; dynamic shape·custom attention·데이터 의존 sampler가 통상적 변환 실패 지점).
- **정당화를 정직하게 재서술**: 방어 가능한 분할 근거는 (1) 저수준이 소비 못 하는 저렴한 비-텔레옵 데이터, (2) 사람이 인스펙션·보정 가능한 인터페이스, (3) 진짜로 분리되는 클록. '파라미터 예산 분할로 성공률 향상'은 어떤 발표 시스템도 지지하지 않으며, 이 하드웨어에서는 제어율까지 대가로 치름.
