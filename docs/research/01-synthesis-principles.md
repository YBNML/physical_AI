# 2모델 "대뇌/소뇌" 조작 아키텍처 — 의사결정용 기술 리뷰

> ## ⚠️ 이 문서를 읽기 전에
>
> 이 문서는 **1라운드 종합**이며, **손목 F/T 입력과 Galbot G1 확정 이전**에 작성되었습니다. 여러 결론 — 특히 **C7 "Model B는 학습된 IK"**, **RGB 관련 결론(O3)**, **4.7Hz 캡**, **8%/94% 인용**, **헤드 프레임 금지(O7)** — 이 이후 [07-REVISION-01](07-REVISION-01-with-ft.md)과 [10-REVISION-02](10-REVISION-02-galbot-g1.md)에서 **철회·수정되었습니다.**
>
> 역사적 기록이자 방법론 참고로 보존합니다. **최신 판정은 [메인 문서](../2026-07-28-dual-brain-architecture-analysis.md)와 07/10을 보십시오.**
>
> *(원문 영어 · 국문 번역본. 인용문은 원문을 병기했습니다.)*

**범위:** Apple Silicon Mac mini에서 돌아가는 physical-AI 조작 로봇을 위해 제안된 로컬 2모델 설계 — Model A (RGB + 현재 EE 포즈 → 목표 EE 포즈)와 Model B (목표 포즈 + 관절값 + RGB → 관절 명령) — 에 대한 평가.

**문서 전체에 쓰인 근거 강도 규약:**
- **[M]** = 이 리뷰 중 사용자의 실제 기기에서 측정 (Apple M4 base, 10 CPU / 10 GPU 코어, 32GB 통합, PyTorch 2.11/MPS, batch 1)
- **[P]** = peer-reviewed 또는 확립된 출판 결과
- **[X]** = 2025–2026 arXiv preprint, 검색은 했으나 독립 검증·재현하지 않음. 방향성 참고용
- **[V]** = 벤더/기업 블로그, ablation 미공개. 아키텍처 서술은 신뢰 가능, 성능 주장은 아님
- **[I]** = 리뷰어의 추론. 측정도 인용도 아님

---

## 1. 판정

**옳은 것:** 분해하려는 직관은 건전하고, 분리 지점의 선택(Cartesian, "손이 어디로 가는가"와 "팔이 어떻게 거기 가는가" 사이)은 로보틱스 40년과 모든 keyframe 기반 조작 시스템이 수렴한 바로 그 지점이며, 포즈 형태의 중간 표현은 계층형 VLA 문헌 전체에서 가장 강한 근거를 갖습니다 (RT-Affordance: 예측 포즈 68% vs 동일 backbone·데이터의 평면 RT-2 28% [X]; HDP 80.18% vs 평면 Diffusion Policy/ACT 15–18% [P]). 저수준에 자체 지각을 주는 것도 옳고, 그것이 진짜 dual-system을 rate만 두 개인 단일 모델과 구분하는 정의적 기준입니다.

**틀린 것:** 세 가지, 심각도 순으로.

**(a)** 명시된 *목표* — 고정 파라미터 예산을 두 모델에 나눠 작업 성공률을 높인다 — 는 문헌이 어디에서도 지지하지 않는 유일한 주장입니다. 통제 실험을 돌린 출판 연구가 없고, 가장 잘 통제된 대리 실험에서는 실기체 준정적 태스크에서 monolith가 근소하게 *앞섭니다* (HiRT 71.3 vs 70.0 [X]). 단순 계층은 단기 조작에서 평면 VLA와 정확히 동률이고 (69.63% vs 69.57% [X]), 런타임 계층 vs 계층 형태의 학습 데이터를 직접 ablate한 유일한 랩은 이득 대부분이 데이터에 있다는 것을 발견했습니다 (π0.5 "implicit HL" [X]).

**(b)** 명세된 Model B — 목표 포즈 + 관절 → 관절 명령 — 는 IK + servo loop이고, 이 기기에서 **5.31µs [M]** 에 정확히 풀리는 것을 학습 네트워크의 현실적 25–60Hz로 대체하는 것입니다. 그리고 IK는 회귀 네트워크가 구조적으로 만들 수 없는 *타입 있는 infeasibility 신호*("no solution", "at joint limit", "near singularity")를 반환합니다. redundant 팔에서는 추가로 underdetermined 명령이고, 8% → 94% 성공률을 회복시키는 수정은 신경망이 아니라 스칼라 하나입니다 **[X]**.

> *(REV.1 정정: (b)는 **철회**되었습니다. 손목 F/T가 입력에 있으면 `(포즈, 관절)`이 전단사 map이라는 전제가 깨집니다 — 동일한 `(포즈, q)`가 자유공간·가벼운 접촉·끼인 접촉 전부일 수 있습니다. "학습이 더 나쁘다"는 주장도 근거 없음으로 철회됐습니다. 다만 residual 처방 자체는 다른 이유로 살아남습니다: 최적화 퇴화. 07 §1 참조. 그리고 여기 인용된 8%/94% 수치는 **오인용**입니다 — 해당 표는 최종 행동 기준 분류이고 Model B는 관절을 출력하므로 96% 열에 속합니다.)*

**(c)** 이 특정 하드웨어에서 분리는 rate 스토리를 분리하기는커녕 *역전*시킵니다: 하나의 Metal GPU 위의 두 모델은 latency의 **합**을 치릅니다 — Model B가 RGB를 소비하는 데 필요한 최소인 ViT-B/16 인코더가 단독 62.75ms p50이고 동시 실행되는 대뇌급 모델과의 경합 하에서 **214ms p50 / 289ms p99 [M]**, 즉 "빠른" 루프가 4.7Hz에 안착합니다 — "느린" 루프가 되려던 것보다 느립니다.

> *(REV.1 정정: (c)도 **철회**되었습니다. 214/289ms는 MPS 고유 현상이지 아키텍처 속성이 아닙니다. 올바른 placement(Model 1은 GPU, Model 2는 CPU)에서 Model 2는 자체 ViT-S/16@224를 갖고도 **59.5Hz p50 / 50.4Hz p99**로 돌면서 Model 1이 동시에 4.2Hz를 유지합니다 **[M]**. placement 버그를 아키텍처 버그로 진단한 것입니다.)*

**가장 큰 단일 수정:** *Model B를 명령 경로에서 해석적 IK + impedance control 위의 유계 residual로 강등하고, 분리를 capacity 배분이 아니라 latency·값싼 상위 데이터·검사 가능성으로 재정당화하라.* 이 목록의 나머지 전부는 그 한 수의 세부 조정입니다.

---

## 2. 주장 채점표

| # | 명시된 주장 | 판정 | 근거 |
|---|---|---|---|
| C1 | 고정 파라미터 예산을 두 특화 모델에 나누면 작업 성공률이 오른다 | **반박됨** | 통제 실험(파라미터·데이터·init·rate 고정)을 돌린 논문이 없습니다. HiRT 실기체 준정적: 계층 70.0 vs monolith 71.3 **[X]**. 단순 계층 vs 평면 VLA 단기: 69.57% vs 69.63% — 정확한 동률 **[X]**. MoE scaling law(capacity 분리를 엄밀히 연구한 유일한 분야): routing 이득이 *모델 크기에 따라 감소*하는 멱법칙을 따름 (Clark et al. 2022 **[P]**); 활성 파라미터 고정 시 expert를 늘리면 memorization은 개선되나 reasoning은 *포화*하고, 약간 더 넓은 dense 모델이 푸는데 expert를 아무리 늘려도 못 푸는 문제가 존재 (Mixture of Parrots, NeurIPS 2024 **[P]**) |
| C2 | 분리는 일반화를 개선한다 | **부분 지지 — 단 메커니즘은 capacity가 아님** | 실제 이득 존재: RT-H 새 장면에서 +8–12% **[X]**; 계층이 장기에서 +41.8pt, reasoning 중심에서 +30.0 **[X]**. 그러나 귀속되는 메커니즘은 항상 (a) 공유 가능한 재사용 중간 primitive, (b) 상위가 소비할 수 있는 off-domain 데이터, (c) 인터넷 pretraining입니다. HAMSTER는 상위를 off-domain 샘플 120만 개로, 하위를 in-domain 에피소드 320개로 학습합니다 **[X]** |
| C3 | 분리는 제어 rate를 개선한다 | **이 하드웨어에서 반박됨; 전용 가속기에서만 지지** | Helix는 S1과 S2를 **전용 GPU**에서 돌려 7–9Hz / 200Hz를 얻습니다 **[V]**; Gemini Robotics는 backbone을 **클라우드**에 둡니다 **[X]**. 하나의 Metal GPU에서: ViT-B/16 62.75 → 214ms p50(6.3배), 289ms p99(9.2배) 경합 하 **[M]**; 805M 대뇌급 모델은 467 → 500ms p50 / 628ms p99 **[M]**. GPU 총 처리량은 보존됩니다. 반사실: SmolVLA가 **단일** 모델의 async chunking으로 동일 성공률, 완료 ~30% 단축(9.7s vs 13.75s), 처리량 2배(19 vs 9 사이클) **[X]**; OpenVLA-OFT는 7B monolith의 행동 생성을 26배 빠르게, latency를 3배 낮추고 LIBERO 97.1%를 기록 **[X]** |
| C4 | 포즈는 올바른 *종류*의 중간 표현이다 | **지지** | RT-Affordance 예측 68% / oracle 76% vs RT-2 28%, 그리고 *더 큰* backbone의 goal-image 조건부 24% **[X]**; 비파지 작업 70% vs 3% **[X]**. HDP 80.18% vs 57.72%(PerAct+플래너), 15–18%(평면 DP/ACT) **[P]**. GHOST fold-onesie 80% vs 평면 DP 10% **[X]**. RT-Trajectory 미지 41작업에서 63% vs 29% **[X]**. **제안에서 가장 강하게 지지받는 요소입니다** |
| C5 | *절대 SE(3) waypoint*가 올바른 포즈 인코딩이다 | **반박됨** | 유일하게 깨끗한 정면 비교(실기체 단일팔, ACT 회귀, 평균 진행도): EE 절대 **69.0%**, 관절 절대 77.3%, 관절 delta 88.0%, EE delta **89.6%** **[X]**. chunk 단위가 스텝 단위보다 ~10pt 우위, O(1) vs O(k) 노이즈 증폭 논거 **[X]**. TRI의 LBM은 6D rotation의 상대 SE(3)로 수렴 **[X]**; GR00T N1.6은 절대 관절각을 명시적으로 버리고 state-relative chunk로 **[X]**. 절대는 또한 ~2.1mm RMS / 3.2° hand-eye 캘리브레이션 bias를 영구히 각인시키는데, delta는 이를 상쇄합니다 |
| C6 | *맨 6/7-vector*가 충분한 인터페이스 payload다 | **반박됨** | 성공한 누구도 raw vector를 넘기지 않았습니다. GHOST는 **포즈에 대한 GMM**을 각 카메라 평면에 dense distance-field heatmap으로 투영하고 heatmap이 single-pixel mask를 이기는 것을 발견 **[X]**; RT-Affordance는 그리퍼 윤곽을 이미지에 **렌더링**하고 캘리브레이션이 없을 때만 텍스트 토큰을 넘길 것이라고 명시 **[X]**; RoboDual은 latent **와** 이산화 행동을 함께 넘김 **[X]**. HDP의 공개 인터페이스는 (a_trans, a_rot, **a_grip**) — 그리퍼 비트는 선택이 아님 **[P]**. 제안에서 완전히 빠진 것: 그리퍼 aperture, duration/속도, compliance, 중복성 스칼라, 불확실성 |
| C7 | Model B(학습된 포즈 → 관절 명령)는 잘 동기부여된 모듈이다 | **반박됨** | 이건 IK입니다. 이 기기 실측: DLS 6×7 solve = **5.31µs (188kHz)** **[M]**. IKFast closed-form 5–50µs; TRAC-IK 서브ms에 >99.8% 해결률 vs 순정 KDL ~96% **[P]**. 학습된 IK는 *덜* 정확: IKFlow mm 수준 병진, 최대 ~1.5° 회전 **[P]**. 어떤 출판된 계층형 VLA도 학습된 저수준에게 이것을 시키지 않습니다: RT-1/RT-2/RT-A는 Cartesian delta 출력, HiRT는 EE delta, TRI의 LBM은 상대 SE(3), GR00T N1.6은 관절각을 버림. 추가로, q_current로 시드하고 argmin‖q − q_prev‖를 취하는 해석적 IK는 **hard branch-continuity 보장**을 줍니다. bimodal(팔꿈치 위/아래) 타깃에 대한 L2 학습 회귀기는 두 유효 branch의 산술 평균으로 수렴하고, 그 FK는 명령된 포즈 근처에도 없습니다 **[I]** |
| | | | ⚠️ **REV.1에서 철회.** 손목 F/T 입력이 확정되면서 입력 집합 퇴화 전제가 무너졌습니다. 07 §1 참조 |
| C8 | Model B에 RGB가 필요하다 | **부분 지지 — 셋 중 하나만 정당** | 정당: Model A의 체계적 캘리브레이션 bias를 상쇄하는 폐루프 visual servoing — 단 그러면 (이미 bias를 담은 metric 포즈가 아니라) **이미지 공간** 목표를 받아야 하고, 카메라 rate로 돌아야 하며, 학습 없이 캘리브레이션-프리 servoing을 이미 푸는 online Jacobian 추정 기반 고전 uncalibrated IBVS와 벤치마크해야 합니다 **[P]**. 부당: collision-aware 중복성(RGB가 아니라 depth/geometry가 필요; cuRobo/QP-IK가 완전성 보장과 함께 수행)이나 접촉 적응(팔이 관절 위치만 받으면 무효). 비용이 심각: 연산 제약 플랫폼에 비전 인코더 2개, 그리고 OpenHelix probing은 실제 dual system의 latent가 "largely insensitive to changes in visual information"임을 발견 **[X]** |
| | | | ⚠️ **REV.1에서 결론 철회.** 두 모델이 같은 프레임을 소비하므로 인코더 공유가 자명하고(절감은 6%뿐), 이미지는 팔당 중복 DoF 해소와 팔꿈치 충돌 인지에 필요합니다 |
| C9 | 저수준은 자체 지각을 가져야 한다 | **지지** (범주로서) | OpenHelix가 정확히 이것을 dual system의 정의로 쓰고, 이 기준으로 π0과 GR00T N1을 범주에서 *배제*합니다 **[X]**. 제안은 종류 면에서 옳습니다. C8의 반론은 원칙이 아니라 비용과 인코더 크기에 관한 것입니다 |
| C10 | 두 모델을 따로 학습해 합성할 수 있다 | **명세된 대로는 반박됨; 수정하면 지지** | OpenHelix Table 6: **projector 사전 정렬 없이는 모든 구성이 0/0/0/0/0**, 있으면 첫 작업 성공률 94–96% **[X]**. GHIL-Glue: 핸드오프만 고쳐서 **CALVIN +25%, 실기체 54% → 70% [X]**. GHOST가 저수준을 freeze한 채 oracle 교체: **36.7% → 90% [X]**. Helix는 latent를 통해 S1→S2 역전파 **[V]**; GR00T N1은 "tightly coupled and jointly optimized" **[X]**; FiS-VLA의 기여 전체가 두 모델 분리를 *되돌리는* 것 — 분리가 "limits System 1 from fully leveraging the rich pretrained knowledge"하기 때문 **[X]** |
| C11 | Mac mini에서 두 모델을 로컬로 돌리는 것이 가능하다 | **부분 지지** | 메모리는 괜찮음(32GB 중 ~2.5–7GB) **[M/I]**. 연산이 구속 조건: **3.78 TFLOPS fp16 지속, 104 GB/s 달성** **[M]** — 연산은 RTX 4090의 약 1/43, 대역폭은 1/10. 600 토큰에 대한 3B trunk prefill이 **3.74초 (0.27Hz) [M]** — π0.5/GR00T급은 여기 배포 불가. MLX는 프로세스당 Metal dispatch queue가 1개이고 별도 스레드의 동시 모델은 크래시 (ml-explore/mlx #3078, 미수정 종료) **[X]**. macOS: PREEMPT_RT 없음, isolcpus 없음, `kern.sched=edge`가 스레드를 P/E 코어 사이로 이주 **[M]**, SocketCAN 없음, EtherCAT master 없음, ROS 2는 REP-2000 **Tier 3**(소스 전용, CI 없음) **[P]** |
| C12 | 대뇌/소뇌 명명이 적절하다 | **반박됨** | §4 참조. 실제 소뇌를 절제하면 ataxia/dysmetria, Model B를 절제하면 마비. 소뇌는 운동뉴런으로 직접 투사가 *없습니다* **[P]** |
| C13 | Model A(RGB + 상태 → 목표 포즈)는 잘 동기부여된 모듈이다 | **지지** | 이것이 RT-Affordance의 affordance generator, GHOST의 π_hi, PerAct/RVT keypose 계열의 핵심입니다. 그리고 모든 oracle ablation이 남은 오차가 사는 곳이라고 말하는 절반입니다 (GHOST 배포 36.7% vs oracle 90% **[X]**; RT-Affordance 예측 68% vs oracle 76% **[X]**) |
| C14 | 두 절반을 로컬에서 처음부터 학습해도 된다 | **반박됨** | 랜덤 init vs 사전학습 VLM: **LIBERO 77.5% → 97.8%, ~20pt [X]**; 연구 전반 15–25pt가 전형. 이는 *문헌의 모든 단기 계층 프리미엄(0–9pt)보다 크고*, monolith도 쓸 수 있는 레버입니다. Helix의 헤드라인은 인터넷 pretraining 덕에 텔레옵 ~500시간, "<5% of prior VLA datasets"로 작동했다는 것 **[V]** |

---

## 3. 선행 연구 지도

### 3.1 비교표

| 시스템 | 상위 | 하위 | 인터페이스 | Rate | 학습 결합 | 근거 품질 |
|---|---|---|---|---|---|---|
| **제안** | RGB + EE 포즈 → 목표 SE(3) 포즈 | 목표 포즈 + 관절 + RGB → 관절 명령 | **절대 SE(3) 포즈, raw vector** | 미명시 | **분리** | — |
| **RT-1** [P] | 없음 (monolithic) | — | — | 3Hz, <100ms 예산 | — | 데모 13만, 시행 3천 |
| **RT-2** [P] | 없음 (monolithic) | — | — | 1–3Hz(55B), ~5Hz(5B) | — | 실기체 평가 6천 |
| **RT-H** [X] | π_h → language motion | π_l → 7-DoF 행동 | **자연어 구절**, 인코더로 | ~RT-2 rate; async 1스텝 선행 | **같은 모델, 프롬프트 2개** | 오프라인 MSE 30.2→24.9(oracle 17.9); 새 물체 55→65; 개입 40→63 |
| **RT-Trajectory** [X] | 사람 / VLM / hindsight | RT-1급 정책 | **이미지에 그린 2D/2.5D 스케치** | ~3Hz | 분리 (HL이 네트워크가 아님) | 미지 41작업에서 63% vs 29%; 2.5D 38% vs 2D 0% |
| **RT-Affordance** [X] ← *제안에 가장 가까움* | PaLM-E 1B, 이미지+언어 → **keyframe의 EE 포즈** | PaLM-E 1B, Cartesian EE 행동 | **SE(3) 포즈, 컬러 그리퍼 윤곽으로 이미지에 렌더링** | 미보고; 재계획 "fixed or adaptive" | 분리 예측기, 공유 backbone 계열 | 예측 68% / oracle 76% vs RT-2 28%; 예측기 이미지 750장 없으면 77%→24%, 웹 없으면 →11% |
| **HAMSTER** [X] | off-domain 120만으로 finetune한 VLM | 3D-aware 정책, in-domain 320 에피소드 | **coarse 2D pixel path** + 그리퍼 플래그 | HL 에피소드당 ~1회 | 분리 | OpenVLA 2배 이상; 저자들이 path가 "cannot communicate force or rotation"임을 인정 |
| **HDP** [P] ← *아키텍처적으로 가장 가까움* | PerAct → next-best EE 포즈 | RK-Diffuser → **관절** 궤적, FK distillation | **(a_trans, a_rot, a_grip)** | keyframe 트리거 | 미분 가능 FK를 통한 distillation | 80.18% vs 플래너 57.72%, 평면 15–18%; **pose-diffuser IK 오류율 24.55%** |
| **GHOST** [X] | DINOv3 + transformer → **3D EE 포즈에 대한 GMM** | Diffusion Policy, goal-conditioned | **3D 포즈를 각 카메라에 dense heatmap으로 투영** | 15Hz, chunk 16 | 분리 | 평면 DP 10% 대비 80%; **oracle HL 90% vs 배포 36.7%** |
| **PerAct / RVT / RVT-2** [P] | keypose 예측 | **고전 motion planner, 학습 파라미터 0** | SE(3) keypose + 그리퍼 | HL sub-Hz, LL servo rate | 해당 없음 | RVT-2 작업당 ~10 데모로 RLBench ~81.4% |
| **HiRT** [X] ← *가장 잘 통제된 검증* | InstructBLIP-7B + LoRA | 35M(시뮬) / 150M(실기체), EE delta | **단일 latent, MAP-pooled 최종 레이어**; FiLM + cross-attn + prefix | monolith 4.1Hz 대비 9.8Hz; VLM async, 캐시 | joint | **정적: 분리 70.0 vs monolith 71.3. 동적: 75 vs 48** |
| **RoboDual** [X] | OpenVLA-7B, LoRA | ~20M DiT + 16M sensory 인코더 | **latent 그리고 이산화 행동**, cross-attn | generalist 3.9Hz, specialist 28.6Hz, 시스템 15Hz | joint | OpenVLA 대비 실기체 +26.7%; 데모 5%에서도 강함 |
| **OpenHelix** [X] ← *유일한 체계적 연구* | LLaVA 변형 | 3D Diffuser Actor | 전부 조사: `<ACT>` 토큰, 최종 레이어, 중간 레이어, MaxPool, action+lang latent | 질의당 1→60 스텝 sweep | 양쪽 다 연구 | **HL 빈도 sweep이 평평(94/97/95/95/95/95/95); 사전 정렬 없으면 0/0/0/0/0; 사전학습 LL 3.53 vs scratch 2.85; aux pos/rot head 3.45→4.01** |
| **π0** [X] | — (transformer 1개) | 300M action expert (별도 *가중치*, 같은 forward pass) | **하나의 transformer 내부 self-attention — 직렬 인터페이스 없음** | chunking으로 50Hz, ~10 flow 스텝 | 그래프 1개 | 분리-vs-monolith ablation 없음 |
| **π0.5** [X] ← *가장 타격이 큰 ablation* | 같은 가중치, 1차 질의 | 같은 가중치, flow expert | **자연어 subtask 문자열**; proprio를 텍스트 토큰으로 | LL 50Hz; HL "lower frequency" | 모델 1개 | **"implicit HL"(런타임 HL 없음, subtask 데이터는 학습에 유지)이 2위; 전체 π0.5가 사람 HL oracle을 이김; GPT-4를 HL로 쓰면 더 나쁨** |
| **GR00T N1 / N1.5 / N1.6** [X] | Eagle-2 VLM 1.34B | DiT ~0.86B | **레이어 12 은닉 상태, cross-attention 경유** | 10Hz / 120Hz 주장; L40에서 16-action chunk당 63.9ms 측정 | "tightly coupled, jointly optimized" | **분리-vs-monolith ablation 미공개.** N1→N1.5 이득(실기체 43.3%→83.0%)은 *VLM freeze*, adapter/LayerNorm 수정, FLARE에 귀속 — 분리가 아님. N1.6은 *저수준* DiT를 2배로 |
| **Helix (Figure)** [V] | 7B VLM, 카메라 6개 | **80M** 시각운동 정책 | **단일 연속 latent**, S1 토큰 공간으로 투영 | **7–9Hz / 200Hz, 전용 GPU** | **latent를 통해 S1→S2 gradient 역전파** | **정량 결과 전혀 미공개** |
| **Gemini Robotics 1.5** [X] | GR-ER 1.5 orchestrator | GR 1.5 multi-embodiment VLA | **개방 어휘 자연어** (VLA가 "도구") | 제어용 rate 미제시; ER은 성공 감지에 5Hz | 분리 모델, agentic | Fig. 4가 이득을 Motion Transfer + multi-embodiment 데이터에 귀속, **계층이 아님**. 개발 평가의 >90%가 시뮬 |
| **FiS-VLA** [X] | VLM | **같은 VLM의 마지막 2 블록** | **파라미터 공유 — 직렬 인터페이스 없음** | 117.7Hz, 최적 async 비율 1:4 | 공동학습; "slow loss" 제거 시 69%→62% | 동기: 분리 모델이 S1의 VLM pretraining 활용을 막는다 |
| **SmolVLA** [X] ← *핵심 반사실* | 없음 (450M monolith) | — | action queue만 | 환경 사이클 30fps; async | — | **LIBERO 87.3% vs OpenVLA-7B 76.5%; async = 동일 성공률, ~30% 빠름, 처리량 2배** |
| **OpenVLA-OFT** [X] | 없음 (7B monolith) | — | — | OpenVLA 대비 **생성 26배 빠름, latency 3배 낮음** | — | **LIBERO 97.1%, π0·Octo·DP·MDT를 이김** |

### 3.2 실제 시스템들의 인터페이스가 제안과 다른 지점

제안은 **독립 학습된 두 네트워크 사이에 raw 절대 SE(3) vector를 넘기고, 저수준은 관절 공간 학습 모델**입니다. 인터페이스로 문헌을 분류하면:

- **학습된 연속 latent** — Helix, HiRT, GR00T N1(레이어 12), RoboDual, LCB(`<ACT>` 토큰), DP-VLA. *모든 대규모 프로덕션 dual system이 쓰는 것입니다.* 전부 경계를 넘는 gradient로 joint 학습됩니다.
- **자연어** — RT-H("move arm forward"), π0.5 subtask 문자열, Gemini Robotics 1.5 orchestrator→VLA. 가장 싸고 디버깅 쉬움. *장기 분해*에 쓰이고 폐루프 운동에는 절대 안 쓰임.
- **이미지 공간 spatial goal** — RT-Trajectory(2.5D 스케치), HAMSTER(2D path), GHOST(포즈→heatmap), RT-Affordance(포즈→렌더링 윤곽). *포즈를 쓰는 모든 시스템이 숫자를 넘기는 대신 이미지에 렌더링합니다.*
- **파라미터 공유 / 직렬 인터페이스 없음** — π0, π0.5, FiS-VLA.
- **하이브리드** — RoboDual은 latent *와* coarse 이산화 행동을 함께 넘기고, ablation에서 둘이 서로 다른 정보를 나른다고 말합니다.

**중요도 순 세 가지 구체적 이탈:**

1. **아무도 맨 숫자 포즈 vector를 넘기지 않습니다.** 인터페이스가 *포즈인* 두 시스템(RT-Affordance, GHOST)은 둘 다 그것을 이미지 공간에 재표현해서 저수준의 시각 인코더가 관측과 같은 좌표계에서 공간적으로 attend할 수 있게 합니다. GHOST는 추가로 점추정이 아니라 *분포*(GMM)를 넘깁니다. RT-Affordance는 토큰화된 텍스트 값을 넘기는 것이 "when camera calibration is unavailable"의 fallback이라고 명시합니다.
2. **Cartesian 목표에서 관절 명령을 내는 학습된 저수준을 가진 시스템이 없습니다.** RT-Affordance의 LL은 Cartesian, HiRT는 EE delta 출력, TRI의 LBM은 6D rotation의 상대 SE(3), GR00T N1.6은 절대 관절각에서 명시적으로 *멀어졌습니다*. HDP는 관절을 출력하지만 — 미분 가능 순기구학을 통해 pose diffuser를 distillation한 것뿐이고, 그 ablation은 포즈 전용 변형의 **IK 오류율이 24.55%**임을 보입니다. 이것이 학습된 pose→joint map의 정직한 헤드룸 추정치입니다. π0.5는 관절 목표를 명령하지만 "simple PD controllers... without any additional trajectory planning or collision detection"이 추종합니다.
3. **아무도 두 절반을 독립 학습해 붙이지 않습니다.** 작동하는 레시피는 셋: latent를 통한 end-to-end 역전파(Helix), HL freeze + projector 사전 정렬 + prompt-tune(OpenHelix), 또는 VLM 전체 freeze 후 adapter + action model만 학습(GR00T N1.5). 정렬을 건너뛴 측정값은 **0/0/0/0/0**.

**보정을 위한 capacity 비율:** Helix 7B/80M (**87:1**), RoboDual 7B/학습 가능 20M (**350:1**), π0 3B/300M (**10:1**), GR00T N1 1.34B/0.86B (~1.6:1, 그리고 그래프 1개). **아무도 50/50으로 나누지 않습니다.** NVIDIA가 N1.6에서 성능을 더 원했을 때 한 일은 *저수준 DiT를 2배로 키우고* *상위를 더 freeze*한 것이었습니다.

---

## 4. 신경과학 비유: 도움이 되는 지점과 오도하는 지점

### 4.1 진짜로 도움이 되는 지점

- **다중 rate 계층은 실재하고, 비유의 최고 기여입니다.** 생물은 둘이 아니라 세 계층으로 돌아갑니다: 단연접 신전반사 ~20–40ms; 경피질 long-latency 반사 ~50–100ms; 시각 유도 수의적 보정 ~150–250ms **[P, 자릿수 수준]**. 이것*이* 계층에 대한 올바른 논거이고, 다음에 대응됩니다: 500–2000Hz 고정 servo/impedance, 20–50Hz 학습 반응 정책, 0.5–10Hz 의미 정책.
- **Model A ≈ parietal AIP → premotor F5는 공정한 대응입니다.** 비전에서 파지 관련 물체 affordance를 추출하고 손 형상을 선택하는 것이 정확히 그 회로가 하는 일입니다 **[P]**.
- **Capacity 배분 직관이 유용하게 뒤집힙니다.** 인간 소뇌 ≈690억 뉴런 vs 대뇌피질 ≈160억 (Herculano-Houzel **[P]**) — "저수준"에 4배 많은 유닛이지만, 학습된 readout 하나를 가진 거대한 *넓고 얕고 희소한* 확장층으로 조직돼 추론당 비용이 쌉니다. 진지하게 받아들이면 이는 **넓고 얕고 싼** 저수준을 논증하는데, 이것이 고rate 로컬 정책에 정확히 맞는 형태이고 — 두 번째 이미지 소비 transformer의 정반대입니다. *(유보: 뉴런 수는 파라미터 수의 약한 대리이고, granule cell은 거의 랜덤 확장 feature입니다. 휴리스틱으로 취급하십시오 **[I]**.)*
- **저수준은 raw가 아니라 추상화된 지각을 받아야 합니다.** 소뇌 입력은 corticopontine → 교뇌핵 → 이끼섬유로 도착하며 이미 피질이 처리한 것입니다. **망막→소뇌 투사는 없습니다.** *(약간의 단순화: 시각 정보는 교뇌 중계로 소뇌에 도달하고 retinal slip은 VOR/OKR을 위해 하올리브에 도달합니다. 정확한 진술은 소뇌의 시각 입력이 저차원의 운동/오차 관련이지 물체 인식 수준이 아니라는 것입니다 **[P]**.)* 이는 인코더를 복제하는 대신 **Model A의 시각 feature를 공유**할 것을 논증합니다.

### 4.2 적극적으로 오도하는 지점

- **Model B는 소뇌가 아니라 피질척수로 + 척수입니다.** 소뇌는 **운동뉴런으로 직접 출력이 없습니다.** 유일한 출력은 심부소뇌핵 → (a) 시상 → 피질, (b) 적핵/망상체 → 척수 개재뉴런입니다 **[P, 교과서]**. 다른 곳에서 생성된 명령을 편향시킬 뿐입니다. Model B는 명령을 *생성*합니다.
- **Ablation 검사가 판정합니다.** 소뇌 제거 → ataxia, dysmetria, 의도진전(~3–5Hz, 예측 요소가 제거된 지연 피드백 루프의 서명), 근긴장저하. **운동은 남고 근력은 보존됩니다** **[P]**. Model B 제거 → 로봇은 정지합니다. **절제가 부조화가 아니라 마비를 낳는 모듈은 명령 경로입니다.**
- **가장 견고하게 귀속되는 소뇌 기능이 통째로 빠져 있습니다: 예측.** 소뇌는 **forward model**입니다 — 원심성 복사 + 상태 → 예측된 미래 상태 / 예측된 감각 결과로 ~100–200ms 감각 지연을 보상. 외측 소뇌에 TMS를 가하면 reach가 **~138ms 낡은** 손 위치로부터 계획됩니다 (Miall et al., PLoS Biol 2007 **[P]**); 건강한 피험자는 ~60ms *예측 선행*, 소뇌 실조 환자는 ~172ms *지연* **[P]**. Model B에는 원심성 복사도, 예측도, 예측 오차 신호도 없습니다. **소뇌를 소뇌로 만드는 가장 핵심적인 것이 빠진 것입니다.**
- **가장 컨트롤러다운 이론조차 그것을 배타적이 아니라 가산적으로 만듭니다.** Kawato의 feedback-error-learning: 소뇌 역모델이 기존의 조잡한 피드백 컨트롤러 *위에* 놓이고, 출력이 피드백 컨트롤러의 것과 **합산**되며, 피드백 컨트롤러 자신의 출력이 등상섬유 교사 신호로 쓰입니다 **[P]**. **baseline은 절대 사라지지 않습니다.**
- **위상(topology)이 틀렸습니다.** 소뇌-대뇌 회로는 **닫힌 루프**입니다: 특정 피질 영역에서 입력을 받는 소뇌 영역이 바로 그 영역으로 되돌아 투사하며, M1·전운동·안구운동·전전두·두정에 대해 반복됩니다 (Kelly & Strick, J. Neurosci. 2003 **[P]**). Kelly & Strick은 이것을 "the traditional view of the cerebellum as merely executing commands from higher brain centers"와 대비되는 것으로 명시적으로 규정하는데 — 그것이 정확히 제안의 엄격한 A→B 위상입니다. 또한 두 개의 박스가 직렬인 것이 아니라 *다수의 병렬 루프*입니다.
- **"상위"조차 피드백 컨트롤러입니다.** 경피질 long-latency 반사 루프가 M1을 통과합니다 (Scott, Nat. Rev. Neurosci. 2004 **[P]**, 정설이라기보다 영향력 있는 이론). 포즈를 내고 침묵하는 상위는 피질이 하는 일이 아닙니다.
- **세 번째 기능 블록이 통째로 빠졌습니다: 기저핵.** 탈억제에 의한 행동 선택, 그리고 *언제 전환하거나 중단할지* **[P]**. 로봇에서 이것은 subtask/종료/전환 정책인데 — 제안이 전혀 명시하지 않았고, 체계적 오케스트레이션 연구가 결정적 설계 변수 중 하나로 지목한 것입니다(장기에서 단순 계층 40.6% vs 최선 67.1% **[X]**).

### 4.3 신경과학에 대한 정직한 유보

소뇌의 forward model vs inverse model 질문은 현 문헌에서 **미해결**로 명시돼 있고, Ivry의 *timing hypothesis*가 살아 있는 경쟁 프레임워크입니다 **[P, 논쟁 중]**. Marr–Albus의 LTD-as-memory도 더 이상 정설이 아닙니다 — 평행섬유 LTD가 손상된 쥐도 운동 과제를 학습하고 (Schonewille et al., Neuron 2011 **[P]**), 분야는 "여러 부위에서의 가소성"으로 이동했습니다. **이 중 무엇도 명명을 구제하지 않습니다:** 모든 경쟁 이론이 소뇌를 예측과/또는 타이밍에 관한 것으로 만들고, 어느 것도 소뇌를 이미지로부터 관절 명령을 내는 유일한 생성자로 만들지 않습니다.

**실무 권고:** Model B의 이름을 바꾸거나("저수준 정책" / "모터 컨트롤러"), 그 이름을 정의하는 기능을 추가하십시오 — 자신의 원심성 복사로부터 다음 관절 상태와 다음 스텝 시각 feature를 예측하는 **forward-model head**. 그 head는 같은 데모에서 나오는 공짜 supervision이고, 표현을 정규화하며, 예측을 앞으로 굴려 추론 latency를 보상하게 해주고(실제 소뇌 기능), 그 예측 오차가 Model A 재질의를 위한 자연스러운 이상 트리거가 됩니다.

---

## 5. 주요 반론, 중복 제거 후 순위

### O1 — FATAL: Model B는 IK를 더 나쁘게 학습한 것

> ⚠️ **REV.1에서 전제 철회.** 손목 F/T가 입력에 있으면 입력 집합이 퇴화적이지 않습니다. 다만 residual 처방은 *최적화* 퇴화(다음 관절각 MSE의 ~99%가 IK로 설명됨)라는 다른 이유로 살아남습니다. 07 §1 참조.

**반론.** "목표 EE 포즈 + 현재 관절 → 관절 명령"은 IK 시그니처 + servo입니다. 이 기기 실측: DLS 6×7 solve = **5.31µs (188kHz) [M]**. IKFast closed-form 5–50µs; TRAC-IK 서브ms에 >99.8% vs KDL ~96% **[P]**. 학습된 Model B는 3자릿수 느리고, 해석적이 서브마이크론인 곳에서 mm~cm 정확도입니다. 더 나쁘게, IK는 분기 가능한 **타입 있는 실패**("no solution", "at joint limit", "near singularity")를 반환하는데, 회귀기는 런타임 신호 없이 항상 자신감 있고 그럴듯하며 틀린 숫자를 반환합니다. 가중: IK는 다중해이고(6-DoF에서 최대 8 branch, 7-DoF에서 1차원 null-space manifold), bimodal 타깃에 대한 L2 학습 회귀기는 **두 유효 branch의 산술 평균**으로 수렴하는데 그 순기구학은 목표 근처에도 없습니다. 확률적 head라면 대신 연속 타임스텝에서 branch를 뒤집습니다. q_current로 시드하고 argmin‖q − q_prev‖를 취하는 해석적 IK는 어떤 모방 목적함수도 제공하지 않는 **hard continuity 보장**을 줍니다. **이 실패는 오프라인 MSE에 보이지 않고 데이터가 많아질수록 *악화*됩니다.**

**구체적 실패.** "맨 왼쪽 머그를 집어 맨 오른쪽에 놓아라." 작업공간 중앙선을 넘으면 형상 전환이 강제되고, 데모에는 거의 동일한 포즈에서 두 branch가 모두 들어 있습니다. Model B는 그리퍼가 옆으로 돌아간 채 15cm 벗어난 중간 형상을 내거나, 33ms 한 스텝에 손목을 ~180° 회전시킵니다.

**완화.** pose→joint map을 학습하지 마십시오. `Model A → TRAC-IK(관절 한계 + singularity 회피, null-space 자세 편향) → 500–1000Hz impedance/PD`를 먼저 만드십시오 — 주말 하나의 통합 작업, 학습 데이터 0, 그리고 OpenVLA/π0/RT-2 배포가 실제로 쓰는 구성입니다. 학습된 Model B는 **clipped 가산 residual**로만 재도입하십시오: `q_cmd = IK(target, seed=q_prev) + clip(f_θ(·))`. 학습이 기구학을 건드려야 한다면 *정확한 해들 중에서 선택하는 것*(어느 branch / 어떤 swivel angle)으로 제한하십시오 — 기구학은 정확하게 남고 네트워크는 선호만 표현합니다.

> ⚠️ *REV.1: "clipped"는 오히려 해롭습니다 — wrench의 가치 전체가 자유공간 대비 큰 접촉 중 보정인데, 자유공간 residual에 맞춘 clip이 그 신호를 억압합니다. contact-phase gating을 쓰십시오.*

---

### O2 — FATAL: 안전 envelope가 없고 Model B가 유일한 명령 생성자

**반론.** 관절/속도/가속/저크 clamp 없음, 토크 임계값 없음, 작업공간 경계 없음, 도달가능성 검사 없음, NaN assert 없음, watchdog 없음, 그리고 결정적으로 — **강등해서 물러날 baseline 컨트롤러가 없습니다.** 출판된 oracle ablation에서 Model A의 실패율은 꼬리가 아니라 *정상 운전 영역*입니다: RT-Affordance의 예측기가 값싼 이미지 750장 없이 24%, 웹 공동학습 없이 11%로 떨어짐 **[X]**; GHOST는 배포된 상위를 90% oracle 대비 36.7%로 측정 **[X]**. **상위 출력의 1/3~1/2이 틀린 것이 baseline입니다.**

**구체적 실패.** 새로운 장면에서 Model A가 테이블보다 40cm 아래로 목표를 회귀합니다. Model B가 외삽하고 팔이 명령 속도로 표면에 처박힙니다. 아무것도 중단시킬 수 없습니다. 7-DoF의 2차 버전: 6-DoF 포즈가 팔을 결정하지 않으므로 팔꿈치가 null space를 통해 작업공간 옆의 무엇으로든 휘둘립니다.

**완화.** **두 모델 아래에, 가중치가 아니라 코드로, 어느 모델이 존재하기도 전에 고정된 학습되지 않은 반사층을 만드십시오**: 관절 위치/속도/가속/저크 clamp; 토크 또는 모터 전류 임계값 + 자동 후퇴; 작업공간 AABB; self-collision + **Model B가 보기도 전에 Model A의 포즈에 적용되는 도달가능성 필터**(A를 재트리거하는 타입 있는 거부를 반환); 세 지점에서 finite-and-in-range assert + 거부-후-유지; 스텝 크기 제한기; last-good을 유지하다 속도를 0으로 ramp하는 deadman watchdog. 그다음 Model B를 residual로 재구조화해 residual→0이면 작동하는 로봇이 복원되게 하십시오 — 생물학적 표현형(마비가 아니라 dysmetria)과 일치합니다.

---

### O3 — 이 하드웨어에서 FATAL: 분리가 rate 스토리를 역전시킨다

> ⚠️ **REV.1에서 철회.** 214/289ms는 두 모델을 MPS에 공동 스케줄링한 아티팩트입니다. 올바른 placement에서 Model 2는 CPU에서 59.5Hz p50입니다 **[M]**. 아래 완화책 (a)"RGB 삭제"도 함께 철회됐고, (b)"CPU로"는 오히려 핵심 처방으로 승격됐습니다.

**반론.** 실측 **[M]**: ViT-B/16(Model B가 RGB에 필요한 최소) = 단독 62.75ms p50 → 동시 실행되는 805M 대뇌급 모델과의 경합 하에서 **214.29ms p50 / 288.91ms p99**, 즉 **4.7Hz**. 대뇌 자신도 467 → 500ms p50 / 628ms p99로 열화됩니다. GPU 총 처리량은 보존되고, 두 모델은 **합**을 치릅니다. Helix는 **전용 GPU [V]** 로 25배 분리를 얻고, Gemini Robotics는 **클라우드 [X]** 를 씁니다. 추가로: 카메라를 먹는 Model B는 가속기 속도와 무관하게 카메라 프레임 rate(30–60Hz)에 하드 캡되므로 "RGB 위의 200Hz 반사층"은 존재할 수 없습니다. 그리고 MLX는 프로세스당 Metal dispatch queue가 1개이고 동시 스레드 크래시가 있습니다(#3078, 미수정 종료) **[X]**.

**구체적 실패.** A를 3Hz, B를 100Hz로 예산했는데 A가 ~2Hz, B가 4.7Hz p50 / 3.5Hz p99로 측정됩니다. staleness 포함 photon-to-joint-command 지연 ≈ 700–850ms. 0.2m/s로 미끄러지는 물체는 보정이 도착하기 전에 15cm 가 있습니다.

**완화, 순서대로.** (a) **Model B에서 RGB 삭제** — 인터페이스 + 관절 + 속도 + wrench를 주십시오. proprio 규모 네트워크는 **2.18ms p50 / 2.80ms p99 [M]**, 스레드 2개 CPU의 3.2M proprio 네트워크는 **GPU가 포화된 상태에서 1.17ms p50 / 1.78ms p99 (858Hz) [M]**. (b) **Model B를 MPS가 아니라 CPU에** — 같은 모델이 MPS에서 23.6–43ms p50에 106–129ms p99로 측정됐습니다. 즉 4배 느리고 꼬리가 16배 나쁘며, *게다가* A의 처리량을 훔칩니다 **[M]**. 이 한 줄 변경(`device='cpu'`, `set_num_threads(2)`)이 분리를 허구에서 실제 25–140배 rate 비율로 바꿉니다. (c) **뭘 만들기 전에 rate를 싼 방법으로 사십시오**: 단일 모델의 action chunking + Real-Time Chunking + async 실행이 ~50줄로 동일 성공률, 완료 ~30% 단축, 처리량 2배를 줬습니다 **[X]**. chunk 길이는 거의 공짜(5→250 행동 = latency +11%)인데 denoising 스텝은 비싸고(10→50 = 5배), autoregressive 토큰 action head는 ~102배 페널티 **[X]**. (d) ≥500Hz servo 루프를 macOS에서 절대 돌리지 마십시오 — 팔 컨트롤러나 유선 이더넷의 $60–150 Linux SBC로 밀어내십시오(~0–4ms, 사실상 무료 **[X]**). 이 한 수가 macOS 실시간성 반론 전부를 무효화합니다.

---

### O4 — MAJOR: 인터페이스가 행동이 아니다

**반론.** 맨 SE(3) 포즈는 그리퍼 명령도, duration/속도도, compliance도, 중복성 스칼라도, 불확실성도 나르지 않습니다. 귀결: (i) 그리퍼 열고 접근과 그리퍼 닫기가 거의 동일한 포즈에서 거의 동일한 손목 이미지로 일어나므로 Model B의 학습 문제가 *형식적으로 ill-posed*입니다 — 반쯤 닫힌 그리퍼를 내거나, RGB에서 task phase를 조용히 재추론합니다(더 작은 모델로 Model A의 일을 다시 하는 것). (ii) 시간 없는 waypoint는 궤적이 아니고, 접근 속도*가* 곧 충격력입니다. (iii) 7-DoF에서 포즈는 팔을 결정하지 않습니다 — 좁은 공간 작업에서 실측 96%(관절) / 94%(ERJ = 포즈 + 중복성 스칼라 1개) / **8%(순수 작업공간)**, "찬장에서 컵 꺼내기"는 0% **[X]**. (iv) compliance 채널이 없으면 접촉 작업이 없습니다 — force-aware VLA 실측 **+23.2%**(ForceVLA over π0) ~ **+38.0pt**(FAVLA over vision-only π0) **[X]**. (v) 점추정은 bimodal affordance(프라이팬: 손잡이 또는 테두리)를 공중의 포즈로 mode-average하는데, 오프라인 MSE에는 보이지 않습니다. HAMSTER 저자들도 자기 인터페이스가 "cannot communicate nuances such as force or rotation"임을 인정합니다 **[X]**.

> ⚠️ *(iii)의 8%/94%/96%는 **오인용**입니다 — 최종 행동 기준 분류이고 Model B는 관절을 출력하므로 96% 열입니다. 중복성 문제 자체는 실재하지만 이 수치로 논증하면 안 됩니다. REV.1 §1 3a 참조.)*

**완화.** 최소 실행 단위로 확장하십시오: **8–16 waypoint chunk × (Δp(3), 6D-rotation(6), 그리퍼 aperture(1), dt(1))**, 그리고 헤더에 stiffness(6 또는 3단계 플래그), swivel angle ψ(1), 신뢰도/엔트로피(1), phase(1), 선택적으로 60차원 latent 사이드 채널(RoboDual 패턴). Model B에게는 vector를 concat하는 대신 **Model B가 이미 소비하는 RGB에 heatmap/오버레이로 렌더링해서** 전달하십시오(GHOST, RT-Affordance). 점이 아니라 **mixture**를 내십시오.

---

### O5 — MAJOR: 절대 포즈는 실측상 최악의 추상화이고 캘리브레이션 bias를 각인시킨다

**반론.** EE 절대 **69.0%** vs 관절 절대 77.3% vs 관절 delta 88.0% vs **EE delta 89.6%** **[X]**; chunk 단위가 스텝 단위보다 ~10pt 우위. 절대는 또한 캘리브레이션 체인 전체를 체계적이고 관측 불가능한 offset으로 상속합니다: hand-eye에 대해 ~2.1mm RMS 병진 / ~3.2° 회전, 더하기 로봇 위치 오차, 더하기 자세 의존적 compliant 변형 **[P]**. Model B는 이것을 볼 수 없습니다 — 학습 중 라벨과 관측 양쪽에 같은 bias가 각인됐기 때문입니다. *측정된* EE 포즈에 앵커된 delta는 양 끝점이 그것을 공유하므로 대부분을 상쇄합니다.

**구체적 실패.** 12mm 버튼 누르기: ~4–5mm 자세 의존 offset은 작업공간 가까운 쪽에서는 작동하고 먼 쪽에서는 안정적으로 빗나간다는 뜻입니다. 데이터 문제로 보이지만 프레임 문제입니다.

**완화.** chunk 시작 시점 포즈 기준의 chunk 단위 SE(3) **delta**, **6D continuous rotation**과 함께(Euler 절대 금지, raw quaternion 절대 금지 — wrap-around 불연속은 공짜 회귀 오차원입니다). **제안 전체에서 가치 대비 비용이 가장 좋은 단일 수정입니다.** *유보: 69.0/77.3/88.0/89.6 표는 검증하지 못한 2026 preprint 출처입니다 — 인용 전 재확인하십시오 **[X]**.*

---

### O6 — MAJOR: 미분 불가능한 seam에서의 연쇄 분포 이동

**반론.** hindsight 정답 포즈로 학습하고 Model A의 오차 분포에 배포된 Model B는 교과서적 covariate shift이고, 어느 쪽도 보정할 gradient·보상·신호가 없습니다. 학습 중에는 모든 (포즈, RGB) 쌍이 상호 일관적입니다. 배포 시에는 포즈가 이미지 속 물체에서 10–20mm 벗어나 있고, B는 *비일관 쌍을 본 적이 없으므로* 학습된 prior가 "목표는 옳다, 거기로 servo하라"입니다. 실측 비용: GHIL-Glue가 핸드오프만 고쳐 **CALVIN +25%, 실기체 54% → 70% [X]**; GHOST가 LL을 freeze한 채 oracle 교체로 **36.7% → 90% [X]**; OpenHelix는 projector 사전 정렬 없이 **모든 구성에서 0/0/0/0/0 [X]**. HRL 문헌이 같은 것을 형식적으로 명명합니다(HIRO의 off-policy correction이 정확히 이것을 위해 존재합니다).

**완화.** (a) B 학습 중 A의 **실측** held-out 오차 공분산으로 인터페이스를 노이즈 augmentation(추측한 Gaussian이 아니라). (b) 필수 **DAgger 라운드**: A가 수렴한 뒤 rollout해서 *실패를 포함한* 실제 인터페이스 메시지를 로깅하고, 재라벨하고, 그 분포로 B를 재학습. (c) GHIL-Glue 스타일 progress/feasibility 필터. (d) 첫날부터 **stale** 목표로 B를 학습(HiRT는 최신 latent를 캐시하고 LL을 그것으로 학습); A의 실측 p50→p99 산포가 ~128ms **[M]** 이므로 staleness는 고정이 아니라 *jitter*입니다. (e) 최선: seam을 미분 가능하게 만들기 — 포즈와 나란히 연속 latent를 넘기고 역전파하거나, A를 freeze하고 projector를 사전 정렬.

---

### O7 — MAJOR: 인터페이스에 타임스탬프·프레임 스탬프·staleness 정책이 없다

> ⚠️ **REV.1: "헤드 프레임 금지"는 명시된 그대로는 철회.** OC-VLA는 카메라 프레임이 base 프레임을 이기는 것을 측정합니다(실기체 +10.0pp). 단 정적 카메라 조건부입니다. 그리고 여기 인용된 "14% vs 29%"는 **논문에 존재하지 않는 수치**입니다(실제는 14.0pp vs 21.3/16.7/16.0pp). 올바른 처방은 금지가 아니라 인터페이스에서 base_link로 해석적 변환하는 것입니다.

**반론.** "헤드/카메라 프레임 기준" 포즈는 *이미지 촬영 시각의* 헤드 extrinsic 없이는 무의미합니다. 실측 A 주기 500ms p50 / 628ms p99 **[M]** 에서 20°/s로 패닝하는 헤드는 0.6m에서 ~6° ≈ **63mm**의 측방 목표 오차를 만들고, 목표가 조용히 미끄러지는데 어디에서도 오차가 제기되지 않습니다. 카메라 프레임 행동은 extrinsic이 알려져 있고 정적일 때 측정 가능하게 *이깁니다*(시뮬 연속 +8.0, 시뮬 이산 +13.8, 실기체 고정 카메라 +10.0, 그리고 새 시점에서 대략 절반의 열화) **[X]** — 그러나 그 결과는 캘리브레이션 오차 민감도를 전혀 연구하지 않고 정적 T를 전제합니다.

**구체적 실패.** 파지가 *간헐적으로* 실패합니다 — 헤드가 마침 움직이고 있었을 때만. **가능한 최악의 디버깅 서명이고, 몇 주 동안 지각 문제로 오진될 것입니다.**

**완화.** 인터페이스를 **타입 있고 타임스탬프 있고 버전 있는 메시지**로 만드십시오: capture_stamp, **촬영 시각의** 헤드/베이스 관절 상태, frame_id, seq, validity_horizon, 그리고 불일치 시 B가 거부하는 (A버전, B버전, 캘리브레이션버전) 해시. Model A는 카메라 프레임으로 예측하고(시점 불변성 이득 유지), **자신의 출력 단계에서** 촬영 시각 헤드 상태로 base_link로 변환합니다. Model B는 base_link만 봅니다. 지평을 넘으면: 유지하고 속도를 0으로 감쇠, **절대 외삽 금지.**

---

### O8 — MAJOR: 이미 포화된 절반에 capacity를 쓰고 있다

**반론.** 모든 oracle ablation이 잔여 오차가 상류에 있다고 말합니다: GHOST가 LL을 freeze한 채 상위만 교체해 **36.7% → 90%** **[X]**; RT-Affordance 예측 68% vs oracle 76% **[X]**; PerAct/RVT/RVT-2 계열 전체가 **학습 파라미터 0**의 저수준으로 작업당 ~10 데모에서 정밀 조작을 달성 **[P]**. 한편 작동하는 모든 분리는 격렬하게 비대칭이고(87:1, 350:1, 10:1) 상단이 인터넷 사전학습돼 있으며, 50/50 처음부터 로컬 분리는 추가로 ~20포인트 pretraining 레버를 포기합니다(LIBERO 랜덤 init 77.5% vs 사전학습 97.8% **[X]**) — 문헌의 모든 단기 계층 프리미엄보다 큽니다.

**완화.** 분리를 25:1 이상으로 만들고, 인터넷 사전학습 VLM을 Model A로 freeze하고, 분리의 *실제* 메커니즘인 값싼 비텔레옵 데이터를 주입하십시오: 손 주석 정지영상 ~750–1500장(수집 ~1h, 라벨링 ~2h; RT-Affordance의 예측기가 그것 없이 77% → 24%로 떨어짐) + 웹 공동학습(없으면 →11%) **[X]**.

---

### O9 — MODERATE: 전환·종료·재시도 정책이 없다

**반론.** 포즈는 "여기로 가라"고 말할 수 있지만 "이미 시도했고 실패했다"고 말할 수 없습니다. 파지 실패 후 장면은 시각적으로 변하지 않으므로 A는 같은 포즈를 내고 B는 — 결정론적 함수이므로 — 같은 명령을 냅니다. **Livelock.** π0.5와 Hi Robot 둘 다 명시적 실패 감지와 재계획을 만듭니다. 체계적 오케스트레이션 연구는 전환/종료를 결정적 변수로 발견했고, ablate된 계층이 ~95%에서 거의 0으로 붕괴했습니다 **[X]**. 이것이 §4의 기저핵 모양 구멍입니다.

**완화.** 트리거를 명시적으로 정하십시오(OpenHelix는 *rate*가 거의 무관하다고 말합니다 — 매스텝부터 에피소드당 1회까지 94/97/95/95/95/95/95 **[X]** — 따라서 설계 노력을 빈도가 아니라 트리거에 쓰십시오): chunk 60% 소진; 그리퍼 상태 전이; 추종 오차 > τ; forward-model 예측 오차 포화; IK-infeasible 거부; 조작자 인터럽트. 인터페이스에 `attempt=k` 필드를, 그리고 B에서 A로 향하는 상향 상태 채널을 추가해 두 번째 시도에서 A가 다른 mode를 샘플링할 수 있게 하십시오.

---

### O10 — MODERATE: 실패 귀속 메커니즘이 없고, 효과 크기가 노이즈 바닥 아래다

**반론.** 독립 학습된 두 모델 + 미분 불가 인터페이스 + oracle 하네스 없음은 모든 실패에 구분 불가능한 설명이 넷 있다는 뜻입니다(A의 포즈 bias, B의 실행, staleness, 캘리브레이션 드리프트). 가중: 추적하는 효과가 태스크가 있는 곳에서 미미합니다 — HiRT의 실기체 준정적 분리-vs-monolith 격차는 **1.3포인트, monolith 쪽으로 [X]**; 단기에서 단순 계층 vs 평면 VLA는 정확한 동률 **[X]**. TRI는 정책을 구분하는 것 자체에 **작업·정책·조건당 50 rollout, 총 1,800 시행, blind 무작위 A/B, Bayesian posterior, Bonferroni 보정**이 필요했습니다 **[X]**.

**완화.** 모델 **이전에** 하네스를 만드십시오: **Oracle-A**(hindsight 정답 포즈를 B에 replay — B의 상한)와 **Oracle-B**(B를 IK + impedance로 교체하고 A의 실시간 포즈 공급 — A의 상한). 전체 타임스탬프와 함께 인터페이스를 로깅하고 trace를 오프라인 replay하십시오. **동적**과 **장기** 태스크에서 평가하십시오 — 출판된 계층 우위가 안정적으로 나타나는 유일한 영역입니다. 조건당 ≥50 rollout을 예산하거나, 실험이 결론 낼 수 없다고 앞에서 명시하십시오.

---

### O11 — MODERATE: finiteness/liveness 계약이 없다, 그것도 deadline miss가 정상인 backend에서

**반론.** attention이나 normalization에서의 fp16 overflow는 일상적입니다. A에서 온 NaN 포즈가 B로 전파되고, B의 입력 LayerNorm이 그것을 *유한하지만 쓰레기인* 값으로 바꿉니다 — NaN보다 나쁩니다. 순진한 검사가 걸리지 않기 때문입니다. 실측 p99(대뇌 628ms, 경합 ViT 289ms **[M]**; 12M proprio MLP조차 0.80ms p50에 최대 38.7ms, ~48배 꼬리 **[M]**)는 deadline miss가 정상 운전 거동이라는 뜻이고, 따라서 "메시지가 늦으면 어떻게 되는가"는 발견되는 것이 아니라 *설계되어야* 합니다.

**완화.** ~50줄: A의 출력, B의 출력, servo 입력 세 곳에서 finite-and-in-range assert + 통과가 아니라 거부-후-유지; 스텝 크기 제한기; K회 deadline miss 후 속도를 0으로 ramp하는 deadman timer; 모든 거부를 원인 입력과 함께 로깅.

---

### O12 — MODERATE: 데이터 비용이 2배가 아니라 순차적이고 재진입적이다

**반론.** B의 데이터셋은 A가 존재하기 전엔 수집할 수 없고(O6 참조), `A 학습 → A rollout → 재라벨 → B 학습`이 강제됩니다. A를 재학습할 때마다 — 데이터 추가, 프레임 관례 변경, hand-eye 재캘리브레이션 — B의 fit이 조용히 무효화되고, 이를 잡을 공유 CI가 없습니다.

**구체적 실패.** 6주차에 A를 개선하며 출력 프레임을 바꿉니다. A의 단독 지표는 개선됩니다. 삽입 작업의 end-to-end 성공률은 55% → 30%로 떨어집니다. **로그상 두 모델이 개별적으로는 옳아 보입니다 — 인터페이스가 사람이 읽을 수 있는 포즈이기 때문입니다.**

**완화.** `(A버전, B버전, 캘리브레이션버전)`을 섞일 수 없는 **하나의 배포 아티팩트**로 고정하고, 메시지 헤더의 해시로 강제하십시오. 고정된 로그 인터페이스 trace를 B에 replay해 명령 경계와 smoothness를 assert하는 오프라인 CI 테스트를 추가하십시오.

---

### O13 — MODERATE: compliance 근거가 당신 하드웨어에서 무효일 수 있다

**반론.** 팔이 관절 위치 명령만 받는다면(SO-100/SO-101, Koch, 대부분의 취미용 팔, 많은 산업용 position-mode 인터페이스), 학습된 Model B는 접촉 작업에서 IK + 위치 servo와 *정확히* 같은 표현력을 갖고, stiffness 채널은 장식용 필드이며, 모든 접촉 근거가 증발합니다. 접촉 문헌은 명확합니다: "when the clearance between peg and hole is small, relying solely on velocity or position controllers is insufficient, as the tolerance and the robot's precision are at the same level (sub-millimeter)" **[P]**.

**완화.** 팔의 명령 인터페이스를 **먼저, 30분 안에** 확정하십시오. 위치 전용이면: 목표 작업을 자유공간 pick/place/reach로 **서면으로** 한정하고, 설계의 compliance 절반을 명시되지 않은 가정으로 안고 가지 말고 삭제하십시오.

> *(REV.2: G1은 **위치 전용**으로 확정됐습니다. 단 그리퍼 관절만은 velocity와 effort가 유효합니다.)*

---

## 6. 대안 아키텍처 3종 정면 비교

### 6.1 후보

**ΔChunk — "분리를 유지하고 나머지 전부를 고친다."** Model A = SmolVLA-450M(frozen SmolVLM backbone + LoRA), 256-float 스탬프 메시지 방출: 16 waypoint × (Δp, 6D-rot, 그리퍼, dt) + stiffness + swivel + 엔트로피 + phase + 60차원 latent 사이드 채널, GPU에서 2–4Hz 이벤트 트리거. Model B = TRAC-IK + impedance 위의 3.2M(v0) → 18M(v1) **clipped residual**, **CPU에서**. 둘 아래에 학습되지 않은 반사층, servo는 Mac 밖.

**CIR-1 — "Model B를 삭제한다."** 같은 chunk 단위 delta 인터페이스를 내는 450M 사전학습 VLA 1개, 로봇 자체 컨트롤러의 1kHz TRAC-IK + Cartesian impedance로 전달, 그리고 맨 아래에 학습된 것 정확히 하나: **0.18M 파라미터, 카메라 없는, hard-clipped force residual**(관절/속도/wrench/포즈오차/stiffness/phase에 대한 2-layer GRU → 6차원 wrench 보정, ±8N clip)을 impedance 컨트롤러에 합산. 목표 포즈와 subtask 예측은 **auxiliary 학습 손실**로 생존.

**TRISYS-500 — "모델 1개, 가중치 세트 2개, 클럭 도메인 3개."** π0.5/GR00T 형태. System 2 = **frozen** SmolVLM2-500M(32개 decoder 레이어 중 첫 16개), 그 **레이어 16 은닉 상태**(176×720, LayerNorm-Linear-GELU-Linear-LayerNorm adapter 경유)가 인터페이스 — 포즈가 아님. System 1 = **같은 forward 그래프**의 ~100M flow-matching DiT, 캐시된 prefix에 cross-attend, 3M conv stem을 통한 자체 128×128 손목 크롭, 4 flow 스텝으로 50스텝 chunk 단위 SE(3)-delta chunk 방출. 아래에 해석적 척수층. 목표 포즈와 subtask head는 prefix에서 갈라진 auxiliary 손실로 유지.

### 6.2 정면 비교

| 차원 | ΔChunk | CIR-1 | TRISYS-500 |
|---|---|---|---|
| 사용자가 명시한 아키텍처 보존 | **예** (모델 2개, 포즈 인터페이스) | 아니오 (모델 1개 + 고전 척추) | 부분 (가중치 2세트, 그래프 1개; 포즈는 aux 손실로 강등) |
| 인터페이스 | 포즈 chunk + 60차원 latent (하이브리드) | 포즈 chunk (순수) | **학습된 latent** (176×720) |
| 인터페이스 검사·사람 교정 가능 | **예** | **예** | 로깅용 aux head 디코딩으로만; **교정 불가** |
| 인터페이스 아래 학습 파라미터 | 3–18M residual | **0.18M** residual | ~100M action expert |
| 총 학습 파라미터 | ~468M | ~450M | 실행 ~370M / checkpoint 600M |
| 분리 비율 | 25–140:1 | **2500:1** | ~2.7:1 (단 그래프 1개) |
| 이 기기에서의 GPU 경합 | **없음** (A는 GPU, B는 CPU) **[M]** | **없음** (모델 1개) | **없음** (프로세스 1개, 그래프 1개) |
| 실측 빠른 루프 latency | 6.29ms p50 / 7.92 p99 (18M + 크롭, CPU, GPU 포화) **[M]** | **0.05ms p50 / 0.06 p99** (CPU 1스레드) **[M]** | 50-action chunk당 ~60–72ms (생성 14–17Hz) **[M 루프라인에서 I]** |
| 유효 명령 rate | 100–160Hz setpoint | 200Hz 재타이밍, 1kHz force | chunking + RTC로 매끄러운 30–50Hz |
| 연쇄 shift 노출 | 중간 (DAgger로 완화) | **거의 0** (고전 블록이 도달 가능한 모든 목표에 대해 구조적으로 옳음) | **0** (그래프 1개, joint 학습) |
| Joint 학습 붕괴 위험 (0/0/0/0/0) | 중간 — Stage-4 부분 joint finetune 필요 | **없음** | 필수 adapter 사전 정렬 단계로 완화 |
| Mac에서 학습 가능 | Stage 1–2는 GPU 대여 필요 | A는 ~1 GPU-day 대여; residual은 **Mac CPU에서 20분** | **불가** — A100/4090에서 24–48h 필요 |
| 작동하는 로봇까지 시간 | ~4–6주 | **~2주** (residual off가 이미 출하 가능한 시스템) | ~4–5주 |
| 첫 정직한 숫자까지 시간 | ~5–6주 | **~2–3주** | ~5주 |
| 능력 천장 | SmolVLA급 | SmolVLA급 | SmolVLA급 (이 기기에서 3B는 prefill 3.74초 **[M]** — 배포 불가) |
| 접촉 집약 능력 | 하드웨어가 허락하면 양호 | 하드웨어가 허락하면 양호; residual이 *특정적으로* force 추종 보정기 | 양호; chunk에 stiffness 채널 |
| Cross-embodiment / 팔 교체 | 양호 (Cartesian 인터페이스) | 양호 | 약함 (embodiment별 인코더 없이는 latent가 embodiment에 엮임) |
| End-to-end gradient 봉쇄 | 부분 (Stage 4 선택) | **예, 설계상** | **아니오** — 그것을 유지하는 경로 |
| Graceful degradation | 예 (residual→0) | **예** (residual→0이면 고전 스택 전체 복원) | 부분 (S1 결함은 여전히 명령 경로, 척수층으로 완화) |

### 6.3 권장

**CIR-1을 먼저 만드십시오. TRISYS-500을 명시적 업그레이드 목표로 유지하십시오. human-in-the-loop 포즈 교정이 주된 제품 목표가 아니라면 ΔChunk를 만들지 마십시오.**

**CIR-1을 먼저 하는 이유 — 넷, 순서대로:**

1. **어차피 필수 baseline입니다.** 세 설계 전부 같은 척추가 필요합니다 — 무언가가 Cartesian 목표를 관절 운동으로 바꾸고 팔을 안전하게 지켜야 합니다. CIR-1*이* 그 척추 + 사전학습 VLA 1개입니다. 그것을 만드는 건 우회가 아니라 세 계획 어느 것의 첫 2주이고, 차이는 CIR-1이 거기서 멈추고 측정한다는 것뿐입니다.
2. **평가 계획이 요구하는 정직한 대조군입니다.** "`Model A + 고전 IK`가 분리의 5포인트 이내로 오면 Model B를 삭제한다"는 판정 규칙은 `Model A + 고전 IK`를 만들지 않고는 평가할 수 없습니다. 먼저 만들면 가설이 측정으로 바뀌고 — oracle ablation들(GHOST 36.7% vs 90%; RT-Affordance 68% vs 76%; PerAct/RVT가 학습 파라미터 0의 저수준으로 작업당 ~10 데모)을 감안하면 **정직한 사전확률은 그것이 이기거나 비긴다**입니다.
3. **엔지니어링 시간당 성공을 압도적으로 많이 삽니다.** 작동하는 로봇까지 ~2주 vs ~4–6주. 데이터셋 1개, 학습 1회, 모델 1개, 인터페이스 정렬 문제 없음, GPU 경합 문제 없음, 순차적으로 막히는 두 번째 데이터셋 없음, 재진입적 DAgger 유지보수 세금 없음.
4. **유일한 학습 컴포넌트의 역할이 한 문장에 들어가고 대응하는 고전 baseline이 있습니다.** 0.18M residual은 impedance 컨트롤러가 닫지 못한 추종 오차를 force와 proprioception으로 상쇄합니다. IK를 하지 않고, 계획을 하지 않고, visual servoing을 하지 않습니다. 라벨은 피드백 컨트롤러 자신의 출력 — Kawato의 등상섬유 유사체, 사람 라벨링 0. **0.05ms p50 [M]** 으로 측정되므로 1kHz 루프 안에 20배 여유로 돕니다. 제거하면 마비가 아니라 엉성한 추종이 나옵니다.

**TRISYS-500이 시작점이 아니라 업그레이드인 이유.** TRISYS는 목록에서 아키텍처적으로 *가장 방어 가능한* 설계입니다 — 모든 프로덕션 dual system이 수렴한 것과 일치합니다(frozen 사전학습 VLM, cross-attention 경유 중간 레이어 은닉 상태, forward 그래프 1개, action expert, aux subtask/포즈 손실, chunk 단위 delta, 아래에 해석적 척추). O3, O6, O10 일부, O12를 한 번에 구조적으로 제거하기도 합니다. 그러나 GPU 대여와 24–48시간 joint 학습이 들고, 인터페이스 검사 가능성을 포기하며(포즈가 latent 대비 갖는 유일한 이점이자 RT-H의 최대 단일 수치를 만든 것: 개입 40% → 63% **[X]**), 그리고 결정적으로 — **이 하드웨어에서 CIR-1을 명백히 이기지 못합니다.** 3.78 TFLOPS 가속기가 둘 다 SmolVLA급 능력에 캡하기 때문입니다. 자신의 가장 강한 대조군("같은 모델, aux 포즈 head, 런타임 계층 없음")이 그것과 비길 수 있고, 그것이 정확히 π0.5 ablation이 발견한 것입니다.

**ΔChunk가 지배당하는 이유.** 원안에 가장 충실하고 모든 구체적 결함을 올바르게 고칩니다. 그러나 4–6주의 엔지니어링 — 프로세스 2개, 버전 있는 스키마, 공유 메모리 transport, DAgger 라운드, 부분 joint finetune, 귀속 하네스 2개 — 을 근거가 반박하는 가설(C1)을 보존하는 데 쓰고, 그 3–18M residual은 모든 oracle ablation이 이미 포화됐다고 말하는 절반 위에 앉습니다. 정직한 v1 기대 결과는 "residual은 측정 가능한 일을 하지 않았고, 고전 코어가 일을 했다"입니다. **예외:** 결과물이 *디버깅 가능하고 사람이 교정 가능한* 로봇이라면 — RViz에 렌더링하고, 6-DoF 마우스로 끌고, 유닛테스트하고, 교정을 학습에 되먹일 수 있는 포즈 — 그러면 ΔChunk의 검사 가능한 인터페이스가 **곧 제품**이고, 성공률과 무관하게 올바른 선택입니다. 정당한 목표이며, 그것이 당신 목표라면 명시적으로 선언되어야 합니다.

**업그레이드 경로는 재작성이 아니라 config 변경입니다.** CIR-1과 TRISYS-500은 척추 전체, 안전 계층, 모델 아래 인터페이스 스키마, 평가 하네스를 공유합니다. CIR-1 → TRISYS 이동은: 단일 VLA를 frozen-VLM + action-expert 쌍으로 교체하고, aux 포즈 head는 하던 일을 계속하게 두는 것. **그것이 이 순서에 대한 가장 강한 단일 논거입니다.**

---

## 7. 실험 계획 (압축)

**연산에 하루 쓰기 전에 네 가설을 분리하십시오.** 셋은 싸고, 하나가 실제로 주장한 것입니다.

| | 가설 | 사전확률 | 검증 비용 |
|---|---|---|---|
| H1 | 파라미터/데이터/init/rate 고정 시 두 모델이 하나를 이긴다 | **부정적** | 며칠 (E0) |
| H2 | 분리가 달성 가능한 명령 rate를 올린다 | **교란됨** — SmolVLA는 단일 모델의 async chunking으로 얻음 | 1주 (E3), 측정 |
| H3 | 분리 덕에 상위가 값싼 비텔레옵 데이터를 먹는다 | **강하게 긍정적** | 2주 (E2c) |
| H4 | 포즈 인터페이스는 검사·사람 교정 가능하다 | **긍정적, 과소평가됨** | ~무료 |

*(E0–E5의 상세는 [02-empirical-plan.md](02-empirical-plan.md)에 전문이 있습니다. 아래는 압축본입니다.)*

### 사전 점검 (몇 시간)
- **P1 (30분):** 팔의 명령 인터페이스가 무엇을 받는가? 위치 전용 → O13 발동; 자유공간으로 서면 한정.
- **P2 (2시간):** 2프로세스 동시성 측정. M4-base 기준: ViT-B/16 **단독 62.75ms → 경합 214/289ms p50/p99 [M]**. 일치하면 **모델 코드를 쓰기 전에 H2가 죽습니다.**
- **P3 (5분):** 카메라 프레임 rate = Model B의 시각 반응성 천장. 적어두십시오.
- **P4 (반나절):** hand-eye extrinsic 오차 측정(AprilTag 그리드, AX=XB). 공차가 그 수치의 ~2배 미만인 작업은 절대 포즈 인터페이스를 쓸 수 없습니다.

### E0 — 가능한 가장 값싼 반증 (3–5일, 노트북, 새 텔레옵 0)

**oracle 인터페이스**(hindsight 정답 미래 EE 포즈, LIBERO 상태 vector의 numpy 슬라이스)로 저수준 정책을 학습시키십시오. 이것이 **아키텍처 전체의 천장**입니다 — 앞으로 만들 어떤 Model A도 정답을 이길 수 없습니다.

장소: LIBERO-Spatial + LIBERO-Object (20작업, 데모 1000개 동봉, macOS CPU에서 실행, robosuite가 config로 `OSC_POSE` ↔ `JOINT_POSITION` 전환). ~25M 파라미터 3개 arm, 3 seed, 각 500 에피소드 (n=500에서 MDE ≈ 8.9pt, n=1500에서 ≈ 5.1):

- **C0 MONO** — RGB + proprio → action chunk
- **C6 SPLIT-ORACLE** — 정답 목표 포즈 + 관절 + RGB → action chunk
- **C8 ORACLE-IK** — 정답 목표 포즈 → `OSC_POSE`, **학습 파라미터 0**

**Kill 규칙:**
- `C6 − C0 ≤ +5` 또는 CI∋0 → **H1 사망.** oracle 정보는 특권적입니다. GHOST의 oracle→배포 낙폭은 −53pt, RT-Affordance는 −8. 천장이 얇다는 것은 배포된 분리가 monolith *아래로* 확실히 간다는 뜻입니다.
- `C8 ≥ C6 − 5` → **Model B 사망.** 파라미터 0이 25M과 동률. 프로젝트는 CIR-1이 되고 주말에 출하되며 예산 전체가 Model A로 해방됩니다.
- `+5 < C6 − C0 < +10` → **LIBERO-Long**과 **CALVIN ABC→D**에서만 재실행. 천장이 단기에서 얇고 장기에서 두꺼우면 이건 장기 프로젝트입니다 — 계속하기 전에 서면으로 선언하십시오.

### E1 — 모든 교란변수로부터 분리를 격리 (2–3주, GPU 대여 ~$100–250)

총 파라미터 P / 데모 D / init / 유효 rate R / 행동 표현을 전부 일치시킨 사다리:

| ID | Arm | 무엇을 분리하는가 |
|---|---|---|
| C0 | MONO | baseline |
| C0s | P_A와 P_B에서의 MONO | 파라미터 스케일링 곡선 — 없으면 "분리 이득"과 "유효 모델이 커진 것"을 구분 불가 |
| **C1** | **MONO-AUX** — C0 + auxiliary 미래 포즈 head, 런타임 미사용 | **π0.5 "implicit HL" 대조군.** 학습 신호로서의 분해 vs 아키텍처로서의 분해 |
| C2 | MONO-2PASS — C1 가중치, 실제 2단 런타임 추론 | 런타임 계층 vs 파라미터 분리 |
| C3 | SPLIT-JOINT — 네트워크 2개, 연속 인터페이스 관통 gradient | 네트워크 2개 vs *독립 학습된* 네트워크 2개 |
| **C4** | **SPLIT-SEQ — 명시된 그대로의 제안** | 만들고 싶은 것 |
| C5 | SPLIT-SEQ + 인터페이스 DAgger 1라운드 | covariate-shift 세금 |
| C6 | SPLIT-ORACLE (E0에서 이월) | 아키텍처 천장 + 귀속 분모 |
| **C7** | **SPLIT-IK — 학습된 A → 해석적 IK + OSC** | Model B가 학습될 필요가 있는가 |

교차 교란(전수 교차가 아니라 단계적 스크린): **pretraining** {random, 사전학습 비전, 전체 VLM} — 이것만으로 ~20pt이고 이걸 측정하고 있는 것일 수 있음; **총 파라미터** {10, 25, 60, 150M}; **chunk horizon** {1, 16, 50}; **HL 갱신 간격** {1, 10, 30, 60}; **일치 vs native 명령 rate**; **행동 표현** {EE절대, EE-chunk-delta, 관절절대, 관절-chunk-delta}; **인터페이스 인코딩** {raw vector, waypoint chunk, GMM heatmap, 포즈+latent}; **데모 수** {10, 25, 50, 100, 200}/작업.

**GO 조건은 `C5 − max(C0, C1, C2, C7) ≥ +8`이 4개 스위트 중 ≥3에서, 95% CI 비중첩으로.** **이름 붙은 NO-GO:** `C1 ≥ C5−3` → 이득은 학습 신호였음, C1 출하. `C7 ≥ C5−3` → Model B 불필요. 사전학습 init에서 우위 소멸 → pretraining을 측정한 것. rate 일치 시 우위 소멸 → latency를 측정한 것. `C3 ≥ C5+8` → 분리 학습이 결함. `C5 − C4 ≥ +10` → DAgger가 영구 필수.

### E2 — 계층이 이기는 것이 *허용되는* 영역 (1–2주)
- **E2a 동적:** CALVIN-D / ManiSkill3에서 물체 속도 {0, 1, 3, 10} cm/s. 평균이 아니라 **기울기**를 보고. HiRT의 동적 격차는 75 vs 48, 정적은 70.0 vs 71.3. 속도에 평평 → H2 사망.
- **E2b 교란 복구:** 물체 밀기 {2, 5, 10} cm, 200ms 동안 5N 팔 밀기, 0.5초 가림, 그리퍼 미끄러짐. 지표: 교란 전 정상 궤도 조건부 복구율. 그리고 **livelock 비율.**
- **E2c 데이터 출처 (H3 — 아마 진짜 결과):** `C4+ARM`(A를 주석 정지영상 ~750–2000장 + B가 소비 불가한 웹/시뮬 데이터로 추가 학습) vs `C0+ARM`(정직한 대조군 — monolith도 흡수할 수 있는가?). **GO 신호:** `C4+ARM − C4 ≥ +15` 이면서 `C0+ARM − C0 < +5`.
- **E2d 일반화, 7축 각각 따로 보고, 절대 평균 금지:** 새 인스턴스 / 새 카테고리 / 학습 convex hull 밖 위치 / 조명 / 카메라 포즈 / 방해물 / 지시문 재표현. **각각을 C6과 함께 실행** — 축별 `C6 − C4` 격차가 곧 실패 귀속입니다.
- **E2e:** 실기체에 몇 달 쓰기 전 SimplerEnv 상관 점검.

### E3 — 대상 기기에서의 rate·시스템 측정 (1주, 학습 없음)
4개 구성 측정: (1) MONO + chunking + async; (2) SPLIT, 프로세스 2개; (3) B에서 RGB를 뺀 SPLIT; (4) A는 MLX/Metal, B는 CoreML을 `cpuAndNeuralEngine`에 고정하고 GPU 명시적 제외한 SPLIT — **`powermetrics`로 ANE 상주 검증**, CoreML이 조용히 GPU로 폴백하기 때문. **H2 사망 조건:** MONO+chunking이 p99 < 2×p50으로 ≥30Hz 달성. **GO 조건:** SPLIT이 MONO 유효 rate의 ≥2배를 빠른 루프 p99 < 50ms로 달성.

### E4/E5 — 실기체 (3–4주 후 4–6주; **E1이 ≥15pt를 준 경우에만**)
15포인트 기준선은 임의가 아닙니다: 실기체가 감당하는 arm당 ~400 rollout이 80% 검정력에서 ~10pt를 탐지하고, HiRT의 실기체 준정적 격차는 **1.3포인트**였습니다. E1이 시뮬에서 8포인트를 줬다면 실기체는 분해할 수 없습니다.

**텔레옵 예산:** E0/E1/E2a/b/d는 **새 데모 0**(벤치마크 동봉). E2c는 *정지영상* ~750–2000장(~1h + 주석 2–3h), 텔레옵 0. E4 파일럿: **5작업 × 50데모 = 250, 리셋 포함 ≈6–10h**. E5: **새 데모 0**. 실기체 표본 효율 곡선이 필요하면: +750.

**하드웨어:** SO-101 페어(~$400, 위치 전용 — compliance 검증 불가)가 H1/H3/H4에 대한 E4/E5에 충분합니다. **E2가 접촉 작업에서 분리의 승리를 보여준 경우에만** Franka를 1주 대여/차용하십시오. H1을 검증하려고 Franka를 사지 마십시오.

**E4 전 필수:** 반사층, 타입 있고 타임스탬프 있는 인터페이스, Oracle-A/Oracle-B 하네스, 오프라인 replay 회귀 테스트. 하네스 없이는 모든 실기체 실패에 구분 불가능한 원인이 넷 있고 각각 일주일이 듭니다.

**지표(분포로 보고, 절대 평균 금지):** 성공 + 진행 점수; 제어 rate p50/p5; photon-to-command latency p50/p95/p99/max; **jitter 비율 p99/p50**(건강한 edge 참조: 평균 150.5ms, **표준편차 0.13ms**); 인터페이스 staleness p50/p99; queue underrun 비율; **task phase별로 나눈** 포즈 추종 오차 병진/회전 RMS+p95; 인터페이스 오차 ‖A의 포즈 − 정답‖(가장 진단적인 단일 수치); 최종 배치 오차; RMS 관절 저크 + 관절 속도 부호 반전(IK branch-flipping과 chunk 경계 저크를 잡음); 피크·RMS 접촉력; jam/stall 비율; 복구율·복구 시간; livelock 비율; 축별 일반화; 교차점 N*이 있는 성공률-vs-N데모; 100 rollout당 안전 트립(관절 한계, 속도, IK-infeasible, 작업공간, NaN, watchdog, 사람 개입); **oracle 격차 C6−C4**와 **고전 격차 C7−C4**.

**총평: 핵심 주장을 $0으로 일주일 안에 반증할 수 있습니다(E0). 이후 전부가 E0 통과에 조건부이고, E0는 천장을 검증하므로 거기서의 음성은 더 많은 데이터로도, 더 나은 Model A로도, 더 큰 예산으로도 구제되지 않습니다.**

---

## 8. 당신만 답할 수 있는 질문

*(이 표는 메인 문서 13절에 최신 상태로 옮겨져 있습니다. Q1·Q2·Q4·Q10은 이후 G1 확정으로 답이 나왔습니다.)*

| # | 질문 | 왜 결정적인가 | 각 답이 권고를 어떻게 바꾸는가 |
|---|---|---|---|
| **Q1** | **팔의 명령 인터페이스가 무엇을 받는가 — 관절 위치 전용인가, 토크 / Cartesian impedance / 가변 강성인가?** | compliance 논거 전체를, 따라서 학습된 저수준이 IK가 못 하는 일을 *하나라도* 할 수 있는지를 게이팅 | **위치 전용** → stiffness 채널과 접촉 근거 삭제; 자유공간 pick/place/reach로 서면 한정; CIR-1의 residual은 보정할 것이 거의 없음 — CIR-1 마이너스 residual, 즉 `VLA + IK + retimer` 출하. **토크/impedance(Franka FCI, KUKA FRI, QDD)** → residual이 진짜 역할을 함; 명세대로의 CIR-1; 접촉 작업이 범위에 진입; 손목 F/T 센서($3–5k)가 가치 대비 최고의 센서 구매 |
| | | | ✅ *REV.2: **위치 전용**으로 확정. 단 그리퍼만 velocity/effort 유효* |
| **Q2** | **6-DoF 팔인가 7-DoF 팔인가?** | 7-DoF에서 EE 포즈는 *underdetermined* | **6-DoF** → 중복성 문제는 경미(branch 선택뿐, IK 시딩으로 처리); 인터페이스에서 swivel 필드 생략 가능. **7-DoF** → swivel/arm-angle 스칼라가 인터페이스에 **필수**, IK 호출에 null-space 자세 편향 필수. 신경망이 아니라 숫자 7개짜리 수정 |
| | | | ✅ *REV.2: **7-DoF × 2로 확정**, 그리고 canonical S–R–S이므로 `(포즈, ψ)`에 closed-form 해가 존재* |
| **Q3** | **목표 작업 종류는 — 자유공간 pick-and-place, 접촉 집약(삽입/닦기/문), 변형체, 장기 다단계 중?** | 실측 계층 프리미엄은 **단기에서 정확한 동률**(69.57 vs 69.63)이고 **장기/reasoning에서 +30~+42pt**. 포즈 waypoint 인터페이스는 준정적 keyframe 작업에서 입증됐고 연속/접촉 작업에서 실패한다고 문서화됨 | **단기 pick-and-place** → 분리는 아무것도 사주지 않는 것으로 측정됨; monolith(CIR-1)를 만들고 멈추십시오. **장기/다단계** → 계층이 진짜로 정당화됨; 명시적 전환/종료 정책도 추가(O9). **접촉 집약** → Q1이 hard 전제조건; 인터페이스*와* Model B 입력 양쪽에 force 추가. **변형체** → 어려운 쪽은 Model B가 아니라 Model A; 전부 Model A에 |
| **Q4** | **움직이는 헤드/베이스가 설계에 포함되는가, 조작 중 카메라가 정적인가?** | 카메라 프레임 포즈는 *촬영 시각의* 헤드 extrinsic 없이 무의미; 500ms latency와 20°/s 팬이면 ~63mm의 조용한 오차 | **정적 카메라** → 카메라 프레임 예측이 안전하고 +8~+14pt 가치; 타임스탬프 계약은 위생이지 긴급하지 않음. **움직이는 헤드** → 촬영 시각 헤드 상태 스탬프와 즉시 base_link 변환이 **필수이고 핵심**; v1 단순화로 조작 단계 동안 헤드 freeze 고려 |
| | | | ✅ *REV.2: 목 2-DoF 구동, 틸트 40.6°뿐이라 **조작 중 토르소 상시 이동**. head→arm_base는 목 관절 2개에만 의존* |
| **Q5** | **Mac mini가 hard 제약인가 선호인가?** | 실측 천장 **3.78 TFLOPS / 104 GB/s [M]**; 3B trunk prefill이 **3.74초 [M]**. 같은 유선 스위치의 중고 RTX 3090/4090이 backbone을 ~8–10배 빠르게, 진짜 priority stream과 TensorRT int8과 함께, M4 Pro 업그레이드 정도 가격으로, ~0–4ms 네트워크 비용에 돌림 | **Hard 제약** → Model A를 ~450M–1B로 캡, GPU에 모델 1개, 빠른 경로의 모든 학습 컴포넌트는 CPU, servo는 Mac 밖. SmolVLA 규모의 CIR-1 또는 TRISYS-500. **선호** → GPU 박스를 사고 π0.5급 모델을 돌리십시오. 그리고 monolithic 32–57Hz면 분리의 latency 근거 전체가 증발한다는 점에 유의 |
| | | | ✅ *REV.3: RTX 3090 박스 확보. 그리고 REV.2: SDK가 Linux 전용이라 Mac은 로봇과 통신 불가* |
| **Q6** | **사전학습 VLM이 허용되는가, 전부 로컬에서 처음부터 학습해야 하는가?** | 랜덤 init vs 사전학습이 **~20pt (LIBERO 77.5% → 97.8%)** — 문헌의 모든 단기 계층 프리미엄보다 큼 | **사전학습 허용** → 사전학습 VLM을 Model A로 freeze, 나머지 LoRA; 세 대안 모두 유효. **처음부터 강제** → ~20포인트를 포기하는 것이고 이는 어떤 아키텍처 선택도 회복할 수 없는 양; 분리의 가장 강한 지지 메커니즘(인터넷 사전학습 frozen 상단)이 불가능해지고, 정직한 권고는 "monolith를 만들고 계층에 기대하지 말라"가 됨 |
| **Q7** | **손 주석 정지영상 ~750–2000장(≈수집 1h + 주석 2–3h)을 모을 의향이 있는가?** | 모든 포즈 인터페이스 성공의 *실제* 메커니즘. RT-Affordance 예측기: 없으면 77% → **24%**, 웹 공동학습 없으면 → **11%** | **예** → H3가 검증 가능하고 아마 진짜 결과; 분리가 데이터 접근으로 존재를 획득; E2c 실행. **아니오** → 분리의 유일하게 잘 입증된 성능 이점이 불가능; 모든 비용을 치르고 이득은 0; CIR-1(모델 1개)이 명백히 옳음 |
| **Q8** | **목표가 연구 결과인가, 작동하는 로봇인가, 디버깅/교정 가능한 로봇인가?** | 세 목표가 세 개의 다른 아키텍처를 가리킴 | **작동 로봇** → CIR-1, ~2주. **연구 결과** → 이 공간에서 유일하게 진짜 미검증인 구성은 원안의 학습된 pose→joint 저수준이고, 미검증인 이유는 작동하지 않기 때문. 흥미로운 미검증 질문은 대신 "모든 것을 일치시킨 상태에서 명시적 포즈 vs 학습된 latent"이고 이건 아무 논문도 돌리지 않음. **디버깅/교정 가능** → ΔChunk; 포즈 인터페이스가 제품, RT-H의 개입(40% → 63%)이 근거, 성공률 동률이 수용 가능한 결과 |
| **Q9** | **조건당 실기체 rollout을 실제로 몇 번 감당할 수 있는가?** | HiRT의 분리-vs-monolith 격차는 **1.3포인트**; TRI는 정책 구분에 Bonferroni 보정 포함 1,800 시행 필요 | **<20/조건** → 이 설계들을 구분할 수 없음; 실기체 비교를 하지 말고 E0/E1을 시뮬에서 돌린 뒤 더 단순한 쪽을 출하. **50–100/조건** → ~10–15pt 탐지 가능; E1 GO 기준을 ≥15pt로. **≥400/조건** → 명세대로의 E5가 의미 있음 |
| **Q10** | **단일 팔인가 양팔인가?** | 각 5mm 오차의 독립적 절대 목표 2개는 최대 10mm *상대* 오차로 합성되고, 강체 양손 파지에서 이는 스퀴즈(내부 힘, 관절 정지) 또는 갭(낙하) | **단일** → 위의 전부가 적용. **양팔** → **상대 변환**이 독립적인 두 포즈가 아니라 first-class 인터페이스 필드여야 하고, IK/impedance 정식화가 협응되어야 함. 진짜로 더 많은 엔지니어링이고 v2로 명시적으로 미뤄야 함 |
| | | | ✅ *확정: **양팔.** 상대 변환이 first-class 필드여야 함* |

---

## 9. 단계별 로드맵

*(이 로드맵은 메인 문서 14절에 REV.2/REV.3 반영본으로 옮겨져 있습니다.)*

### Phase 0 — 결정하고 측정 (1주, $0)
1. Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q10에 **서면으로** 답하십시오. 여럿이 이 계획의 가지 전체를 삭제합니다.
2. P1–P4 실행: 하드웨어 인터페이스 조사, 실제 기기에서의 2프로세스 동시성 측정, 카메라 rate 천장, hand-eye 캘리브레이션 오차 예산.
3. **게이트:** P2가 6.3배/9.2배 경합 페널티를 재현하면 **[M]**, 이 하드웨어에서 H2는 죽었습니다. 분리는 이 시점부터 CPU/GPU 분할이거나 단일 그래프여야 합니다.

### Phase 1 — 척추 (빌드 1–5일차, 학습 0, 데이터 0)
URDF → TRAC-IK → Ruckig 저크 제한 retimer → 팔 자체 컨트롤러나 유선 이더넷 Linux SBC의 Cartesian impedance(또는 위치 servo) → **학습되지 않은 반사층**(clamp, 임계값, 작업공간 AABB, 도달가능성 + self-collision 필터, NaN assert, 스텝 크기 제한기, deadman watchdog) → replay 도구가 딸린 **타입 있고 타임스탬프 있고 버전 있는 인터페이스 스키마.**

**인수 테스트:** 6-DoF 마우스로 3Hz로 chunk 메시지를 쓰며 **인터페이스를 통해** 팔을 텔레옵하십시오. 그것이 안 되면 어떤 모델 학습도 도움이 안 됩니다.

**산출물:** 안전하게 움직이는 로봇과, 로깅·replay·유닛테스트 가능한 스키마.

### Phase 2 — 값싼 반증 (E0, 3–5일, 노트북, $0)
LIBERO 학습 3회: MONO, SPLIT-ORACLE, ORACLE-IK. 각 1,500 에피소드. **프로젝트 전체에서 가치가 가장 높은 한 주입니다** — 아키텍처의 *천장*을 검증하므로 음성 결과가 더 나은 Model A나 더 많은 데이터로 구제되지 않습니다.

**게이트:** `C6 − C0 ≤ +5` → H1 사망, 곧장 Phase 3으로 CIR-1으로 가고 분리 걱정을 그만두십시오. `C8 ≥ C6 − 5` → Model B 영구 사망.

### Phase 3 — Model A + 척추, residual off (2–4주차)
- 텔레옵: 8–12작업에 걸쳐 **250–400 에피소드** (~6–10h).
- **`lerobot/smolvla_base`에서** finetune — 절대 처음부터 하지 마십시오. LM에 LoRA, action expert는 전체 학습. GPU 대여 ~1일 (~$20–60). **Mac에서 finetune하지 마십시오.**
- 데이터 mixture: 자체 텔레옵 ~60%, 주석 정지영상 750–1500장 ~15%, 웹 VQA/grounding ~15%(망각 방지), 자동 추출 subtask 라벨 ~10%(그리퍼 전환 keyframe, 사람 라벨링 0).
- 비용이 거의 없으면서 분해 이득 대부분을 나르는 auxiliary 손실: **목표 포즈 head**(OpenHelix의 단일 최대 이득, 3.45 → 4.01)와 **subtask 언어 head**(π0.5의 "implicit HL" 발견).
- 인터페이스: chunk 단위 SE(3) delta, 6D rotation, 그리퍼 aperture, dt, stiffness, swivel, 신뢰도, phase.
- Async chunk 실행: H=50, 큐 ~60–70% 소진 시 재계획, Real-Time Chunking prefix freezing, 가중 평균 temporal ensembling.
- **Oracle-A와 Oracle-B 하네스를 나중이 아니라 지금** 만드십시오.

**산출물: 4주차 말에 residual을 끈 채 작동하는 로봇.** 이것이 나머지 전부가 이겨야 할 baseline입니다.

### Phase 4 — 정직하게 측정 (4–6주차)
E3(대상 기기의 rate/jitter) + 시뮬의 E1 사다리 + E2a/E2b/E2d. 작업별·축별로, 평균이 아니라 p99로 보고하십시오. 이름 붙은 NO-GO 판정을 써서 분리에 정당화할 것이 남았는지 결정하십시오.

**분기점.** `C1 ≥ C5 − 3` 또는 `C7 ≥ C5 − 3`이면 끝입니다: Phase 3을 출하하고, 음성 결과를 정직하게 쓰고, 멈추십시오. **정당하고 유용한 결과입니다.**

### Phase 5 — Residual (7주차, ~2일)
Phase 3 배포 후에만. A + 척추를 **residual = 0**으로 rollout, 200Hz 로깅, 라벨 = impedance 컨트롤러가 닫지 못한 오차(피드백 컨트롤러 자신의 출력이 교사 신호 — 사람 라벨링 0). 0.18M GRU를 **Mac CPU에서 ~20분** 학습. residual이 자기 입력 분포를 바꾸므로 DAgger 1라운드. ±8N hard clip과 L1 크기 페널티를 추가해 graceful degradation을 희망이 아니라 학습 목표로 만드십시오.

### Phase 6 — E2c, 데이터 출처 검증 (7–9주차)
H3를 검증하는 실험이고, 아마 당신의 진짜 결과입니다. `C4+ARM` vs `C0+ARM`. **여기서 분리가 존재를 획득하거나 불필요함이 확인됩니다.**

### Phase 7 — 조건부: 실기체 검정력 비교 (9–15주차)
E1이 ≥15포인트를 준 경우에만. E4 파일럿(3 arm × 100 rollout, 계측 shakedown) 후 E5(3 arm × 400 rollout, blind 무작위, Bonferroni 보정). 정직하게 예산: n=400에서 최소 탐지 효과는 ~9.9포인트.

### Phase 8 — 조건부 업그레이드: TRISYS-500 (12–17주차)
Phase 4가 Model A에 진짜 능력 천장을 보였고 더 크거나 다르게 구조화된 상위가 그것을 들어올릴 경우에만. 단일 VLA를 하나의 그래프 안의 frozen-VLM + adapter + flow-matching action expert로 교체하되, **adapter 사전 정렬을 별도 단계로 필수**(OpenHelix: 건너뛰면 0/0/0/0/0), joint 공동학습, 학습 마지막 20%에 staleness conditioning. 척추·안전 계층·스키마·평가 하네스는 불변 — **그것이 순서가 중요한 이유입니다.**

### 명시적 off-ramp
- **Q6 = 처음부터 학습 강제** → Phase 5–8 생략; 어떤 아키텍처도 회복 못 할 것을 이미 포기했습니다. monolith를 만들고 기대치를 그에 맞추십시오.
- **Q1 = 위치 전용 팔** → Phase 5 생략; `VLA + IK + retimer` 출하; 작업 목록을 서면으로 제한.
- **Q3 = 단기 pick-and-place 전용** → Phase 6–8 생략; 이 영역에서 분리는 아무것도 사주지 않는 것으로 측정됨.
- **Q8 = 디버깅/교정 가능 로봇이 목표** → Phase 4 이후 ΔChunk로 전환하고 개입 도구를 만드십시오(RViz 목표 렌더링, 6-DoF 마우스 교정, 교정 replay 학습). 성공률 동률을 올바른 결과로 수용하고 대신 개입 이득을 측정하십시오(RT-H의 40% → 63%가 재현 목표).

---

## 잔여 불확실성 — 있는 그대로

1. **결정적 실험이 존재하지 않습니다.** 총 파라미터 고정·데이터 고정·rate 일치·init 일치로 monolith vs 포즈 인터페이스 계층 vs latent 인터페이스 계층을 비교한 출판 연구가 없습니다. "분리는 파라미터당 정확도를 사지 않는다"는 제 결론은 인접 근거(MoE scaling law, π0.5의 implicit-HL ablation, HiRT의 monolith 마진, FiS-VLA의 파라미터 공유 결과, pretraining ablation)로부터의 *추론*이지 직접 반박이 아닙니다. **E0/E1은 정확히 이 공백을 당신 환경에서 닫도록 설계되었습니다.**

2. **여러 핵심 수치가 미검증 2026 preprint입니다.** 행동 공간 표(69.0 / 77.3 / 88.0 / 89.6), LIBERO pretraining 격차(77.5% → 97.8%), 오케스트레이션 연구의 성공률, GHOST의 oracle 교체, HDP의 24.55% IK 오류율, ER/ERJ 8%/94%/96% 삼중값 — 전부 최종 문서에 인용하기 전 원 출처 재검증이 필요합니다. 문서 전체에서 **[X]** 로 표시했습니다.

3. **π0.5의 Figure 13 수치를 추출하지 못했습니다** — 저자들이 명시한 순서(full > implicit HL > no HL)와 축자 결론만 보고합니다. 이것이 리뷰 전체에서 가장 중요한 반대 근거이므로, 그것을 근거로 build/no-build 결정을 하기 전에 arXiv 2504.16054의 ablation 절을 직접 읽으십시오.

4. **Helix는 정량 결과를 전혀 공개하지 않습니다.** 7B/80M, 7–9Hz/200Hz, 단일 latent, 전용 GPU, end-to-end 역전파 세부는 전부 Figure AI 자체 블로그 출처입니다. 아키텍처 서술은 신뢰 가능, 성능 주장은 미검증.

5. **GR00T N1은 분리-vs-monolith ablation을 공개하지 않습니다.** 제안이 수사적으로 가장 닮은 시스템이, 확인 가능한 범위에서, **자기 분리가 monolith를 이긴다는 증거가 없습니다.** 설계가 Kahneman 인용으로 정당화되어 있습니다.

6. **VLA급 정책이 hand-eye 캘리브레이션 오차에 따라 어떻게 열화되는지 측정한 출판 연구가 없습니다.** 2.1mm / 3.2° 수치는 캘리브레이션 방법론 논문에서 온 것이지 정책 성능 연구에서 온 것이 아닙니다. 캘리브레이션 오차가 절대 포즈 인터페이스를 유의미하게 제약한다는 추론은 peg-in-hole 공차 논거로부터의 제 추론이지 **[I]** 직접 측정 결과가 아닙니다.

7. **제가 보고한 Apple Silicon 수치는 실측 [M] 이지만, M4 Pro가 아니라 M4 *base*에서입니다.** M4 Pro는 연산과 대역폭 둘 다 대략 2배일 것입니다. 또한 해소하지 못한 직접 모순 데이터 하나: 한 출처가 ResNet-50이 Apple Silicon에서 RTX 4090보다 ~3배만 느리다고 주장하는데 — 거의 확실히 batch-1 latency가 아니라 대배치 처리량 비교이지만, 외삽에서 가장 큰 단일 불확실성 원천입니다. **어느 숫자도 믿지 말고 측정하십시오.**

8. **학습 실행 가능성을 깊이 조사하지 않았습니다** — 추론과 배포만 했습니다. 두 결합된 정책을 joint로 학습하는 것(Helix와 GR00T가 실제로 하는 것)은 상당히 더 큰 연산 문제이고, Mac mini는 거의 확실히 그것에 부적합합니다. **세 대안 모두 주 학습 실행에 GPU 대여를 전제합니다.**

9. **MLX 동시성 이슈의 해결 상태를 검증하지 않았습니다.** 이슈 #3078은 maintainer 수정 없이 종료된 것으로 보이지만 MLX는 빠르게 움직입니다. 30분짜리 실험 — MLX 프로세스 2개, 하나는 3B 모델을 루프로, 하나는 50M 모델을 100Hz로, 작은 모델의 p99를 로깅 — 이면 확정적으로 판정되고, 이를 전제로 설계하기 전에 실행해야 합니다.
