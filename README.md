# physical_AI

피지컬 AI **대뇌–소뇌 이중 모델 아키텍처**에 대한 기술 분석 및 설계 문서.

하나의 monolithic VLA에 모든 파라미터를 쏟는 대신, 상위 모델(대뇌)이 "손을 어디에 둘지"를 예측하고 하위 모델(소뇌)이 "어떻게 움직일지" 관절값을 결정하는 2단 분리 구조를 평가합니다.

**대상 구성**

| 항목 | 값 |
|---|---|
| 로봇 | Galbot G1 (양팔 7-DoF × 2, 목 2-DoF, 다리/허리 5, 손목 F/T ×2, 옴니 베이스) |
| 학습·시뮬 | RTX 3090 24GB / 64GB RAM / Ubuntu |
| 설계·검증 | Mac mini M4 32GB |
| 배포 추론 | 로봇 온보드 AGX Orin 64GB |
| 데이터 | 텔레오퍼레이션 자체 수집 + RoboCOIN (공개 실기체 G1 데이터) |

---

## 문서

### 메인

**[docs/2026-07-28-dual-brain-architecture-analysis.md](docs/2026-07-28-dual-brain-architecture-analysis.md)** — 현재 유효 개정 **REV.3**

문서 앞부분의 "이 문서 읽는 법"을 먼저 보십시오. 초판은 느슨한 서술을 근거로 작성됐고, 이후 정확한 입출력 사양과 대상 로봇이 확정되면서 **일부 결론이 반증됐습니다.** 개정 블록이 본문보다 앞에 놓인 이유입니다.

| 개정 | 계기 |
|---|---|
| **REV.3** | 연산 자원 확정, 시뮬 임베디먼트 결정 |
| **REV.2** | 대상 로봇 Galbot G1 확정 (공식 URDF/SDK 직접 검증) |
| **REV.1** | 정확한 I/O 다이어그램 (손목 F/T가 입력에 있음) |
| 본문 0–15절 | 초판 — REV.1–3이 뒤집은 항목 제외하고 유효 |

### 원자료 — [docs/research/](docs/research/)

리서치 에이전트 32개의 산출물 (영어 원문, findings별 출처·신뢰도 표기).

| 파일 | 내용 |
|---|---|
| [01-synthesis-principles.md](docs/research/01-synthesis-principles.md) | 주장별 채점표 C1–C14, 선행연구 지도, 반론 O1–O13, 대안 비교 |
| [02-empirical-plan.md](docs/research/02-empirical-plan.md) | 실험 계획 E0–E5, 대조군 사다리, 지표 정의, GO/NO-GO 표 |
| [03-synthesis-config-specific.md](docs/research/03-synthesis-config-specific.md) | 양팔·로컬 추론·자체 텔레옵·사업화가 판정을 바꾸는 방식 |
| [04-research-digest.md](docs/research/04-research-digest.md) | 계층형 VLA / 인터페이스 표현 / 고전 로보틱스 / 엣지 배포 |
| [05-critiques.md](docs/research/05-critiques.md) | 적대적 비판 3렌즈 — "IK 재발명" / "정보 병목" / "시스템·안전" |
| [06-alternatives.md](docs/research/06-alternatives.md) | 대안 아키텍처 3종 (ΔChunk / TRISYS-500 / CIR-1) |
| [07-REVISION-01-with-ft.md](docs/research/07-REVISION-01-with-ft.md) | F/T 입력 반영 개정 — 어느 결론이 철회됐는지 |
| [08-adversarial-recheck.md](docs/research/08-adversarial-recheck.md) | 초판 결론에 대한 적대적 재검증 |
| [09-recheck-findings.md](docs/research/09-recheck-findings.md) | 힘 조건부 정책 · 7-DoF 중복성 · 헤드 프레임 · 누락 필드 |
| [10-REVISION-02-galbot-g1.md](docs/research/10-REVISION-02-galbot-g1.md) | Galbot G1 확정 사실로 설계 질문 해소 |
| [11-galbot-g1-platform-facts.md](docs/research/11-galbot-g1-platform-facts.md) | G1 플랫폼 사실 — 검증된 것 / 확인 못 한 것 |

### 벤치마크 — [docs/research/benchmarks/](docs/research/benchmarks/)

**M4 Mac mini에서 실제로 실행한** 측정 스크립트. 문서에서 `[M]`으로 표기된 수치의 출처입니다.

| 스크립트 | 측정 대상 |
|---|---|
| `hybrid.py` | 대뇌 컴포넌트별 지연 (SigLIP / LM 트렁크 / flow expert) + DLS IK |
| `modelb.py` | 소뇌 CPU vs MPS, GPU 경합 하 |
| `s2s1.py` | System-2 / System-1 동시 실행 |
| `benchmjx.py` | MuJoCo / MJX |

```bash
python3 docs/research/benchmarks/hybrid.py   # PyTorch + MPS 필요
```

하드웨어나 macOS 버전이 다르면 숫자가 달라지지만, 결론의 방향은 바뀌지 않습니다.

---

## 핵심 결론 요약

**계층 분리 자체는 유효합니다.** 현재 작동하는 모든 프론티어 시스템(Helix, GR00T N1, π0.5, RoboDual)이 채택한 구조입니다.

**그러나 인터페이스가 틀렸습니다.** 그들 중 누구도 6-DoF 절대 포즈를 인터페이스로 쓰지 않습니다 — 전부 학습된 잠재벡터 또는 언어입니다. 행동공간 정면 비교에서 EE 절대 포즈는 4개 선택지 중 꼴찌(69.0% vs EE 델타 89.6%)입니다.

**"파라미터를 나누면 성능이 오른다"는 명제는 문헌 근거를 찾지 못했습니다.** 가장 잘 통제된 비교(HiRT)에서 정적 작업 기준 계층형 70.0 vs 단일 71.3으로 분리가 근소하게 졌습니다. 분리가 이긴 사례는 전부 **주파수 분리 · 값싼 상위 데이터 · 사전학습** 중 하나로 설명됩니다.

→ 정당화를 **데이터 재사용 + 주파수 분리 + 상업적 업데이트 주기**로 교체해야 합니다.

**손목 F/T 입력이 학습된 저수준 모델을 조건부 정당화합니다** (REV.1). 렌치는 기구학의 함수가 아니므로 정보 퇴화가 해소됩니다. 다만 최적화 퇴화는 남아, 해석적 IK 항을 명시한 **잔차 구조 + 접촉 게이팅**이 필요합니다.

**Galbot G1 확정 사실** (REV.2, 공식 URDF/SDK 직접 검증):
- 팔 명령은 **위치 전용** — 학습된 컴플라이언스는 도달 불가 (그리퍼만 velocity/effort 유효)
- 팔은 **정준 S–R–S 7-DoF** → arm angle ψ 하나 추가하면 폐형 해석 IK 존재
- `head → arm_base`는 **목 관절 2개**에만 의존 (다리/허리/베이스는 공통 조상이라 소거)
- 목 틸트가 **40.6°뿐** → 테이블을 보려면 토르소를 내려야 하므로 **조작 중 토르소는 상시 이동**
- **GalbotSDK는 Linux 전용** → Mac은 로봇과 통신 불가

---

## 다음 단계

**가장 중요한 단일 측정:** `set_joint_commands`의 폐루프 대역폭과 종단간 지연 (로봇 HPU + 외부 호스트, 동일 세션).

루프 레이트가 모든 공개 출처에서 미문서화입니다. **30Hz면 "빠른 소뇌"가 대뇌보다 의미 있게 빠를 여지가 없어 이 아키텍처의 전제 자체가 무너지고, 125Hz면 어드미턴스가 작동합니다.** 정책 코드 한 줄 쓰기 전에 재야 합니다.

**로봇 없이 이번 주에 닫히는 쟁점 3개** (RoboCOIN 오프라인 분석, GPU·시뮬 불필요):

1. **목 관절 운동 통계** → 헤드 프레임 논쟁 판정 (0.6m에서 1도당 ~10mm)
2. **ψ 분포 감사** → 중복성 스칼라 필수 여부 (<20° 생략 가능 / >60° 필수)
3. **헤드 카메라 내 양손 가시 비율** → 자가 캘리브레이션 채널의 생사

상세는 메인 문서 REV.3 참조.

---

## 주의

- **`[X]` 표기 수치는 2025–2026 프리프린트**로 독립 검증하지 않았습니다. 대외 자료에 인용하기 전 원문을 확인하십시오 (15절에 목록 있음).
- **`[M]` 표기만 실측치**입니다. 추정치와 문헌값을 구분해 표기했습니다.
- 15절에 **확인 못 한 항목과 벤더에 물어야 할 것**을 명시했습니다.
</content>
