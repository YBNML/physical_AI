# 실행 가이드 (RUNBOOK)

**어느 기계에서 무엇을 돌리고, 결과를 어떻게 보고할지.**

용어가 생소하면 [GLOSSARY.md](GLOSSARY.md), 전체 계획은 [PLAN.md](PLAN.md).

---

## 0. 기계별 역할 한눈에

| 기계 | 위치 | 할 수 있는 것 | 할 수 없는 것 |
|---|---|---|---|
| **Mac mini M4** | 🏠 집 | 기구학, 인터페이스, RoboCOIN 분석, MuJoCo | ❌ CUDA ❌ **GalbotSDK(로봇 통신)** ❌ 학습 |
| **1660 Super 6GB** | 🏠 집 | **E0**, eval, 데이터 전처리 | ❌ Isaac Lab(16GB) ❌ π0 파인튜닝(24GB) |
| **RTX 3090 24GB** | 🏢 회사 | 학습, RoboTwin, **SDK 호스트**, GATE-1 | ⚠️ 다른 업무와 공유 |
| **개인 AGX Orin** | 🏢 회사 | **TensorRT 엔진 빌드**, 배포 테스트, SDK 호스트(aarch64) | ❌ 학습 |
| **Galbot G1** | 🏢 회사 | 실기체 전부 | — |

> **왜 이렇게 나뉘는가**
> - GalbotSDK는 `linux-x86_64` / `linux-aarch64` 바이너리만 출하합니다 → **Mac은 로봇과 통신 불가**
> - TensorRT 엔진은 아키텍처 종속(Orin sm_87 ≠ 3090 sm_86) → **Orin 엔진은 Orin에서 빌드**
> - 회사 시간이 희소 자원 → **로봇 없이는 불가능한 것에만** 쓰기

---

## 1. 최초 1회 — 어느 기계든

```bash
git clone git@github.com:YBNML/physical_AI.git
cd physical_AI
bash setup.sh                 # conda 환경 생성 + 자체 검증
# 설치 없이 진단만:
bash setup.sh --check
```

설치 후 확인:

```bash
make test                     # 기구학 검증 6종 — 6/6 통과해야 함
```

**6/6이 안 나오면 그 기계에서는 아무것도 진행하지 마시고 출력을 알려주십시오.**

---

## 2. 🏢 RTX 3090 (회사) — clone 후 바로

### 2-1. 🔴 GATE-1 — SDK 폐루프 대역폭 **[최우선]**

**이 프로젝트에서 가장 중요한 단일 측정입니다.** 결과에 따라 아키텍처 전제가 바뀝니다.

```bash
# 1) 먼저 드라이런으로 스크립트가 도는지만 확인 (로봇 불필요)
python tools/measure_loop_rate.py --dry-run

# 2) SDK 어댑터를 실물에 맞게 수정
#    tools/measure_loop_rate.py 의 G1Adapter 클래스만 고치면 됩니다.
#    "⚠️ 확인 필요" 로 표시된 3곳:
#      - from galbot_sdk import Robot   (실제 모듈/클래스명)
#      - Robot() 생성자 인자 (IP 등)
#      - robot.get_joint_state / set_joint_commands 시그니처

# 3) 실제 측정 — 3090에서 (다른 부하 없을 때!)
python tools/measure_loop_rate.py --host external --out results_3090.json

# 4) 로봇 온보드에서도 (SSH로 들어가서)
python tools/measure_loop_rate.py --host onboard --out results_onboard.json
```

> ⚠️ **측정 중 그 머신에서 다른 워크로드를 돌리지 마십시오.** jitter를 재는 것이라 p99/p99.9가 오염됩니다.
> ⚠️ **안전:** 단일 관절(`left_arm_joint4`)만 ±2°로 움직입니다. 팔 주변을 비우고 e-stop을 손 닿는 곳에.

**📩 알려주실 것**
```
1. 스크립트가 자동 출력하는 "판정" 블록 전체
2. results_3090.json / results_onboard.json 파일
3. G1Adapter 를 어떻게 고쳤는지 (실제 API 이름)
```

**이 결과가 바꾸는 것**

| 명령 rate 천장 | 결론 |
|---|---|
| **≥100 Hz** | admittance 작동. 2모델 분리 유지 가능 |
| 50–100 Hz | 경계. 접촉 작업 범위를 서면으로 한정 |
| **<50 Hz** | 🔴 **분리 전제 붕괴.** 단일 모델 + 벤더 WBC로 전환 |

---

### 2-2. 🔴 컨버터 포크 검증 — **데이터 수집 전 필수**

F/T는 **기록된 에피소드에 소급 추가가 불가능합니다.** 에피소드 1개 받기 전에 끝나야 합니다.

```bash
# 1) 드라이런 (MCAP 불필요)
python tools/mcap2lerobot_fork.py --dry-run

# 2) 실제 MCAP 안의 토픽 이름 확인 — F/T 토픽명을 모릅니다
python tools/mcap2lerobot_fork.py --list-topics <어떤_mcap_파일>

# 3) 토픽명을 코드에 반영한 뒤 실제 변환
python tools/mcap2lerobot_fork.py --input <mcap_dir> --output <out_dir>
```

**📩 알려주실 것**
```
1. --list-topics 출력 전체  ← F/T 토픽 이름을 여기서 확정합니다
2. 변환 후 자동 출력되는 "검증 리포트"
   - F/T 채널이 실제로 채워졌는가 (전부 0이면 센서 미연결 의심)
   - wrench 크기 분포 (정지 ~0.5N / 운동 중 ~2N 잔차가 정상)
   - psi 계산 성공률
```

---

### 2-3. 학습 (게이트 통과 후)

```bash
# RoboTwin 2.0 설치 후 — 7-DoF embodiment 가 있는지 확인 필요
python tools/check_robotwin_embodiments.py    # (없으면 수동 확인)
```

**📩 알려주실 것**: RoboTwin 임베디먼트 5개 중 **7-DoF가 있는지**. 없으면 ψ 실험은 시뮬에서 못 하고 실기체+RoboCOIN으로만 해야 합니다.

---

## 3. 🏠 1660 Super (집) — E0

### 3-1. 🔴 GATE-2 — E0

**로봇도 3090도 안 쓰고 프로젝트의 핵심 가설을 반증할 수 있는 실험입니다.**

```bash
# 1) 파이프라인 검증 (LIBERO 불필요)
make e0-smoke

# 2) LIBERO 설치 후 실제 실행
bash experiments/e0/run_all.sh
```

**📩 알려주실 것**
```
1. run_all.sh 마지막에 출력되는 판정 블록
2. C0 / C6 / C8 각각의 성공률과 95% CI
3. 총 소요시간
```

**이 결과가 바꾸는 것**

| 조건 | 결론 |
|---|---|
| `C6 − C0 ≤ +5` | 🔴 **가설 사망.** 천장이 얇으면 배포 시엔 monolith 아래 |
| `C8 ≥ C6 − 5` | 🔴 **학습 Model 2 사망.** Model 1 + closed-form IK로 |
| `C6−C0 ≥ +10` 이고 `C6−C8 ≥ +5` | ✅ 진행 |

---

### 3-2. RoboCOIN 분석 (Mac에서도 가능)

```bash
# 1) 스키마부터 확인 — 컬럼명이 다를 수 있습니다
python tools/robocoin_analysis.py --data <robocoin_path> --inspect

# 2) 컬럼명이 다르면 COLUMNS 딕셔너리 수정 후
python tools/robocoin_analysis.py --data <robocoin_path> --all
```

**📩 알려주실 것**
```
1. --inspect 출력 (컬럼명과 23차원 레이아웃)
2. --all 의 A1/A2/A3 판정 3개
```

**이 결과가 바꾸는 것**

| 분석 | 임계값 | 결론 |
|---|---|---|
| **A1 목 운동** | 2.0s 청크 p95 **>6°** | 🔴 헤드 프레임 폐기, `torso_base_link`로 직접 예측 |
| **A2 ψ 분포** | bin 내 산포 **>60°** | ψ 스칼라가 인터페이스에 필수 |
| **A3 양손 가시율** | 양손 동시 **<30%** | 자가 캘리브레이션 채널 사망 → 인터페이스 변환 필수 |

---

## 4. 🏠 Mac mini (집)

```bash
make test        # 기구학 검증 6종
make analysis    # RoboCOIN (데이터 있으면)
```

**Mac에서 하는 일:** 기구학·인터페이스 설계, RoboCOIN 오프라인 분석, MuJoCo 검증, 문서.
**Mac에서 못 하는 일:** 로봇 통신, 학습, CUDA 시뮬.

---

## 5. 🏢 AGX Orin (회사) — 나중에

배포 단계에서 씁니다. 지금 할 것:

```bash
# 세대/메모리 확인
cat /etc/nv_tegra_release
free -h
```

**📩 알려주실 것**
```
1. 개인 Orin 의 메모리 (32GB dev kit / 64GB 모듈?)
2. G1 온보드 컴퓨트가 무엇인지 (AGX Orin 64GB 맞는지)
   → 둘이 같으면 개인 Orin 이 완벽한 배포 트윈이 됩니다
```

---

## 6. 🏢 실기체 확인 항목 (회사 갈 때)

코드 없이 눈으로 확인하는 것들입니다.

| # | 확인할 것 | 왜 |
|---|---|---|
| **1** | **엔드이펙터 좌우 구성** — 양쪽 다 그리퍼인가, 한쪽이 석션컵인가 | 출처 3개가 답이 3개. **한쪽이 석션이면 양팔 협응 전제가 무너집니다** |
| **2** | **텔레옵 리그가 bilateral 인가** — 로봇이 받는 힘을 조작자가 느끼는가 | unilateral이면 **wrench 채널에 학습 가능한 구조가 없습니다.** 힘 관련 계획 전체의 전제 |
| **3** | 손목 F/T 실물 장착 확인 | 스펙엔 있으나 유닛별 확인 필요 |
| **4** | 헤드 카메라가 스테레오 RGB인지 depth인지 | 매뉴얼과 논문이 충돌 |

---

## 7. 벤더 문의 (영업/FAE)

공개 소스로는 불가능한 것들입니다.

1. 🔴 **`*_PVT_BYPASS_CTRL` semantics** — "bypass"가 effort나 Kp/Kd를 전달하는가
   → **전달하면 컴플라이언스 지향 Model 2가 되살아납니다. 가장 가치 높은 질문**
2. **F/T 데이터시트** — 범위/분해능/노이즈/샘플레이트/마운팅 프레임, **중력·페이로드 보상 여부**
3. **헤드 카메라 extrinsic** (`head_link2` 기준) + 모델/FOV/스테레오 baseline
4. **세대 확인** (Charlie / Foxtrot / Golf / `G1_V2.2B`) — 세대 간 관절 수가 다름
5. **안전 인증** — ISO 10218 / 13849 PL / TS 15066 / TÜV / CE

---

## 8. 보고 템플릿

결과를 알려주실 때 이 형식이면 제가 바로 판정에 반영할 수 있습니다.

```
[기계]  3090 / 1660 / Mac / Orin / 실기체
[작업]  GATE-1 / GATE-2 / RoboCOIN / 컨버터 / 실물확인
[결과]  (스크립트가 출력한 판정 블록을 그대로 붙여넣기)
[막힌 곳] (에러 메시지 전문)
[고친 것] (어댑터/컬럼명 등을 수정했다면 무엇을)
```

**에러가 나면 전문을 그대로 주십시오.** 특히 `⚠️ 확인 필요` 로 표시된 부분에서 나는 에러는 제가 검증할 수 없었던 지점이라 실물 정보가 필요합니다.

---

## 9. 우선순위 — 시간이 없다면

| 순위 | 작업 | 기계 | 이유 |
|---|---|---|---|
| **1** | GATE-1 루프 레이트 | 3090 + 로봇 | 아키텍처 전제를 죽일 수 있음 |
| **2** | 컨버터 포크 검증 | 3090 + 로봇 | **수집 시작하면 소급 불가** |
| **3** | 엔드이펙터 좌우 확인 | 실기체 | 양팔 전제 |
| **4** | E0 | 1660 | 가설 자체를 죽일 수 있음 |
| **5** | RoboCOIN 3종 | Mac/1660 | 헤드 프레임·ψ 결정 |

**1과 4는 서로 독립이라 병렬 가능합니다.** 회사에서 1·2·3을 하고, 집에서 4·5를 돌리면 일주일 안에 큰 불확실성 대부분이 닫힙니다.
