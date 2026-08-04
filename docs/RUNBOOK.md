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

### 2-0. ⚠️ GalbotSDK 는 매번 `source` 가 필요합니다

**이걸 안 하면 `import galbot_sdk` 가 실패합니다.** conda 환경만으로는 부족합니다 —
SDK 가 `LD_LIBRARY_PATH` 와 `PYTHONPATH` 를 직접 세팅하기 때문입니다.

```bash
source /opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh
conda activate physical_ai

python -c "import galbot_sdk; print(galbot_sdk.__file__)"
# → /opt/galbot/galbot_sdk/linux-x86_64-gcc940/lib/python/galbot_sdk/__init__.py
```

`make probe` / `make probe-live` / `make gate1-real` 은 이 source 를 자동으로 합니다.
경로가 다르면 `make SDK_SETUP=/실제/경로/setup.sh probe`.

`.bashrc` 에 넣어두면 편합니다.

---

### 2-1. ✅ SDK 시그니처 — **2026-07-31 확정 완료**

`galbot_sdk` 는 **pybind11 확장 모듈**이라 `inspect.signature()` 가 전부 `(?)` 로
나옵니다. 실제 인자는 docstring 안에 있고, `make probe` 가 그걸 긁습니다.
**3090 에서 246/296 메서드의 시그니처를 복원했고, 어댑터는 그 기준으로 확정됐습니다.**

```bash
# 0) 맥에서도 되는 자체 검증 (파서·안전차단·FK 비교 로직)
make probe-check

# 1) 정적 조사 — 로봇 불필요. SDK 가 갱신되면 다시 돌릴 것
make probe
#    → sdk_surface_<hostname>.json / .md

# 2) 읽기 전용 실물 조회 — 로봇 전원 필요, 여전히 안 움직임
make probe-live
#    → sdk_live_<hostname>.json / .md

# 3) SDK FK 와 우리 FK 대조 — 로봇 전원 필요, 안 움직임
make fk-check
#    → robot/assets/fk_crosscheck_<hostname>.json

# 4) ⚠️ 진입점 조사 — probe-live / fk-check 가 "No constructor defined!" 로
#    막혔다면 이걸 돌리십시오
make probe-entry
```

> 🔴 **2026-07-31 미해결 — `GalbotRobot` / `GalbotMotion` 을 어떻게 얻는지 모릅니다.**
>
> ```
> TypeError: galbot_sdk.GalbotMotion: No constructor defined!
> ```
>
> 이건 pybind11 이 `py::init<>()` **없이** 바인딩한 클래스에서 나는 에러입니다.
> 즉 이 클래스들은 **직접 생성하는 게 아니라 어딘가에서 받아오는** 것입니다.
> `make probe-entry` 가 여러 획득 전략을 시도하고, `galbot_sdk/__init__.py` 원문을
> 출력합니다 — 그 파일이 `.so` 가 아니라 **Python 소스**라서 진입점이 거기 있을
> 가능성이 큽니다.
>
> 이게 풀리기 전까지 `probe-live` / `fk-check` / `gate1-real` 은 전부 막힙니다.

**확정된 시그니처가 뒤집은 것 3가지** — 셋 다 측정을 무의미하게 만들 수 있었습니다.

| | 실제 | 그대로 뒀다면 |
|---|---|---|
| `set_joint_positions` | `is_blocking=True` **기본값** | rate 램프가 명령 전송률이 아니라 **모션 완료 시간**을 쟀을 것 |
| `get_joint_positions` | `joint_groups` **필수 인자** | 무인자 호출 → `TypeError` |
| `set_joint_commands` | `time_from_start_s=10.0` **기본값** | 매 명령이 "10초에 걸쳐 도달" → 사실상 안 움직임 |

**관측 채널이 바뀌었습니다.** `JointState` 에는 **timestamp 필드가 없습니다**
(`acceleration/current/effort/position/velocity` 뿐). "로봇 타임스탬프로 상태
갱신률을 본다"는 원래 계획은 이 SDK 에서 불가능합니다. 대신 **모든 `set_*` 가
`ControlStatus` 를 반환**하므로 그게 대체 증거입니다 — SUCCESS 비율이 무너지는
지점이 실제 수락 천장입니다. `ForceData` 와 `GripperState` 에는 `timestamp_ns` 가
있으므로, F/T 쪽에서는 로봇 시계를 쓸 수 있습니다.

**`TARGET_TYPE_OVERRIDE` 는 `execute_joint_trajectory` 의 인자가 아니었습니다.**
실제 시그니처는 `(trajectory: Trajectory, is_blocking=True)` 뿐이고, `TARGET_TYPE_*` 는
`TargetConfig` → `TargetGroupTrajectory` → `SingoriXTarget` → `publish_target` 경로
소속입니다. 값이 **비트 플래그**입니다 — `OVERRIDE(10) = CLEAR(2)|APPEND(8)`,
`PROVERRIDE(14) = CLEAR|PREPENDNOW(4)|APPEND`. `SingoriXTarget` 필드 구성이
미확인이라 이 경로는 아직 못 탑니다.

> **안전:** `probe_sdk.py` 는 `set_` / `move_` / `execute_` 로 시작하는 메서드를
> **코드 레벨에서 하드 차단**합니다 (`_assert_readonly`). `make probe-check` 가
> 그 차단이 실제로 작동하는지 검증합니다. 로봇은 움직이지 않습니다.

**`probe-live` 가 한 번에 답해주는 것들** — 회사 방문 1회를 아낍니다:

| 얻는 것 | 왜 중요한가 |
|---|---|
| `get_joint_names()` 실제 순서 | RoboCOIN(21-D) vs SDK 레이아웃 불일치로 이미 한 번 데였습니다. 이름으로 인덱싱하면 그 오염이 사라집니다 |
| `get_force_sensor_data(LEFT_WRIST_FORCE)` 실값 | **손목 F/T 가 실제로 붙어 있고 값을 주는지** — §6 항목 3 |
| `get_gripper_state()` / `get_suction_cup_state()` | **엔드이펙터 좌우 구성** 단서 — §6 항목 1 |
| `get_sensor_extrinsic(sensor_id, reference_frame)` | **헤드 카메라 extrinsic** — §7 항목 4가 벤더 문의에서 자체 확인으로. `SensorType` 에 `HEAD_CAMERA / HEAD_DEPTH_CAMERA / HEAD_LEFT_CAMERA / HEAD_RIGHT_CAMERA` 가 모두 있어 **스테레오 + depth** 가 API 레벨에서 확인됩니다 |
| `GalbotMotion.forward_kinematics` | **SDK FK 와 우리 FK 대조** (`make fk-check`) |

**📩 알려주실 것**: `sdk_live_*.md` 와 `fk_crosscheck_*.json`.

> **`make fk-check` 가 중요한 이유:** `test_kinematics.py` 의 T1~T7 은 전부 URDF
> **자기일관성** 검정입니다. URDF 가 실기체와 다르면 7/7 통과해도 아무것도 보증하지
> 않습니다. 그런데 그 FK 로 계산한 ψ 와 `T_rel` 이 데이터셋에 **영구히 구워지고**,
> F/T 와 똑같이 소급 수정이 불가능합니다.
>
> `forward_kinematics(target_frame, reference_frame, joint_state=...)` 가 `joint_state` 를
> **인자로** 받는 덕분에, 로봇을 그 자세로 움직이지 않고 임의 관절값으로 대조할 수 있습니다.
> 진단은 **두 종류의 상대변환**으로 tip / root / 실제 URDF 오류를 분리합니다:
>
> | body-rel | spat-rel | 뜻 |
> |---|---|---|
> | ✅ | ✅ | 프레임까지 동일 |
> | ✅ | ❌ | root(기준) 프레임만 다름 — 치명적 아님 |
> | ❌ | ✅ | **tip 프레임만 다름** — 상수 오차라 학습이 흡수해 조용히 틀림 |
> | ❌ | ❌ | 🔴 **URDF 자체가 실기체와 다름** — 변환 중단 |

---

### 2-2. 🔴 GATE-1 — SDK 폐루프 대역폭 **[최우선 측정]**

**이 프로젝트에서 가장 중요한 단일 측정입니다.** 결과에 따라 아키텍처 전제가 바뀝니다.

```bash
# 1) 드라이런 — 로봇 불필요. 스크립트 로직만 검증
make gate1

# 2) 실제 측정 (3090, 다른 부하 없을 때!)
make gate1-real HOST=external

# 3) 로봇 온보드에서도
make gate1-real HOST=onboard
```

> ⚠️ **측정 중 그 머신에서 다른 워크로드를 돌리지 마십시오.** jitter를 재는 것이라 p99/p99.9가 오염됩니다.
> ⚠️ **안전:** 단일 관절(`left_arm_joint4`)만 ±2°로 움직입니다. 팔 주변을 비우고 e-stop을 손 닿는 곳에.

**2026-07-31 개정 — 명령 경로가 둘이라는 게 드러났습니다.**

SDK 표면을 실제로 보니 `TARGET_TYPE_OVERRIDE` / `PROVERRIDE` / `PREPENDNOW` 라는
**궤적 큐 덮어쓰기** 의미론이 따로 있었습니다. 그래서 측정을 3부로 나눴습니다.

| | 경로 | 재는 것 |
|---|---|---|
| **PART A** | `set_joint_commands` 고속 반복 | 명령 rate 천장 + jitter |
| **PART B** | 동일 | 위치 추종 −3dB 대역폭 |
| **PART C** | `execute_joint_trajectory` + `TARGET_TYPE_OVERRIDE` | **재계획 지연(replan latency)** |

**왜 PART C 가 추가됐는가.** 기존 판정은 "Python이 100Hz를 못 내면 분리 전제 붕괴"였는데,
그건 경로 A만 가정한 논리입니다. 경로 B가 실제로 동작하면 **상위가 5Hz로 보내도 온보드가
보간**하므로 상위 rate 천장이 낮다는 사실 자체는 전제를 죽이지 않습니다. 대신 그 경로의
진짜 병목은 **덮어쓰기가 실제 운동에 반영되기까지의 지연**입니다. 이건 rate를 올려도
해결되지 않고, 이 값이 크면 청크 경계마다 이미 지난 상황에 반응하게 됩니다.

**이 결과가 바꾸는 것**

경로 B가 살아 있으면 **그쪽이 주 판정 기준**이고, 경로 A 천장은 보조 지표로 내려갑니다.

| 재계획 지연 p95 (경로 B) | 결론 |
|---|---|
| **≤50 ms** | ✅ 5Hz chunk 스트리밍 성립. 상위 rate가 낮아도 무방 |
| 50–200 ms | 경계. 청크 주기를 그보다 길게 잡아야 하고 접촉 반응이 그만큼 늦음 |
| **>200 ms** | 🔴 청크 경계마다 이미 지난 상황에 반응 |

경로 B를 못 쓸 때만 기존 기준을 적용합니다.

| 명령 rate 천장 (경로 A) | 결론 |
|---|---|
| **≥100 Hz** | admittance 작동. 2모델 분리 유지 가능 |
| 50–100 Hz | 경계. 접촉 작업 범위를 서면으로 한정 |
| **<50 Hz** | 🔴 **분리 전제 붕괴.** 단일 모델 + 벤더 WBC로 전환 |

> ⚠️ 경로 B 실패가 **"미지원"인지 "시그니처 미확인"인지 구분해야 합니다.** 후자면
> `make probe` 로 확정한 뒤 재측정하십시오 — **판정이 뒤집힐 수 있습니다.**
> 스크립트가 실패 시 해당 메서드의 docstring 을 그대로 출력하므로 그걸 보내주시면 됩니다.

**📩 알려주실 것**
```
1. 스크립트가 자동 출력하는 "판정" 블록 전체
2. gate1_results_*.json 파일
3. PART C 가 건너뛰어졌다면 그때 출력된 docstring 진단 블록
4. 판정 블록에 "상태 타임스탬프가 호스트 시계" 경고가 떴는지 여부
```

> **왜 4번을 묻는가:** 로봇이 자체 타임스탬프를 안 주면 호스트 시계로 대체하는데,
> 그러면 `state_dt_*` / `unique_state_frac` 은 **로봇 응답이 아니라 우리 루프**를
> 재는 값이 됩니다. 그걸 모르고 보면 stale 상태를 건강한 것으로 오독합니다.

---

### 2-3. 🔴 컨버터 포크 검증 — **데이터 수집 전 필수**

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

### 2-4. 학습 (게이트 통과 후)

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

**2026-07-31 갱신 — `make probe-live` 로 상당 부분이 자동 확인됩니다.**
아래 표의 "probe" 열이 ✅ 면 눈으로 안 봐도 됩니다.

| # | 확인할 것 | probe | 왜 |
|---|---|---|---|
| **1** | **엔드이펙터 좌우 구성** — 양쪽 다 그리퍼인가, 한쪽이 석션컵인가 | 🟡 단서 | `get_gripper_state` / `get_suction_cup_state` / `get_dexterous_hand_state` 응답으로 좁혀지지만, **API 존재는 장착을 뜻하지 않습니다.** 최종은 육안 |
| **2** | **텔레옵 리그가 bilateral 인가** — 로봇이 받는 힘을 조작자가 느끼는가 | ❌ | unilateral이면 **wrench 채널에 학습 가능한 구조가 없습니다.** 힘 관련 계획 전체의 전제. 육안·조작 확인 필수 |
| **3** | 손목 F/T 실물 장착 + 실제 값 | ✅ | `get_force_sensor_data(LEFT/RIGHT_WRIST_FORCE)` 가 값을 주는지로 확정. 전부 0이면 미연결 의심 |
| **4** | 헤드 카메라가 스테레오 RGB인지 depth인지 | 🟡 단서 | `SensorType` / `GalbotOneFoxtrotSensor` enum 멤버 이름으로 좁혀집니다 |

> **2번이 남는 이유:** SDK는 로봇 쪽만 보여줍니다. 조작자가 힘을 느끼는지는
> 리그 하드웨어 문제라 API로 알 수 없습니다.

---

## 7. 벤더 문의 (영업/FAE)

**2026-07-31 갱신 — SDK 표면 확인으로 일부가 자체 해결되거나 질문이 날카로워졌습니다.**

1. 🔴 **`TARGET_DATA_JOINT_EFFORT` / `TARGET_DATA_FRAME_WRENCH` 가 G1 팔에서 실제로 지원되는가**
   → SDK enum 에 **존재는 합니다.** 그런데 에러 코드에 `UNSUPPORTED_FUNCRION` 이 있으므로
   **선언 ≠ 지원**입니다. 지원된다면 **컴플라이언스 지향 Model 2가 되살아납니다.**
   기존의 `*_PVT_BYPASS_CTRL` 질문을 대체하는, **가장 가치 높은 질문**입니다.
   *(자체 확인도 가능하지만 실기체에 effort를 넣는 실험이라 벤더 답을 먼저 받는 편이 안전)*
2. 🔴 **`TARGET_TYPE_OVERRIDE` / `PROVERRIDE` / `PREPENDNOW` 의 정확한 의미론**
   → 큐를 어디서부터 자르는가, 현재 실행 중인 점을 어떻게 이어붙이는가.
   **action chunk 스트리밍 설계가 여기 직접 걸립니다.** GATE-1 PART C가 실측하지만
   의미론은 문서로 확인해야 합니다
3. **F/T 데이터시트** — 범위/분해능/노이즈/샘플레이트/마운팅 프레임, **중력·페이로드 보상 여부**
   → 값은 `probe-live` 로 얻지만 **보상 여부는 값만 봐선 모릅니다.** 이건 남습니다
4. ~~**헤드 카메라 extrinsic**~~ → ✅ **`get_sensor_extrinsic()` 로 자체 확인.** 문의 불필요
5. **세대 확인** → 🟡 클래스명이 `GalbotOne**Foxtrot**Sensor` 이므로 **Foxtrot 세대**로
   보입니다. 관절 수는 `get_joint_names()` 로 확정. 정식 모델명만 확인하면 됩니다
6. **안전 인증** — ISO 10218 / 13849 PL / TS 15066 / TÜV / CE

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

| 순위 | 작업 | 기계 | 로봇 | 이유 |
|---|---|---|---|---|
| ~~0~~ | ~~`make probe`~~ | 3090 | ❌ | ✅ **2026-07-31 완료.** 246/296 시그니처 확정 |
| **1** | `make probe-live` + `make fk-check` | 3090 + 로봇 | 전원만 | 손목 F/T·관절 순서·카메라 extrinsic·엔드이펙터 + **FK 외부 검증**을 **안 움직이고** 한 번에 |
| **2** | GATE-1 (PART A/B/C) | 3090 + 로봇 | ⚠️ 움직임 | 아키텍처 전제를 죽일 수 있음 |
| **3** | 컨버터 포크 검증 | 3090 + 로봇 | 수집 | **수집 시작하면 소급 불가** |
| **4** | 텔레옵 리그 bilateral 확인 | 실기체 | 육안 | API로 알 수 없는 유일한 항목 |
| **5** | E0 | 1660 | ❌ | 가설 자체를 죽일 수 있음 |
| **6** | RoboCOIN 3종 | Mac/1660 | ❌ | 헤드 프레임·ψ 결정 |

**0번을 먼저 하십시오.** 로봇도 필요 없고 몇 분이면 끝나는데, 그 결과가 GATE-1 어댑터를
확정해서 2번의 실패 위험을 없앱니다. 지금은 메서드 **이름만** 알고 **인자는 모르는** 상태라,
0번 없이 2번을 돌리면 패턴 사다리에 걸려 시간만 씁니다.

**2와 5는 서로 독립이라 병렬 가능합니다.** 회사에서 0·1·2·3·4를 하고, 집에서 5·6을 돌리면
일주일 안에 큰 불확실성 대부분이 닫힙니다.
