# E0 — GATE-2: 아키텍처 천장 측정

**이 프로젝트에서 가장 중요한 단일 실험이자, 가능한 가장 값싼 반증.**

E0 는 "2모델 분리가 단일 모델을 이기는가" 를 묻지 않는다. 그것보다 훨씬 값싸고
훨씬 강한 질문을 던진다:

> Model 1 이 **완벽하다면** 저수준 정책은 얼마나 잘할 수 있는가?

hindsight oracle — 데모의 미래를 그냥 들여다본 정답 EE 포즈 — 을 인터페이스로
먹여서 **아키텍처의 천장**을 잰다. 앞으로 만들 어떤 Model 1 도 정답을 이길 수
없으므로, 이 천장이 얇으면 실제로 배포될 분리는 monolith 아래로 **확실히** 간다.

천장을 재는 데는 Model 1 을 한 줄도 학습시키지 않아도 된다. 그래서 값싸다.

---

## 1. 3개 arm

| arm | 이름 | 입력 | 학습 파라미터 |
|---|---|---|---|
| **C0** | MONO | RGB(2대) + proprio | 26,180,135 |
| **C6** | SPLIT-ORACLE | RGB(2대) + proprio + **정답 목표 EE 포즈** | 26,180,135 |
| **C8** | ORACLE-IK | **정답 목표 EE 포즈** → OSC_POSE 직결 | **0** |

**C0 와 C6 의 파라미터 수는 정확히 같다.** C0 에도 `goal_proj` 가 있고 0 벡터를
먹어 learned null token 이 된다. 시퀀스 길이(130 토큰)도 동일하다.
`models.assert_matched()` 가 비트 단위로 검사하며, 이게 깨지면 E0 의 결론은
전부 무효다 — C6−C0 격차를 "용량이 더 커서" 로 설명할 수 있게 되기 때문이다.

C8 은 모델이 아니다. `dataset.oracle_to_osc_action()` 이 전부다.

### 세 arm 이 각각 죽이는 것

- **C6 ≤ C0** → 가설(H1)이 죽는다. 정답으로도 못 이기면 근사로는 더 못 이긴다.
- **C8 ≈ C6** → 학습된 Model 2 가 죽는다. 목표 포즈만 있으면 closed-form 으로 충분.
- 둘 다 아니면 → 분리가 이기고, 그 이득이 **학습된 소뇌**에서 온다. 진행.

---

## 2. 판정 규칙 — 실행 전에 고정. 결과를 보고 바꾸지 말 것

`evaluate.py` 에 하드코딩돼 있고 자동 출력된다 (`T_H1_DEATH` / `T_M2_DEATH` / `T_PROCEED`).

| 조건 | 판정 | 다음 행동 |
|---|---|---|
| `C6 − C0 ≤ +5` **또는** 95% CI 가 0 포함 | **H1 사망** | 2모델 분리를 접고 단일 모델 + 벤더 WBC 로 |
| `C8 ≥ C6 − 5` | **학습 Model 2 사망** | Model 1 + closed-form IK 로 |
| `C6−C0 ≥ +10` **AND** `C6−C8 ≥ +5` | **진행** | E1 로 |
| `+5 < C6−C0 < +10` | **재실행** | LIBERO-Long / CALVIN ABC→D 에서 다시 |

우선순위는 치명적인 것부터다 (H1 사망 → Model 2 사망 → 진행 → 재실행).
두 사망 규칙은 동시에 발화할 수 있고, 그 경우 리포트에 둘 다 `[x]` 로 찍힌다.

### 검정력 — 왜 seed 3개가 **필수**인가

two-proportion 양측 검정, α=0.05, power=0.80, p=0.5(최대분산 = 보수적):

```
MDE = (z₀.₉₇₅ + z₀.₈₀) · √(2·p(1−p)/n)
```

| n (arm 당) | MDE | 의미 |
|---|---|---|
| 500 (seed 1개) | **8.86 pt** | ±5pt 규칙을 **판정할 수 없다** |
| 1500 (seed 3개) | **5.11 pt** | ±5pt 규칙이 겨우 의미를 가진다 |

`evaluate.py --stats-test` 가 이 두 값을 문서 인용값(8.9 / 5.1)과 대조해 assert 한다.

> ⚠️ **pooled CI 는 anti-conservative 하다.** seed 3개는 같은 500개 held-out
> 에피소드를 다시 푼 것이라 1500 trial 이 독립이 아니다(에피소드 난이도가 공유됨).
> 그래서 리포트가 **seed 별 성공률과 산포**를 함께 출력한다. seed 간 산포가 pooled
> CI 폭보다 크면 pooled p-value 를 믿지 말고 seed 를 더 돌려야 한다.
> C8 처럼 결정론적인 arm 은 seed 간 결과가 동일한지 검사해서, 동일하면
> "실효 n 은 pooled n 이 아니다" 라고 경고한다.

---

## 3. 실험 설계

**데이터:** LIBERO-Spatial + LIBERO-Object = 20 태스크 × 50 데모 = 1000 데모.

**분할: 태스크당 train 25 / eval 25 (기본값).**
40/10 이 아니라 25/25 인 이유는 순전히 검정력이다. 태스크당 25개를 eval 로 빼면
20 × 25 = **seed 당 정확히 500 rollout**, 3 seed pool 로 n=1500 → MDE 5.1pt.
40/10 으로 하면 seed 당 200, pool 600 → MDE 8.1pt 로 ±5pt 규칙이 죽는다.
학습 데이터가 줄면 세 arm 이 **똑같이** 손해를 보므로 우리가 재려는 *대비*는
그대로 유지된다. `--train-demos 40` 으로 바꿀 수는 있다.

**oracle 정의** (`dataset.hindsight_goal`) — 정확히 numpy 슬라이스 한 줄이다:

```python
idx = np.minimum(np.arange(T) + horizon, T - 1)   # K = 16 (0.8s @ 20Hz)
dp   = ee_pos[idx] - ee_pos                       # (3,) 병진 델타
dR6  = R_to_6d(R[idx] · Rᵀ)                       # (6,) 6D 연속 회전
grip = grip_width[idx]                            # (1,) [m] 연속
```

**절대 포즈가 아니라 델타다.** 실측 EE 절대 69.0% vs EE 델타 89.6% [M].
델타는 hand-eye 캘리브 편향(~2.1mm/3.2°)을 양 끝점이 공유해 상쇄하고,
물체 위치가 바뀌었을 때 절대 좌표가 OOD 가 되는 문제를 피한다.

**rollout 시점의 oracle 재표현** (`Episode.oracle_at`) — 여기가 조용히 틀리기 쉬운 곳이다.
rollout 중 로봇은 데모 궤적을 벗어나므로 데모의 델타를 그대로 먹이면 입력 분포가
어긋난다(로봇이 5cm 뒤처져 있는데 "앞으로 3cm" 라고 말하는 셈). 올바른 oracle 은
**t+K 의 절대 목표를 지금 실제 손 위치 기준 델타로 다시 표현**하는 것이다.
실제 Model 1 도 현재 상태를 보고 델타를 내므로 배포 시나리오와도 일치한다.
**C6 와 C8 은 정확히 같은 이 벡터를 받는다** — 그래야 둘의 차이가 오직
"학습된 소뇌 vs closed-form 컨트롤러" 가 된다.

**모델:** ACT 계열. 공유 ResNet 백본(카메라 2대) → 130 토큰
(2 × 8×8 vision + proprio + goal) → transformer encoder 6층 / decoder 6층,
d=384, ffn=1536, head=6 → K=16 학습된 query 로 chunk 를 한 번에 디코드.
autoregressive 가 아닌 이유는 20Hz 제어에서 순차 디코딩이 latency 예산을
그냥 태우기 때문이다.

**공정성 장치:**
- proprio 에 현재 EE 포즈를 넣는다 (**C0 에도**). 안 넣으면 C6−C0 가 "oracle 의
  가치" 가 아니라 "EE 포즈 관측의 가치" 로 오염된다. C0 도 자기 손이 어디 있는지는
  안다. 모르는 건 **미래** 뿐이다.
- C0/C6 는 옵티마이저·스텝 수·LR 스케줄·seed 가 전부 같다. arm 이 바꾸는 것은
  `use_goal` 플래그 하나뿐이다.
- C8 은 매 스텝 재계산한다 (closed-form 이라 비용 0). C8 을 최대한 강하게 두는
  것은 "학습 Model 2 사망" 쪽으로 보수적이다 — **우리 가설에 불리한 방향으로
  기울이는 것이 옳다.**

---

## 4. 실행법

### 4.1 먼저 `--smoke` — LIBERO / torch / GPU 없이 전체 배선 검증

```bash
bash run_all.sh --smoke
```

합성 데이터로 **데이터 → 학습 → rollout → 통계 → 판정** 전 경로를 몇 분 안에
돌린다. 3090 에서 clone 하자마자 이걸 먼저 돌려서 코드가 맞는지 확인할 것.
torch 가 없으면 numpy 전용 폴백 정책으로 대신 돈다 (성능 수치는 무의미하지만
배선은 전부 검증된다).

개별 self-test 도 따로 돌릴 수 있다:

```bash
python dataset.py  --smoke        # oracle 델타↔절대 왕복, drift 보정, quat 규약
python models.py   --smoke        # C0/C6 파라미터 일치, C0 의 goal 차단 확인
python evaluate.py --stats-test   # MDE 8.9/5.1 대조, 판정 규칙 4갈래, 합성 플랜트 정합성
```

`--smoke` 가 검증하는 것 (전부 assert 로 강제):

| 검사 | 왜 |
|---|---|
| oracle 델타 → 절대 포즈 왕복 오차 < 1e-5 | 슬라이스 방향이 뒤집히면 조용히 틀린다 |
| 로봇이 5cm 뒤처지면 델타도 +5cm | rollout 재표현이 학습 분포와 맞는지 |
| C0 파라미터 == C6 파라미터 (비트 단위) | 깨지면 E0 결론이 전부 무효 |
| C0 출력이 goal 입력에 불변 | 대조군 오염 방지 |
| MDE(500)=8.86, MDE(1500)=5.11 | 문서 인용값과 코드가 갈리지 않게 |
| 판정 규칙 4갈래가 전부 발화 | 규칙 로직 자체의 검증 |
| 데모 action 을 재생하면 과제가 풀림 | "완벽히 모방해도 실패하는" 환경이면 smoke 가 무의미 |

### 4.2 본 실험

```bash
export LIBERO_DATA=/path/to/libero/datasets
bash run_all.sh --data $LIBERO_DATA                  # 3090 기본
bash run_all.sh --data $LIBERO_DATA --profile 1660   # 6GB 카드
bash run_all.sh --report-only                        # 판정만 재출력
```

중단 후 재시작하면 이미 만들어진 체크포인트/결과를 건너뛴다
(3090 은 다른 업무와 공유하므로 중단이 정상 상황이다).

**LIBERO 설치** (Linux + NVIDIA 필수. Mac 에서는 offscreen 렌더가 사실상 불가):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -r requirements.txt && pip install -e .
pip install robosuite==1.4.1 h5py
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
python benchmark_scripts/download_libero_datasets.py --datasets libero_object
```

**첫 실행 전 반드시:**

```bash
python dataset.py --inspect --data $LIBERO_DATA   # HDF5 실제 키 구조 확인
```

---

## 5. 하드웨어별 배치 크기 / 예상 시간

### 연산량 (측정 기반)

| 항목 | 값 | 출처 |
|---|---|---|
| 파라미터 | 26,180,135 (C0 = C6) | [M] `models.py --params` |
| 시퀀스 길이 | 130 토큰 | [M] |
| forward 연산 | 1.26 GMACs = 2.5 GFLOP / sample | [M] hook 계수 |
| 학습 스텝 (fwd+bwd ≈ 3×) | **7.6 GFLOP / sample** | [M] |
| M4 CPU 실측 | 31 sample/s (batch 8) ≈ 236 GFLOPS | [M] |

M4 CPU 실측이 FLOP 모델과 일치하므로(31 × 7.6 = 236 GFLOPS, M4 CPU 의 합리적
실효 처리량) 아래 GPU 추정치는 이 모델을 그대로 확장한 것이다.

### 배치 / 시간

**effective batch 는 두 프로파일 모두 64 로 고정한다.** 배치가 달라지면
C0/C6 비교가 아니라 하드웨어 비교가 돼 버린다.

| | **RTX 3090 24GB** | **GTX 1660 Super 6GB** |
|---|---|---|
| 배치 × accum | **64 × 1** | **32 × 2** (OOM 시 16 × 4) |
| AMP dtype | bf16 (Ampere) | fp16 (Turing, tensor core 없음) |
| VRAM 사용 | ~2–3 GB [I] | ~1.5–2 GB [I] |
| 학습 처리량 | 15–30 it/s [I] | 3–5 it/s [I] |
| **학습 1 run** (60k step) | **35–70분** | **3.5–5.5시간** |
| 학습 6 run (2 arm × 3 seed) | **3.5–7시간** | **21–33시간** |
| **평가 1 run** (500 rollout) | **15–30분** | **25–40분** |
| 평가 9 run (3 arm × 3 seed) | **2.5–4.5시간** | **4–6시간** |
| **E0 전체** | **≈ 6–12시간** | **≈ 25–39시간** |

모두 `[I]` 추정이다 — 근거는 위 FLOP 표와 M4 실측 앵커. 평가 시간은 MuJoCo
offscreen 렌더가 지배하며(3090 기준 ~100–200 env step/s [I]), 성공한
에피소드는 조기 종료하므로 성공률이 높을수록 빨라진다.

**권장:** 1660 Super 는 `--smoke` 와 **seed 1개 sanity run**
(`--seeds 0`, 학습 2 run + 평가 3 run ≈ **8–13시간**) 까지만 쓰고,
3 seed 본 실험은 3090 에서 돌릴 것. 1660 로 전체를 돌리면 2–3일 연속 점유가
필요하고, 그래도 seed 1개로는 ±5pt 규칙을 판정할 수 없다(MDE 8.9pt).

### RAM / 디스크

| 항목 | 값 |
|---|---|
| 학습 split 상주 (500 데모 × ~150 step × 2 cam × 128²×3, uint8) | **~7 GB** |
| 평가 split 상주 | **~수십 MB** (`load_rgb=False` — rollout RGB 는 env 에서 온다) |
| 체크포인트 1개 | ~105 MB (26.18M × 4B, weights 만 — optimizer state 는 저장 안 함) |
| 체크포인트 전체 (6개) | ~630 MB |

DataLoader worker 는 **fork 에서만** 쓴다. spawn 이면 7GB 가 worker 마다 복사되므로
`train.py` 가 자동으로 `num_workers=0` 으로 떨어뜨리고 경고한다.

---

## 6. 파일 구조

| 파일 | 역할 |
|---|---|
| `dataset.py` | LIBERO 로딩 어댑터, **hindsight oracle 추출**, 정규화, 합성 데이터 |
| `models.py` | ACT 계열 정책 (C0/C6 공용), numpy 폴백, 용량 일치 assert |
| `train.py` | 학습 루프, seed 고정, AMP, 체크포인트 |
| `evaluate.py` | rollout, **통계 검정 + 판정**, 리포트 집계 |
| `run_all.sh` | 3 arm × 3 seed 전체 실행 + 재시작 안전 |

출력:

```
checkpoints/{runs,smoke}/{c0,c6}_seed{0,1,2}.pt
results/{runs,smoke}/{c0,c6,c8}_seed{0,1,2}.json
results/{runs,smoke}/verdict.json      ← 최종 판정
logs/{runs,smoke}/*.log
```

---

## 7. ⚠️ 확인 필요 — 회사 머신 첫 실행 시

검증 불가능한 외부 API 는 전부 어댑터로 분리했다
(`tools/measure_loop_rate.py` 의 `G1Adapter` 와 같은 패턴).
아래가 틀려도 **해당 클래스/상수만** 고치면 나머지는 그대로 동작한다.

| # | 항목 | 위치 | 틀리면 생기는 증상 |
|---|---|---|---|
| 1 | LIBERO HDF5 의 EE 포즈 키 (`ee_pos`/`ee_ori` vs `ee_states`) | `dataset._EE_POS_KEYS` 등 | 로딩 단계에서 즉시 KeyError (조용히 넘어가지 않게 해 뒀다) |
| 2 | `OffScreenRenderEnv` import 경로·생성 인자 | `evaluate.LiberoRolloutEnv` | 평가 시작 시 즉시 실패 |
| 3 | env obs 의 `robot0_eef_quat` 이 **xyzw** 인지 | `dataset.quat_to_R` | **C8 만** 조용히 망가진다 (회전 오차) |
| 4 | `step()` 이 성공 시 `done=True` 를 주는지 | `LiberoRolloutEnv.step` | 전 arm 성공률 0% |
| 5 | OSC_POSE 정규화 스케일 `output_max` | `dataset.OSC_POS_MAX/OSC_ROT_MAX` | C8 이 과/저 명령 → C8 만 저조 |
| 6 | 이미지 상하 반전 규약 | `dataset.FLIP_RGB` / `canonical_rgb` | 성공률 통째로 0% |
| 7 | 회전 델타가 base frame 인지 EE frame 인지 | `hindsight_goal(ee_frame_rotation=)` | C8 회전 추종 저하 |
| 8 | benchmark task index ↔ hdf5 파일 정렬 순서 | `LiberoRolloutEnv._resolve_task_index` | **이름으로 매칭하도록 짜 뒀다.** 못 찾으면 후보를 출력하고 종료 |

3·5·7 은 **C8 만 선택적으로 망가뜨린다.** C8 이 비정상적으로 낮게 나오면
"학습 Model 2 사망" 규칙이 거짓으로 회피되므로, C8 이 낮을 때는 결론을 내리기
전에 반드시 이 셋을 먼저 확인할 것.

확인 명령:

```python
env.robots[0].controller.output_max      # → [pos×3, rot×3]  (항목 5)
env.reset(); obs.keys()                  # → 관측 키 전체     (항목 2,3,4)
```

---

## 8. 결과 읽는 법

`results/runs/verdict.json` 과 콘솔 리포트에 전부 들어 있다. 순서대로 볼 것:

1. **C8 이 비정상적으로 낮은가?** → 7절 3·5·7 을 먼저 의심. 컨트롤러 설정 문제일 수 있다.
2. **seed 산포가 pooled CI 폭보다 큰가?** → pooled p-value 를 믿지 말고 seed 를 더 돌린다.
3. **실제 n 의 MDE 가 ±5pt 보다 큰가?** → ±5pt 규칙은 판정 불가. 리포트가 경고한다.
4. 그 다음에 판정을 읽는다.

E0 의 결과가 "H1 사망" 이면 **그것이 이 실험의 성공이다.** 값싼 반증이 목적이고,
반증에 성공했다면 E1–E5 에 들어갈 수개월을 아낀 것이다.
