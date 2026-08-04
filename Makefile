# =============================================================================
# physical_AI — 자주 쓰는 명령 모음
#
# 각 타겟 뒤에 [어느 기계에서 도는지] 를 표시한다. 이 프로젝트는 기계마다
# 할 수 있는 일이 다르고(맥=CUDA 없음, 맥=GalbotSDK 불가, 3090=공유 자원),
# 잘못된 기계에서 돌리면 시간만 버리기 때문이다.
#
#   make            도움말
#   make setup      환경 생성/갱신
#   make test       기구학 검증
#
# python 선택 순서
#   1) 명령행 지정        make PY=/path/to/python test
#   2) conda 환경 physical_ai 의 python  (activate 안 해도 된다)
#   3) 그것도 없으면 python3
# GNU make 3.81(macOS 기본) 에서도 돌아가야 하므로 .ONESHELL 등 4.x 문법은 안 쓴다.
# =============================================================================

# GalbotSDK 의 setup.sh 는 bash 문법을 쓴다. make 기본 셸(/bin/sh = dash on Ubuntu)로
# source 하면 "Bad substitution" 이 나고 환경이 안 잡힌다 (2026-07-31 3090 에서 실제 발생).
# 그때는 앞선 셸에서 이미 source 해둔 덕에 우연히 동작했을 뿐이라, 새 셸에서는 실패한다.
SHELL      := /bin/bash

ENV_NAME   ?= physical_ai

# CONDA_EXE 는 conda init 이 export 해 주므로 subprocess 없이 base 를 알아낼 수 있다.
# 없을 때만 conda info 를 부른다 (느리다).
CONDA_BASE := $(if $(CONDA_EXE),$(patsubst %/bin/conda,%,$(CONDA_EXE)),$(shell conda info --base 2>/dev/null))
ENV_PY     := $(CONDA_BASE)/envs/$(ENV_NAME)/bin/python
PY         := $(if $(wildcard $(ENV_PY)),$(ENV_PY),python3)

# --host 는 자유 문자열이 아니라 "측정 위치" 선택지다 (결과 json 에 기록된다):
#   onboard     로봇 온보드 컴퓨터(AGX Orin)에서 측정 — 배포 시 실제 지연
#   external    LAN 너머 외부 머신(회사 3090)에서 측정 — 개발 시 지연
#   unspecified 미지정
# 두 값의 차이가 곧 "정책을 어디서 돌릴 것인가"의 근거가 되므로 둘 다 재야 한다.
HOST       ?= unspecified
MACHINE    := $(shell hostname -s 2>/dev/null || echo unknown)
DATA       ?= data/robocoin
OUT        ?= gate1_results_$(HOST)_$(MACHINE).json

# GalbotSDK 는 setup.sh 를 source 해야 import 된다 (LD_LIBRARY_PATH/PYTHONPATH).
# 경로가 다르면:  make SDK_SETUP=/실제/경로/setup.sh probe
SDK_SETUP  ?= /opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh

.DEFAULT_GOAL := help
.PHONY: help setup check test gate1 gate1-real probe probe-check probe-live probe-entry fk-check analysis inspect e0-check e0-smoke bench lock clean

# ─────────────────────────────────────────────────────────────────────────────
help:  ## [모든 기계]
	@echo ""
	@echo "physical_AI — make 타겟"
	@echo "  python: $(PY)"
	@echo ""
	@echo "  환경"
	@echo "    make setup        [모든 기계]  conda 환경 생성/갱신 + 자체 검증"
	@echo "    make check        [모든 기계]  설치 없이 진단만 (setup.sh --check)"
	@echo "    make lock         [모든 기계]  현재 환경을 기계별 lock 파일로 고정"
	@echo ""
	@echo "  검증 — 로봇 없이 되는 것"
	@echo "    make test         [모든 기계]  기구학 T1~T7, 약 8초"
	@echo "    make gate1        [모든 기계]  GATE-1 dry-run (SDK 없이 스크립트만 검증)"
	@echo "    make e0-check     [1660/3090]  E0 를 이 기계에서 돌릴 수 있는지 점검"
	@echo "    make e0-smoke     [모든 기계]  E0 배선 검증 (합성 데이터, GPU 불필요)"
	@echo ""
	@echo "    make probe-check  [모든 기계]  probe_sdk 파서/안전차단 자체검증"
	@echo ""
	@echo "  SDK"
	@echo "    make probe        [회사 Linux]  GalbotSDK 시그니처 추출 (로봇 불필요)"
	@echo "    make probe-live   [회사 Linux + 로봇]  읽기 전용 실물 조회 (안 움직임)"
	@echo "    make fk-check     [회사 Linux + 로봇]  SDK FK 와 우리 FK 대조 (안 움직임)"
	@echo "    make probe-entry  [회사 Linux]  ⚠️ 핸들 획득이 막혔을 때 진입점 조사"
	@echo ""
	@echo "  실측"
	@echo "    make gate1-real   [회사 Linux + 로봇]  ⚠️ 실기체가 움직인다"
	@echo "    make analysis     [맥 / 1660]  RoboCOIN 3종 분석  DATA=<경로>"
	@echo "    make inspect      [맥 / 1660]  parquet 스키마만 확인  DATA=<경로>"
	@echo "    make bench        [맥 MPS / CUDA]  대뇌·소뇌 지연 벤치마크"
	@echo ""
	@echo "    make clean        [모든 기계]  __pycache__ 정리 (측정 결과는 안 지운다)"
	@echo ""

# ── 환경 ─────────────────────────────────────────────────────────────────────
setup:  ## [모든 기계]
	bash setup.sh

check:  ## [모든 기계] 설치를 건드리지 않는 진단. 새 기계에서 제일 먼저 돌릴 것
	bash setup.sh --check

# 학습을 시작하는 시점에 기계별로 남긴다. environment.yml 은 버전을 풀어 두었으므로
# (플랫폼마다 깨지기 때문) 재현이 필요하면 이 lock 파일을 근거로 삼는다.
lock:  ## [모든 기계]
	conda env export -n $(ENV_NAME) --no-builds > environment.$(shell uname -s).$(shell uname -m).lock.yml
	@echo "→ environment.$(shell uname -s).$(shell uname -m).lock.yml"

# ── 검증 ─────────────────────────────────────────────────────────────────────
# T1 S–R–S / T2 IK 왕복 / T3 psi 스윕 / T4 L2 붕괴 / T5 head 변환 / T6 속도 / T7 tip 프레임.
# 문서에 인용된 수치(warm 2.83ms, psi 스윕 팔꿈치 263mm, 목 1도당 10.5mm)를 직접 재현한다.
test:  ## [모든 기계]
	$(PY) robot/test_kinematics.py

# GalbotSDK 없이 스크립트 경로만 검증한다. 회사에 가기 전에 맥에서 돌려
# 로직 오류를 미리 잡는 용도 — 회사 시간이 희소 자원이기 때문.
gate1:  ## [모든 기계]
	$(PY) tools/measure_loop_rate.py --dry-run --host unspecified \
		--dwell 1.0 --rates 10,50,100 --skip-bode --out /dev/null

# ── GalbotSDK 표면 조사 ──────────────────────────────────────────────────────
# galbot_sdk 는 pybind11 확장이라 inspect.signature() 가 전부 실패한다.
# 실제 인자는 docstring 에만 있으므로 이 도구로 긁어야 어댑터를 확정할 수 있다.
# probe      = 정적 조사만. 로봇에 연결하지 않고 아무것도 움직이지 않는다.
# probe-live = 읽기 전용 실물 조회. set_/move_/execute_ 는 코드에서 하드 차단됨.
#              손목 F/T 실값, 관절 이름 순서, 카메라 extrinsic 을 여기서 한 번에 얻는다.
probe-check:  ## [모든 기계] SDK 없이 파서/안전차단 검증
	$(PY) tools/probe_sdk.py --self-test
	$(PY) tools/fk_crosscheck.py --self-test

# SDK FK 와 우리 FK 대조 — URDF 자기일관성을 벗어나는 첫 외부 검증.
# forward_kinematics 는 joint_state 를 인자로 받으므로 로봇을 움직이지 않는다.
# 이 결과가 통과해야 psi/T_rel 을 데이터셋에 굽는 게 안전하다 (소급 수정 불가).
fk-check:  ## [회사 Linux + 로봇] 로봇은 움직이지 않는다
	@test -f "$(SDK_SETUP)" || { echo "SDK setup.sh 없음: $(SDK_SETUP)"; exit 1; }
	. $(SDK_SETUP) && $(PY) tools/fk_crosscheck.py --n 200

probe:  ## [회사 Linux] 로봇 불필요
	@test -f "$(SDK_SETUP)" || { echo "SDK setup.sh 없음: $(SDK_SETUP)"; \
		echo "  make SDK_SETUP=/실제/경로/setup.sh probe"; \
		echo "  찾기:  find /opt -maxdepth 6 -type d -name galbot_sdk"; exit 1; }
	. $(SDK_SETUP) && $(PY) tools/probe_sdk.py --focus \
		--out sdk_surface_$(MACHINE).json --md sdk_surface_$(MACHINE).md

# GalbotRobot/GalbotMotion 이 "No constructor defined!" 로 생성 안 될 때.
# pybind11 이 py::init<>() 없이 바인딩한 클래스는 직접 생성이 아니라 어딘가에서
# 받아오는 것이다. __init__.py 가 Python 소스이므로 거기 진입점이 있을 수 있다.
probe-entry:  ## [회사 Linux] 진입점 조사. 로봇 불필요
	@test -f "$(SDK_SETUP)" || { echo "SDK setup.sh 없음: $(SDK_SETUP)"; exit 1; }
	. $(SDK_SETUP) && $(PY) tools/probe_sdk.py --entry 2>&1 | tee sdk_entry_$(MACHINE).txt
	@echo "→ sdk_entry_$(MACHINE).txt 를 공유해주십시오."

probe-live:  ## [회사 Linux + 로봇] 읽기 전용. 움직이지 않는다
	@test -f "$(SDK_SETUP)" || { echo "SDK setup.sh 없음: $(SDK_SETUP)"; exit 1; }
	. $(SDK_SETUP) && $(PY) tools/probe_sdk.py --live --focus \
		--out sdk_live_$(MACHINE).json --md sdk_live_$(MACHINE).md

# ⚠️ 실기체가 실제로 움직인다. left_arm_joint4 를 ±2° 로만 흔들지만
#    팔 주변을 비우고 e-stop 을 손 닿는 곳에 둘 것. 측정 중 이 머신에서
#    다른 워크로드를 돌리면 jitter 가 오염된다 (3090 은 공유 자원이라 특히 주의).
#    HOST 를 반드시 지정할 것:  make gate1-real HOST=external   (3090 에서)
#                              make gate1-real HOST=onboard    (Orin 에서)
gate1-real:  ## [회사 Linux + 로봇 LAN]
	@test "$(HOST)" != "unspecified" || { \
		echo "HOST 를 지정할 것: make gate1-real HOST=external  (또는 HOST=onboard)"; exit 1; }
	@test -f "$(SDK_SETUP)" || { echo "SDK setup.sh 없음: $(SDK_SETUP)"; \
		echo "  source 하지 않으면 import galbot_sdk 가 실패합니다."; exit 1; }
	@echo "⚠️  실기체 동작 ($(HOST)). 팔 주변 정리 / e-stop 확인 후 Enter, 중단은 Ctrl-C"
	@read _
	. $(SDK_SETUP) && $(PY) tools/measure_loop_rate.py --host $(HOST) --out $(OUT)

# ── 데이터 분석 ───────────────────────────────────────────────────────────────
# RoboCOIN(공개 G1 실기체 데이터) 3종: 목 기여도 / psi 분포 / 시야 내 가시성.
# ⚠️ 공식 mcap2lerobot 출력에는 F/T 채널이 0개다 — 이 분석으로는 F/T 를 볼 수 없다.
analysis:  ## [맥 / 1660]
	@test -d "$(DATA)" || { echo "데이터 없음: $(DATA)"; \
		echo "  make analysis DATA=/경로/robocoin  또는 huggingface_hub 로 먼저 내려받을 것"; exit 1; }
	$(PY) tools/robocoin_analysis.py --data $(DATA) --all --out robocoin_analysis_$(MACHINE).json

# 컬럼 매핑이 예상과 다를 수 있으므로 전체 분석 전에 스키마부터 본다.
inspect:  ## [맥 / 1660]
	@test -d "$(DATA)" || { echo "데이터 없음: $(DATA)  (make inspect DATA=/경로)"; exit 1; }
	$(PY) tools/robocoin_analysis.py --data $(DATA) --inspect --limit 3

# ── E0 (GATE-2) ──────────────────────────────────────────────────────────────
# 하네스는 experiments/e0/ 에 있다 (dataset/models/train/evaluate/run_all.sh).
# e0-check  = 이 기계에서 E0 를 돌릴 수 있는 상태인가만 본다 (VRAM·bf16·LIBERO).
#             1660 Super 6GB 가 실제 제약이라 sm 을 먼저 확인해야 프로파일이 정해진다.
# e0-smoke  = 합성 데이터로 데이터→학습→rollout→판정 전 경로를 실제로 태운다.
#             GPU/LIBERO 없이도 돌고 수 분이면 끝난다. 본 실험 전에 반드시 통과할 것.
e0-check:  ## [1660 / 3090]
	@echo "── E0 실행 가능성 점검 ──────────────────────────────────"
	@$(PY) -c "import sys; print('  python     ' + sys.version.split()[0])"
	@$(PY) -c "import torch; \
c = torch.cuda.is_available(); \
p = torch.cuda.get_device_properties(0) if c else None; \
cap = torch.cuda.get_device_capability(0) if c else (0, 0); \
print('  torch      ' + torch.__version__); \
print('  CUDA       ' + (p.name + '  %.1f GB' % (p.total_memory / 1024**3) if c else '없음 — E0 불가')); \
print('  sm         ' + ('%d%d' % cap if c else '-') + ('  bf16 지원' if cap[0] >= 8 else '  bf16 불가 → fp16 AMP(GradScaler) 사용')); \
print('  flash-attn ' + ('가능' if cap[0] >= 8 else '불가(sm_80+ 요구) → SDPA math/eager 로'))" \
		2>/dev/null || echo "  torch      없음 — bash setup.sh"
	@$(PY) -c "import libero; print('  LIBERO     OK')" 2>/dev/null \
		|| echo "  LIBERO     없음 — git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && $(PY) -m pip install -e LIBERO"
	@echo "  하네스     experiments/e0/  (C0 MONO / C6 SPLIT-ORACLE / C8 ORACLE-IK)"
	@echo "  다음       make e0-smoke  →  bash experiments/e0/run_all.sh --data <LIBERO 루트>"
	@echo "─────────────────────────────────────────────────────────"

# 합성 데이터 배선 검증. LIBERO/GPU 없이 돌며, 여기서 나온 성공률은
# 과학적 의미가 없다 — "끊긴 데가 없다"만 확인하는 것이다.
e0-smoke:  ## [모든 기계]
	PYTHON=$(PY) bash experiments/e0/run_all.sh --smoke

# ── 벤치마크 ─────────────────────────────────────────────────────────────────
# 문서에서 [M] 로 표기된 지연 수치의 출처. 맥에서는 MPS, CUDA 박스에서는 GPU 로 돈다.
# 기계가 다르면 절대값은 달라지지만 결론의 방향(대뇌 5~10Hz / 소뇌 25~60Hz)은 유지된다.
bench:  ## [맥 MPS / CUDA]
	@$(PY) -c "import torch" 2>/dev/null || { echo "torch 없음 — bash setup.sh"; exit 1; }
	$(PY) docs/research/benchmarks/hybrid.py

# ── 정리 ─────────────────────────────────────────────────────────────────────
# 측정 결과(gate1_results*.json, robocoin_analysis*.json)는 기계별 실측이라 지우지 않는다.
clean:  ## [모든 기계]
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "정리 완료 (측정 결과 json 은 그대로 두었다)"
