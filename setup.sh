#!/usr/bin/env bash
# =============================================================================
# physical_AI — clone 직후 실행하는 환경 셋업 / 진단 스크립트
#
# 왜 필요한가
#   기계가 넷(맥 / 집 1660 Super / 회사 3090 / Orin)인데 각각 할 수 있는 일이 다르다.
#   - GalbotSDK 는 Linux 전용 바이너리라 맥에서는 로봇 통신 자체가 불가능하다.
#   - torch 는 플랫폼마다 다른 휠이 필요하고 CUDA 버전을 박으면 한쪽이 깨진다.
#   그래서 "environment.yml 하나 + 플랫폼 분기는 이 스크립트"로 나눴다.
#   3090 박스에서 git clone 직후 이 파일 하나만 실행하면 바로 돌아가야 한다.
#
# 사용
#   bash setup.sh                생성 또는 갱신 + 자체 검증
#   bash setup.sh --check        아무것도 설치하지 않고 현재 상태만 진단
#   bash setup.sh --no-torch     conda 환경만 만들고 torch 는 건너뜀
#   bash setup.sh --force-torch  torch 가 이미 있어도 재설치
#   bash setup.sh --env NAME     환경 이름 변경 (기본 physical_ai)
#   bash setup.sh --skip-test    기구학 검증(약 8초) 건너뜀
#
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash setup.sh
#     → 특정 CUDA 빌드가 필요할 때만. 기본은 강제하지 않는다.
#
# 멱등성
#   환경이 이미 있으면 create 대신 update 를 쓴다. --prune 은 쓰지 않는다
#   (pip 로 넣은 torch/LIBERO 를 지워버리기 때문). 몇 번을 돌려도 안전하다.
#
# 호환성
#   bash 3.2 문법만 사용한다. macOS 기본 /bin/bash 가 3.2.57 이라
#   연관 배열, mapfile, ${var,,} 같은 bash 4+ 문법은 쓸 수 없다.
#   Ubuntu 의 bash 5 에서도 동일하게 동작한다. zsh 사용자도 `bash setup.sh` 로 실행.
# =============================================================================

set -euo pipefail

# ── 기본값 ───────────────────────────────────────────────────────────────────
ENV_NAME="physical_ai"
MODE_CHECK=0
DO_TORCH=1
FORCE_TORCH=0
RUN_TEST=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML="$REPO_ROOT/environment.yml"

# ── 인자 파싱 ────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --check)        MODE_CHECK=1 ;;
        --no-torch)     DO_TORCH=0 ;;
        --force-torch)  FORCE_TORCH=1 ;;
        --skip-test)    RUN_TEST=0 ;;
        --env)          shift; ENV_NAME="${1:-physical_ai}" ;;
        -h|--help)
            sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            printf '알 수 없는 인자: %s  (--help 참고)\n' "$1" >&2
            exit 2 ;;
    esac
    shift
done

# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────
C_G=""; C_R=""; C_Y=""; C_B=""; C_D=""; C_0=""
if [ -t 1 ]; then
    C_G=$'\033[92m'; C_R=$'\033[91m'; C_Y=$'\033[93m'
    C_B=$'\033[94m'; C_D=$'\033[2m';  C_0=$'\033[0m'
fi

N_FAIL=0
N_WARN=0

p_ok()   { printf '  %s[ OK ]%s %s\n' "$C_G" "$C_0" "$*"; }
p_warn() { printf '  %s[WARN]%s %s\n' "$C_Y" "$C_0" "$*"; N_WARN=$((N_WARN + 1)); }
p_fail() { printf '  %s[FAIL]%s %s\n' "$C_R" "$C_0" "$*"; N_FAIL=$((N_FAIL + 1)); }
p_info() { printf '  %s[ -- ]%s %s\n' "$C_D" "$C_0" "$*"; }
p_note() { printf '         %s%s%s\n' "$C_D" "$*" "$C_0"; }
p_do()   { printf '  %s[ >> ]%s %s\n' "$C_B" "$C_0" "$*"; }

section() {
    printf '\n%s%s%s\n' "$C_B" "$*" "$C_0"
    printf '%s%s%s\n' "$C_D" "────────────────────────────────────────────────────────────────────" "$C_0"
}

banner() {
    local tag=""
    [ "$MODE_CHECK" -eq 1 ] && tag="  [진단 모드 — 아무것도 설치하지 않음]"
    printf '%s====================================================================%s\n' "$C_B" "$C_0"
    printf '%s physical_AI 환경 셋업%s%s\n' "$C_B" "$tag" "$C_0"
    printf '%s====================================================================%s\n' "$C_B" "$C_0"
    printf ' repo : %s\n' "$REPO_ROOT"
    printf ' env  : %s\n' "$ENV_NAME"
}

# =============================================================================
# 1. 플랫폼 감지
# =============================================================================
OS_NAME="$(uname -s)"
ARCH="$(uname -m)"
HAS_NVIDIA=0        # nvidia-smi 가 실제로 GPU 를 보고했는가
GPU_CC_MIN=""       # 가장 낮은 compute capability (예: 75)
GPU_SUMMARY=""

detect_platform() {
    section "[1/5] 플랫폼"

    case "$OS_NAME" in
        Darwin) p_info "macOS / $ARCH  — 설계·기구학 검증·RoboCOIN 분석 담당" ;;
        Linux)  p_info "Linux / $ARCH  — 학습·SDK 호스트 담당" ;;
        *)      p_warn "예상 밖 OS: $OS_NAME (Darwin/Linux 만 검증됨)" ;;
    esac
    p_info "bash $BASH_VERSION"

    if [ "$OS_NAME" = "Darwin" ]; then
        p_info "CUDA 없음 (정상) — GalbotSDK·LIBERO 는 이 기계에서 불가"
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        p_warn "nvidia-smi 없음 → CPU 전용으로 취급. 학습·E0 는 이 기계에서 불가"
        return 0
    fi

    # compute_cap 질의는 드라이버 495+ 에서만 된다. 실패하면 이름만 얻는다.
    GPU_SUMMARY="$(nvidia-smi --query-gpu=name,compute_cap,memory.total \
                   --format=csv,noheader 2>/dev/null || true)"
    if [ -z "$GPU_SUMMARY" ]; then
        GPU_SUMMARY="$(nvidia-smi --query-gpu=name,memory.total \
                       --format=csv,noheader 2>/dev/null || true)"
    fi

    if [ -z "$GPU_SUMMARY" ]; then
        p_fail "nvidia-smi 는 있으나 GPU 질의 실패 — 드라이버 문제일 가능성"
        p_note "확인: nvidia-smi        복구: 드라이버 재설치 후 재부팅"
        return 0
    fi

    HAS_NVIDIA=1
    printf '%s\n' "$GPU_SUMMARY" | while IFS= read -r line; do
        [ -n "$line" ] && printf '  %s[ OK ]%s GPU  %s\n' "$C_G" "$C_0" "$line"
    done

    # compute capability 최솟값 추출: "7.5" → 75 로 바꿔 정수 비교한다.
    GPU_CC_MIN="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                  | tr -d ' .' | sort -n | head -1 || true)"

    if [ -n "$GPU_CC_MIN" ] && [ "$GPU_CC_MIN" -lt 80 ] 2>/dev/null; then
        p_warn "compute capability $GPU_CC_MIN — sm_80 미만 (1660 Super = sm_75 Turing)"
        p_note "bf16 불가(sm_80+ 필요) → AMP 는 fp16 으로. tensor core 없음(TU116)."
        p_note "flash-attention 계열도 sm_80+ 요구 → eager/SDPA math 로 우회."
    fi

    local drv
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    if [ -n "$drv" ]; then p_info "driver $drv"; fi
    return 0
}

# =============================================================================
# 2. conda / mamba 감지 (mamba 우선 — solve 가 훨씬 빠르다)
# =============================================================================
SOLVER_BIN=""
SOLVER_KIND=""

detect_solver() {
    section "[2/5] conda / mamba"

    # conda 는 셸 함수로 설치되는 경우가 많아 비대화형 스크립트에서는 PATH 에
    # 안 보일 수 있다. 그래서 PATH → CONDA_EXE → 흔한 설치 경로 순으로 찾는다.
    if command -v mamba >/dev/null 2>&1; then
        SOLVER_BIN="$(command -v mamba)"; SOLVER_KIND="mamba"
    elif command -v micromamba >/dev/null 2>&1; then
        SOLVER_BIN="$(command -v micromamba)"; SOLVER_KIND="micromamba"
    elif command -v conda >/dev/null 2>&1; then
        SOLVER_BIN="$(command -v conda)"; SOLVER_KIND="conda"
    elif [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE:-}" ]; then
        SOLVER_BIN="$CONDA_EXE"; SOLVER_KIND="conda"
    else
        local p
        for p in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" \
                 "$HOME/anaconda3" /opt/conda /opt/miniforge3 \
                 /opt/homebrew/Caskroom/miniforge/base; do
            if [ -x "$p/bin/conda" ]; then
                SOLVER_BIN="$p/bin/conda"; SOLVER_KIND="conda"; break
            fi
        done
    fi

    if [ -z "$SOLVER_BIN" ]; then
        p_fail "conda / mamba / micromamba 를 못 찾음"
        p_note "Ubuntu 설치 (miniforge — conda-forge 기본, mamba 포함):"
        p_note "  curl -fsSL -o mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        p_note "  bash mf.sh -b -p \$HOME/miniforge3 && \$HOME/miniforge3/bin/conda init bash"
        p_note "  새 셸을 연 뒤 다시 bash setup.sh"
        return 1
    fi

    local ver
    ver="$("$SOLVER_BIN" --version 2>/dev/null | head -1 || true)"
    p_ok "$SOLVER_KIND  $ver"
    p_note "$SOLVER_BIN"
    if [ "$SOLVER_KIND" = "conda" ]; then
        p_note "mamba 가 있으면 solve 가 수 배 빠르다: conda install -n base -c conda-forge mamba"
    fi
    return 0
}

# =============================================================================
# 3. 환경 생성 / 갱신
# =============================================================================
ENV_PREFIX=""
ENV_EXISTS=0
PY=""

resolve_env_prefix() {
    ENV_PREFIX=""
    # (a) 솔버가 직접 보고하는 경로가 가장 정확하다 (--prefix 로 만든 환경 포함)
    ENV_PREFIX="$("$SOLVER_BIN" env list 2>/dev/null \
                  | awk -v n="$ENV_NAME" '$1==n {print $NF}' | head -1 || true)"
    # (b) 못 찾으면 base/envs/NAME 관례를 시도
    if [ -z "$ENV_PREFIX" ]; then
        local base=""
        if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE:-}" ]; then
            base="$("$CONDA_EXE" info --base 2>/dev/null || true)"
        fi
        [ -z "$base" ] && [ -n "${MAMBA_ROOT_PREFIX:-}" ] && base="$MAMBA_ROOT_PREFIX"
        [ -n "$base" ] && [ -d "$base/envs/$ENV_NAME" ] && ENV_PREFIX="$base/envs/$ENV_NAME"
    fi

    if [ -n "$ENV_PREFIX" ] && [ -x "$ENV_PREFIX/bin/python" ]; then
        ENV_EXISTS=1
        PY="$ENV_PREFIX/bin/python"
    else
        ENV_EXISTS=0
        PY="$(command -v python3 2>/dev/null || true)"
    fi
}

setup_env() {
    section "[3/5] conda 환경  $ENV_NAME"

    if [ ! -f "$YML" ]; then
        p_fail "environment.yml 없음: $YML"
        p_note "리포가 온전히 clone 되지 않았습니다."
        return 1
    fi
    p_ok "environment.yml 확인"

    resolve_env_prefix

    if [ "$MODE_CHECK" -eq 1 ]; then
        if [ "$ENV_EXISTS" -eq 1 ]; then
            p_ok "환경 있음 — $ENV_PREFIX"
        else
            p_fail "환경 '$ENV_NAME' 없음"
            p_note "생성: bash setup.sh        (진단 모드라 만들지 않았다)"
            if [ -n "$PY" ]; then
                p_note "아래 패키지 점검은 대신 시스템 python3 기준으로 돌린다 — 참고용일 뿐이다."
            fi
        fi
        return 0
    fi

    local rc=0
    if [ "$ENV_EXISTS" -eq 1 ]; then
        p_do "환경이 이미 있어 update 로 진행 (멱등)"
        # --prune 은 쓰지 않는다: pip 로 넣은 torch/LIBERO 까지 날아간다.
        case "$SOLVER_KIND" in
            micromamba) "$SOLVER_BIN" install -y -n "$ENV_NAME" -f "$YML" || rc=$? ;;
            *)          "$SOLVER_BIN" env update -n "$ENV_NAME" -f "$YML" || rc=$? ;;
        esac
    else
        p_do "환경 생성 (수 분 소요)"
        case "$SOLVER_KIND" in
            micromamba) "$SOLVER_BIN" create -y -n "$ENV_NAME" -f "$YML" || rc=$? ;;
            *)          "$SOLVER_BIN" env create -n "$ENV_NAME" -f "$YML" || rc=$? ;;
        esac
    fi

    if [ "$rc" -ne 0 ]; then
        p_fail "$SOLVER_KIND 가 exit $rc 로 실패 — 위 solver 출력이 원인이다"
        p_note "흔한 원인: 네트워크 차단(사내 프록시), 디스크 부족, 채널 설정 충돌"
        p_note "채널 확인: $SOLVER_BIN config show channels   (conda-forge 만 있어야 한다)"
        return 1
    fi

    resolve_env_prefix
    if [ "$ENV_EXISTS" -ne 1 ]; then
        p_fail "환경은 만들어졌다는데 $ENV_NAME/bin/python 을 못 찾음"
        p_note "확인: $SOLVER_BIN env list"
        return 1
    fi
    p_ok "환경 준비됨 — $ENV_PREFIX"
    return 0
}

# =============================================================================
# 4. torch (pip). CUDA 버전은 강제하지 않는다.
# =============================================================================
install_torch() {
    [ "$MODE_CHECK" -eq 1 ] && return 0
    [ "$DO_TORCH" -eq 0 ] && { p_info "--no-torch → torch 설치 건너뜀"; return 0; }
    [ "$ENV_EXISTS" -ne 1 ] && return 0

    if [ "$FORCE_TORCH" -eq 0 ] && "$PY" -c "import torch" >/dev/null 2>&1; then
        p_info "torch 이미 있음 → 건너뜀 (재설치는 --force-torch)"
        return 0
    fi

    # 인덱스 선택 근거
    #   Linux + NVIDIA : PyPI 기본 휠이 CUDA 빌드다. 여기에 특정 cuXXX 를 박으면
    #                    sm_75(1660)·sm_86(3090)·sm_87(Orin) 중 하나가 깨진다.
    #                    필요하면 TORCH_INDEX_URL 로 사용자가 직접 고른다.
    #   Linux CPU only : 기본 휠은 CUDA 라이브러리를 수 GB 끌고 오므로 cpu 인덱스.
    #   macOS          : universal wheel 하나에 MPS 가 들어 있다.
    local idx=""
    if [ -n "${TORCH_INDEX_URL:-}" ]; then
        idx="$TORCH_INDEX_URL"
        p_do "torch 설치 — 사용자 지정 인덱스"
    elif [ "$OS_NAME" = "Linux" ] && [ "$HAS_NVIDIA" -eq 0 ]; then
        idx="https://download.pytorch.org/whl/cpu"
        p_do "torch 설치 — CPU 휠 (NVIDIA GPU 미검출)"
    else
        p_do "torch 설치 — 기본 휠 (CUDA 버전 강제하지 않음)"
    fi

    local rc=0
    if [ -n "$idx" ]; then
        "$PY" -m pip install --upgrade torch --index-url "$idx" || rc=$?
    else
        "$PY" -m pip install --upgrade torch || rc=$?
    fi

    if [ "$rc" -ne 0 ]; then
        p_fail "torch 설치 실패 (pip exit $rc)"
        p_note "환경 자체는 살아 있다. torch 없이 쓸 수 있는 것: make test / make analysis"
        p_note "재시도: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash setup.sh --force-torch"
    fi
    return 0
}

# =============================================================================
# 5. 자체 검증
# =============================================================================
probe_mod() {
    # $1 모듈명  $2 req|opt  $3 없을 때 안내
    local mod="$1" need="$2" hint="${3:-}"
    local out=""
    if out="$("$PY" -c "import $mod; print(getattr($mod, \"__version__\", \"?\"))" 2>/dev/null)"; then
        printf '  %s[ OK ]%s %-16s %s\n' "$C_G" "$C_0" "$mod" "$out"
    elif [ "$need" = "req" ]; then
        p_fail "$mod 없음 (필수)"
        [ -n "$hint" ] && p_note "$hint"
    else
        p_warn "$mod 없음 (선택)"
        [ -n "$hint" ] && p_note "$hint"
    fi
}

# torch 상세 점검. 문자열 안에 작은따옴표를 쓸 수 없으므로 파이썬 코드는 큰따옴표만 사용.
TORCH_PROBE='
import torch
print("  torch " + torch.__version__)
cu = torch.cuda.is_available()
mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
if cu:
    cap = torch.cuda.get_device_capability(0)
    gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print("  CUDA  %s  sm_%d%d  %.1f GB" % (torch.cuda.get_device_name(0), cap[0], cap[1], gb))
    print("  bf16  " + ("지원" if cap[0] >= 8 else "미지원 — sm_80 미만이므로 fp16 AMP 를 쓸 것"))
elif mps:
    print("  MPS   사용 가능 (Apple GPU)")
else:
    print("  가속기 없음 — CPU 전용")
'

verify() {
    section "[4/5] 패키지"

    if [ -z "$PY" ]; then
        p_fail "실행할 python 이 없음"
        return 0
    fi
    p_info "python: $PY"
    local pyver
    pyver="$("$PY" -V 2>&1 || true)"
    case "$pyver" in
        *3.11*) p_ok "$pyver" ;;
        *)      p_warn "$pyver — 이 프로젝트는 3.11 기준 (기계 간 checkpoint 호환)" ;;
    esac

    probe_mod numpy      req "bash setup.sh 로 환경을 갱신할 것"
    probe_mod scipy      req "bash setup.sh"
    probe_mod pyarrow    req "tools/robocoin_analysis.py 가 parquet 를 읽는 데 필요"
    probe_mod pandas     opt "bash setup.sh"
    probe_mod matplotlib opt "bash setup.sh"
    probe_mod mujoco     opt "URDF 대조 검증(PLAN M5)에 필요"
    # MCAP 는 회사 Linux 에서만 실제로 쓴다 (맥에서는 --dry-run 만 가능).
    # 없으면 tools/mcap2lerobot_fork.py 로 실기체 기록을 한 개도 못 읽는다.
    probe_mod mcap       opt "실 MCAP 변환에 필요 — 회사 Linux 에서 bash setup.sh"
    probe_mod mcap_ros2  opt "MCAP ros2 디코더 — $PY -m pip install mcap-ros2-support"

    if "$PY" -c "import torch" >/dev/null 2>&1; then
        "$PY" -c "$TORCH_PROBE" 2>/dev/null | while IFS= read -r line; do
            printf '  %s[ OK ]%s %s\n' "$C_G" "$C_0" "$line"
        done
    else
        if [ "$OS_NAME" = "Linux" ] && [ "$HAS_NVIDIA" -eq 1 ]; then
            p_fail "torch 없음 — 이 기계는 학습/E0 담당이므로 필수"
            p_note "설치: bash setup.sh   (또는 $PY -m pip install torch)"
        else
            p_warn "torch 없음 (맥에서는 벤치마크에만 필요)"
            p_note "설치: bash setup.sh"
        fi
    fi

    # ── 로봇 전용 의존성: 없는 게 정상인 경우가 많아 FAIL 로 세지 않는다 ──
    if "$PY" -c "import galbot_sdk" >/dev/null 2>&1; then
        p_ok "galbot_sdk 있음 — 실기체 명령 가능"
    elif [ "$OS_NAME" = "Darwin" ]; then
        p_info "galbot_sdk 없음 — macOS 에서는 정상 (SDK 는 Linux 전용 바이너리)"
    else
        p_info "galbot_sdk 없음 — GATE-1 실측 전에 벤더 휠 설치 필요 (⚠️ 모듈명 확인 필요)"
    fi

    if "$PY" -c "import libero" >/dev/null 2>&1; then
        p_ok "LIBERO 있음 — GATE-2(E0) 실행 가능"
    elif [ "$HAS_NVIDIA" -eq 1 ]; then
        p_info "LIBERO 없음 — GATE-2(E0) 를 이 기계에서 돌리려면 설치 필요 (아래 안내)"
    fi

    # ── 리포 파일 온전성 ──
    local f miss=0
    for f in robot/g1_kinematics.py robot/interface.py robot/test_kinematics.py \
             robot/assets/g1_joints_raw.json tools/measure_loop_rate.py \
             tools/robocoin_analysis.py; do
        if [ ! -f "$REPO_ROOT/$f" ]; then
            p_fail "누락: $f"
            miss=1
        fi
    done
    if [ "$miss" -eq 0 ]; then
        p_ok "리포 파일 6종 확인 (기구학·인터페이스·URDF joint·도구)"
    else
        p_note "clone 이 불완전하다: git status / git checkout -- . 로 복구할 것"
    fi
    return 0
}

run_kinematics_test() {
    section "[5/5] 기구학 검증  robot/test_kinematics.py"

    if [ "$RUN_TEST" -eq 0 ]; then
        p_info "--skip-test → 건너뜀"
        return 0
    fi
    if [ -z "$PY" ] || [ ! -f "$REPO_ROOT/robot/test_kinematics.py" ]; then
        p_fail "실행 불가 (python 또는 스크립트 없음)"
        return 0
    fi
    if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
        p_fail "numpy 가 없어 실행 불가 — bash setup.sh 로 환경부터 만들 것"
        return 0
    fi

    local log="${TMPDIR:-/tmp}/physical_ai_kin_test.$$"
    p_info "실행 중 (약 8초 — T6 의 cold seed IK 60회가 대부분을 차지)"
    if "$PY" "$REPO_ROOT/robot/test_kinematics.py" > "$log" 2>&1; then
        local score
        score="$(grep -E "통과$" "$log" | tail -1 | sed "s/^ *//" || true)"
        p_ok "${score:-완료} — S–R–S 구조 / IK 왕복 / psi 스윕 / L2 붕괴 / head 변환 / IK 속도"
        p_note "전체 출력: $PY robot/test_kinematics.py"
    else
        p_fail "기구학 검증 실패 — 아래 마지막 20줄"
        sed 's/^/      /' "$log" | tail -20
        p_note "로그: $log"
        return 0
    fi
    rm -f "$log"
}

# =============================================================================
# 6. 이 기계에서 다음에 할 일
# =============================================================================
guidance() {
    section "이 기계에서 할 수 있는 것 / 없는 것"

    if [ "$OS_NAME" = "Darwin" ]; then
        printf '  %s가능%s  기구학 검증(make test) · RoboCOIN 분석(make analysis) · GATE-1 dry-run\n' "$C_G" "$C_0"
        printf '  %s가능%s  torch MPS 벤치마크 (docs/research/benchmarks/*.py)\n' "$C_G" "$C_0"
        printf '  %s불가%s  GalbotSDK — linux-x86_64/aarch64 바이너리만 존재. 실기체는 회사 Linux 에서\n' "$C_R" "$C_0"
        printf '  %s불가%s  CUDA 학습 · TensorRT 엔진 빌드\n' "$C_R" "$C_0"
        printf '  %s권장%s  LIBERO 는 설치하지 말 것 (robosuite/MuJoCo 렌더 경로가 맥에서 자주 깨진다)\n' "$C_Y" "$C_0"
        return 0
    fi

    if [ "$HAS_NVIDIA" -eq 1 ]; then
        printf '  %s가능%s  학습 · E0 · eval · GATE-1 실측(로봇 LAN 에 있을 때)\n' "$C_G" "$C_0"
        if [ -n "$GPU_CC_MIN" ] && [ "$GPU_CC_MIN" -lt 80 ] 2>/dev/null; then
            printf '  %s주의%s  sm_%s — bf16 없음. AMP 는 fp16(GradScaler) 으로, flash-attention 불가\n' "$C_Y" "$C_0" "$GPU_CC_MIN"
            printf '  %s주의%s  1660 Super 는 6GB — E0 의 ~25M 모델은 들어가지만 batch/이미지 해상도를 줄일 것\n' "$C_Y" "$C_0"
        else
            printf '  %s참고%s  sm_86(3090) — bf16 사용 가능. 다른 업무와 공유되는 GPU 이므로\n' "$C_B" "$C_0"
            printf '         %s학습 전 nvidia-smi 로 여유 VRAM 확인 (GATE-1 측정 중에는 특히)%s\n' "$C_D" "$C_0"
        fi
    else
        printf '  %s불가%s  학습 · E0 — NVIDIA GPU 가 없다\n' "$C_R" "$C_0"
    fi

    printf '\n  %sconda 로 설치할 수 없는 두 가지 (수동)%s\n' "$C_B" "$C_0"
    printf '  1) GalbotSDK  — 벤더 배포 바이너리. Ubuntu 20~24 / x86_64 · aarch64 전용\n'
    printf '       %s벤더 휠을 받아: %s -m pip install <galbot_sdk>.whl%s\n' "$C_D" "${PY:-python}" "$C_0"
    printf '       %s⚠️ 확인 필요 — 실제 배포 형식(whl/deb/tar)과 import 이름을 회사에서 검증할 것.%s\n' "$C_D" "$C_0"
    printf '       %stools/measure_loop_rate.py 의 어댑터 클래스만 고치면 되도록 분리돼 있다.%s\n' "$C_D" "$C_0"
    printf '  2) LIBERO     — PyPI 에 없음. GATE-2(E0) 전용\n'
    printf '       %sgit clone https://github.com/Lifelong-Robot-Learning/LIBERO.git%s\n' "$C_D" "$C_0"
    printf '       %s%s -m pip install -e LIBERO   (robosuite 를 함께 끌고 온다)%s\n' "$C_D" "${PY:-python}" "$C_0"
    printf '       %s헤드리스 서버면 MUJOCO_GL=egl 를 export 할 것.%s\n' "$C_D" "$C_0"
}

# =============================================================================
# main
# =============================================================================
banner
detect_platform

if ! detect_solver; then
    printf '\n%s중단%s — 패키지 매니저가 없어 더 진행할 수 없다.\n' "$C_R" "$C_0"
    exit 1
fi

setup_env || true
install_torch
verify
run_kinematics_test
guidance

# ── 요약 ─────────────────────────────────────────────────────────────────────
printf '\n%s====================================================================%s\n' "$C_B" "$C_0"
if [ "$N_FAIL" -eq 0 ] && [ "$N_WARN" -eq 0 ]; then
    printf ' %s전부 정상.%s  다음: make test / make help\n' "$C_G" "$C_0"
elif [ "$N_FAIL" -eq 0 ]; then
    printf ' %s필수 항목 정상, 경고 %d건.%s  위 [WARN] 을 확인할 것 — 대개 이 기계에서\n' "$C_G" "$N_WARN" "$C_0"
    printf ' 안 쓰는 선택 패키지라 무시해도 된다. 다음: make test / make help\n'
else
    printf ' %s실패 %d건, 경고 %d건.%s  위 [FAIL] 항목을 먼저 해결할 것.\n' "$C_R" "$N_FAIL" "$N_WARN" "$C_0"
    if [ "$MODE_CHECK" -eq 1 ]; then
        printf ' 진단 모드였다. 설치까지 하려면: %sbash setup.sh%s\n' "$C_B" "$C_0"
    fi
fi
printf '%s====================================================================%s\n' "$C_B" "$C_0"

[ "$N_FAIL" -eq 0 ] || exit 1
exit 0
