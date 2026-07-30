#!/usr/bin/env python3
"""
E0 데이터 계층 — LIBERO 로딩 + hindsight oracle 포즈 추출

왜 이 파일이 필요한가
─────────────────────
E0 는 "정답 인터페이스가 주어졌을 때 저수준 정책이 얼마나 잘하는가" 를 재는
실험이다. 그 **정답**을 만드는 곳이 여기다. oracle 은 학습된 Model 1 이 아니라
데모의 미래를 그냥 들여다본 것(hindsight)이다:

    goal(t) = EE_pose(t + K)          # K = ORACLE_HORIZON = 16 (= 0.8s @ 20Hz)

앞으로 만들 어떤 Model 1 도 이 정답을 이길 수 없다. 따라서 이 정답을 먹인
C6 가 C0(monolith)를 못 이기면 배포판 분리는 확실히 더 못한다. 이것이 E0 가
"가능한 가장 값싼 반증" 인 이유다.

핵심 설계 결정 3가지
────────────────────
1) **oracle 은 절대 포즈가 아니라 chunk 시작 시점 기준 델타**로 준다.
   실측: EE 절대 표현 69.0% vs EE 델타 표현 89.6%. 델타는 hand-eye 캘리브
   편향(~2.1mm/3.2°)을 양 끝점이 공유해서 상쇄하고, 절대 좌표의 분포 이동
   (물체 위치가 바뀌면 절대 좌표는 OOD)에 강하다.

2) **proprio 에 현재 EE 포즈를 넣는다 (C0 에도).**
   안 넣으면 C0 는 자기 손이 어디 있는지도 모르는 채로 싸우게 되고,
   C6-C0 격차가 "oracle 의 가치" 가 아니라 "EE 포즈 관측의 가치" 로 오염된다.
   C0 도 현재 EE 포즈는 안다. 모르는 건 **미래** 뿐이다.

3) **train/eval 데모 분할은 기본 25/25.**
   판정 임계가 ±5pt 인데 n=500 의 MDE 는 8.9pt 다. 3 seed 를 pool 해서
   n=1500(MDE 5.1pt)을 만들어야 +5pt 규칙이 의미를 갖는다. 태스크당 25개를
   eval 로 빼면 20태스크 × 25 = seed 당 정확히 500 rollout 이 나온다.
   (--split 40 으로 40/10 분할도 가능하나 seed 당 n=200 으로 떨어진다.)

torch 없이도 import 된다. LIBERO/h5py 는 실제 데이터를 만질 때만 import 한다.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

# 같은 리포의 인터페이스 스키마를 재사용한다 (6D 회전 표현 등)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from robot.interface import R_to_6d, sixd_to_R  # noqa: F401
except Exception:  # 리포 밖에서 단독 실행되는 경우를 위한 폴백
    def R_to_6d(R: np.ndarray) -> np.ndarray:
        """회전행렬 → 6D. 앞 두 열을 그대로 쓴다 (Zhou et al. 2019)."""
        return np.concatenate([R[:, 0], R[:, 1]])

    def sixd_to_R(v: np.ndarray) -> np.ndarray:
        """6D → 회전행렬. Gram-Schmidt 로 직교화."""
        a1, a2 = np.asarray(v[:3], float), np.asarray(v[3:6], float)
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# 상수 — E0 전체가 공유한다
# ═════════════════════════════════════════════════════════════════════════════

CHUNK_K = 16            # action chunk 길이. ACT 계열 표준. 20Hz 기준 0.8s
ORACLE_HORIZON = 16     # oracle 이 들여다보는 미래 스텝 수. CHUNK_K 와 같게 둔다
                        #   → "chunk 끝에서 손이 있어야 할 곳" 이 곧 goal 이 된다

IMG_SIZE = 128          # LIBERO 기본 해상도
N_CAMERAS = 2           # agentview + eye_in_hand
ACTION_DIM = 7          # OSC_POSE delta: [dx,dy,dz, ax,ay,az, gripper]
CONTROL_HZ = 20.0       # LIBERO env control_freq

# proprio: joint(7) + gripper qpos(2) + EE pos(3) + EE 회전 6D(6) = 18
PROPRIO_DIM = 7 + 2 + 3 + 6

# oracle goal: dp(3) + dR6(6) + 목표 gripper 폭(1) = 10
#   ⚠️ G1 실기체로 옮길 때는 robot/interface.py 의 Waypoint.DIM=16 이 된다
#      (여기서 빠진 것: dt — E0 에서는 K/20s 로 상수라 정보량 0,
#       psi — Panda 도 7-DoF 라 의미는 있으나 FK 의존이라 E0 범위 밖,
#       stiffness/squeeze — LIBERO OSC_POSE 는 고정 impedance 라 채널 자체가 없음)
GOAL_DIM = 3 + 6 + 1

# robosuite OSC_POSE 컨트롤러의 정규화 스케일.
# ⚠️ 확인 필요 — robosuite 기본값을 적었다. LIBERO 가 자체 controller config 를
#    쓰면 달라진다. 회사 머신 첫 실행 시 아래로 실측하고 고칠 것:
#      env.robots[0].controller.output_max  →  [pos*3, rot*3]
OSC_POS_MAX = 0.05      # [m]  한 스텝 최대 병진
OSC_ROT_MAX = 0.5       # [rad] 한 스텝 최대 회전
GRIPPER_CLOSE = 1.0     # robosuite 관례: +1 닫기, -1 열기
GRIPPER_OPEN = -1.0
GRIPPER_WIDTH_CLOSED = 0.02   # [m] 이 아래면 "닫힌 것" 으로 본다 (Panda 최대 ~0.08)

# ⚠️ robosuite 의 offscreen 렌더는 OpenGL 원점이 좌하단이라 관측 이미지가
#    상하 반전돼 나온다. LIBERO HDF5 데모도 **같은 관측을 그대로** 저장했으므로
#    둘 다 뒤집거나 둘 다 안 뒤집으면 일관성은 유지된다. 진짜로 중요한 것은
#    학습 입력과 rollout 입력의 방향이 같은 것뿐이다. 그래서 이 상수 하나로
#    두 경로를 동시에 통제한다 — 따로 두면 언젠가 반드시 어긋나고, 어긋나면
#    성공률이 통째로 0 이 되면서 원인 찾기가 매우 어렵다.
FLIP_RGB = True

# ── 합성 플랜트 상수 (--smoke 전용) ──────────────────────────────────────────
# dataset.py 와 evaluate.py 가 **같은 값**을 봐야 한다. 데모 action 이 플랜트
# 와 정합하지 않으면 "데모를 잘 따라해도 과제가 안 풀리는" 무의미한 smoke 가
# 된다(실제로 처음에 그렇게 짰다가 고쳤다).
SYNTH_GAIN = 0.9        # 명령 대비 실제 이동 비율
SYNTH_NOISE = 0.002     # [m] 스텝당 외란 — 100% 로 붙어버리지 않게
SYNTH_TOL = 0.05        # [m] 최종 EE 위치 허용 오차

# LIBERO 태스크 스위트. E0 는 Spatial + Object 두 개 = 20 태스크 × 50 데모 = 1000 데모
SUITES = ("libero_spatial", "libero_object")
N_TASKS_PER_SUITE = 10
N_DEMOS_PER_TASK = 50

ARMS = ("c0", "c6", "c8")
ARM_DESC = {
    "c0": "MONO         : RGB + proprio → action chunk",
    "c6": "SPLIT-ORACLE : 정답 목표 EE 포즈 + proprio + RGB → action chunk",
    "c8": "ORACLE-IK    : 정답 목표 EE 포즈 → OSC_POSE 직결, 학습 파라미터 0",
}
ARM_USES_GOAL = {"c0": False, "c6": True, "c8": True}


_LIBERO_INSTALL_HELP = """
LIBERO / robosuite 를 찾을 수 없습니다.

  E0 실데이터 실행에는 LIBERO 가 필요합니다 (Linux + NVIDIA GPU 권장):

    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO && pip install -r requirements.txt && pip install -e .
    pip install robosuite==1.4.1 h5py
    python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
    python benchmark_scripts/download_libero_datasets.py --datasets libero_object

  Mac 에서는 offscreen 렌더가 사실상 불가합니다. 코드 배선만 확인하려면:

    python dataset.py  --smoke
    python train.py    --smoke
    python evaluate.py --smoke
    bash  run_all.sh   --smoke
""".strip()


def _require_libero():
    """LIBERO import 를 지연시키고, 실패 시 설치법을 알려준다."""
    try:
        import h5py  # noqa: F401
        from libero.libero import benchmark  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"{_LIBERO_INSTALL_HELP}\n\n  (원인: {e})")


# ═════════════════════════════════════════════════════════════════════════════
# 회전 유틸 — scipy 없이 numpy 만으로. 3090 에 scipy 가 없을 수도 있다
# ═════════════════════════════════════════════════════════════════════════════


def rotvec_to_R(v: np.ndarray) -> np.ndarray:
    """축각(axis-angle) → 회전행렬. Rodrigues."""
    v = np.asarray(v, float).reshape(3)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    """
    회전행렬 → 축각. log map.

    θ≈π 근처에서 naive 공식은 0/0 이 되므로 대각 성분에서 축을 복구한다.
    OSC_POSE 명령은 축각이므로 이 경로가 C8 의 정확도를 직접 결정한다.
    """
    R = np.asarray(R, float).reshape(3, 3)
    c = (np.trace(R) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-7:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        # θ≈π: R = 2aaᵀ - I 이므로 대각에서 축 복구
        a = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
        i = int(np.argmax(a))
        if a[i] < 1e-8:
            return np.zeros(3)
        axis = np.array([R[0, i], R[1, i], R[2, i]]) / (2.0 * a[i])
        axis = axis / np.linalg.norm(axis)
        return axis * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * np.sin(th)))


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """
    쿼터니언 → 회전행렬. robosuite 관례인 **xyzw** 순서다.
    ⚠️ 확인 필요 — env obs 의 robot0_eef_quat 이 xyzw 인지 첫 실행 시 확인할 것
       (wxyz 로 잘못 읽으면 회전이 조용히 틀려서 C8 만 망가진다).
    """
    x, y, z, w = np.asarray(q, float).reshape(4)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def canonical_rgb(im: np.ndarray) -> np.ndarray:
    """
    이미지 방향을 하나의 관례로 통일한다. 학습 로더와 rollout 이 **반드시**
    이 함수를 통과해야 한다. FLIP_RGB 설명 참고.
    입력 (...,H,W,3) 의 H 축을 뒤집는다.
    """
    return im[..., ::-1, :, :] if FLIP_RGB else im


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """SVD 로 가장 가까운 SO(3) 원소를 찾는다. 6D 디코딩 후 안전장치."""
    U, _, Vt = np.linalg.svd(np.asarray(R, float).reshape(3, 3))
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


# ═════════════════════════════════════════════════════════════════════════════
# hindsight oracle — E0 의 심장
# ═════════════════════════════════════════════════════════════════════════════


def relative_goal(tgt_pos: np.ndarray, tgt_rot: np.ndarray, tgt_gw: float,
                  ref_pos: np.ndarray, ref_rot: np.ndarray,
                  ee_frame_rotation: bool = False) -> np.ndarray:
    """
    절대 목표 포즈 → ref 포즈 기준 델타 goal (GOAL_DIM,).

    **rollout 에서 이 함수가 학습/평가 정합성의 핵심이다.**
    학습 시 goal 은 "데모의 t 시점 포즈 기준 델타" 다. rollout 에서 로봇은
    데모 궤적을 벗어나므로, 데모의 델타를 그대로 먹이면 입력 분포가 어긋난다
    (로봇이 5cm 뒤처져 있는데 "앞으로 3cm" 라고 말하는 셈).

    올바른 oracle 은 "t+K 에 손이 있어야 할 **절대** 위치" 를 알고, 그것을
    **지금 실제 손 위치 기준 델타**로 다시 표현해 주는 것이다. 실제 Model 1
    도 현재 상태를 보고 델타를 내므로 이쪽이 배포 시나리오와도 일치한다.
    C6 와 C8 은 정확히 같은 이 벡터를 받는다 — 그래야 둘의 차이가 오직
    "학습된 소뇌 vs closed-form 컨트롤러" 가 된다.
    """
    dp = np.asarray(tgt_pos, float) - np.asarray(ref_pos, float)
    dR = (ref_rot.T @ tgt_rot) if ee_frame_rotation else (tgt_rot @ ref_rot.T)
    return np.concatenate([dp, R_to_6d(dR), [tgt_gw]]).astype(np.float32)


def hindsight_goal(ee_pos: np.ndarray,
                   ee_rot: np.ndarray,
                   grip_width: np.ndarray,
                   horizon: int = ORACLE_HORIZON,
                   ee_frame_rotation: bool = False) -> np.ndarray:
    """
    데모 전체 EE 궤적 → 시점별 oracle goal (T, GOAL_DIM).

    이것이 "완벽한 Model 1" 의 출력이다. 그냥 numpy 슬라이스다 —
    미래를 t+K 만큼 당겨오고, chunk 시작 시점 기준 델타로 바꾼다.

        dp(t)   = p[t+K] - p[t]                        (base frame, 3)
        dR6(t)  = R_to_6d( R[t+K] · R[t]ᵀ )            (base frame 상대회전, 6)
        grip(t) = width[t+K]                           (1)

    에피소드 끝에서는 인덱스를 T-1 로 clamp 한다 → 마지막 K 스텝의 goal 은
    "종료 포즈 유지" 가 되고, 이는 실제로 옳은 신호다(작업 완료 후 정지).

    Args:
        ee_pos:  (T,3)
        ee_rot:  (T,3,3)
        grip_width: (T,)  [m] 연속. binary 금지 — 암묵적 파지력 채널이기도 하다
        horizon: K
        ee_frame_rotation: True 면 상대회전을 EE 프레임(R[t]ᵀ·R[t+K])으로 준다.
            ⚠️ 확인 필요 — robosuite OSC_POSE 의 delta 축각이 base frame 해석
               이라고 보고 기본값을 base frame(False)으로 뒀다. LIBERO 컨트롤러
               config 를 실측해서 다르면 True 로 바꾸고 C8 을 재측정할 것.

    Returns:
        (T, GOAL_DIM)
    """
    T = len(ee_pos)
    idx = np.minimum(np.arange(T) + horizon, T - 1)     # ← 정확히 이 한 줄이 oracle 이다

    dp = ee_pos[idx] - ee_pos                            # (T,3)
    if ee_frame_rotation:
        dR = np.einsum("tji,tjk->tik", ee_rot, ee_rot[idx])   # R[t]ᵀ R[t+K]
    else:
        dR = np.einsum("tij,tkj->tik", ee_rot[idx], ee_rot)   # R[t+K] R[t]ᵀ
    dR6 = np.concatenate([dR[:, :, 0], dR[:, :, 1]], axis=1)  # (T,6)

    return np.concatenate([dp, dR6, grip_width[idx, None]], axis=1).astype(np.float32)


def oracle_to_osc_action(goal: np.ndarray,
                         cur_pos: np.ndarray,
                         cur_rot: np.ndarray,
                         chunk_start_pos: np.ndarray,
                         chunk_start_rot: np.ndarray,
                         kp: float = 1.0,
                         ee_frame_rotation: bool = False) -> np.ndarray:
    """
    C8 ORACLE-IK — oracle goal 을 OSC_POSE 명령으로 직결. **학습 파라미터 0개.**

    이 arm 이 존재하는 이유: C6 가 C0 를 이겨도, 그 이득이 "학습된 소뇌" 덕이
    아니라 "목표 포즈만 알면 closed-form 컨트롤러로 충분" 이라서일 수 있다.
    그렇다면 Model 2 를 학습시킬 이유가 사라진다 (Model 1 + IK 로 끝).
    C8 은 그 가능성을 정면으로 친다.

    구현은 pure pursuit 이다. goal 은 chunk 시작 시점 기준 델타이므로 먼저
    절대 목표로 복원한 뒤, **현재 실제 포즈** 와의 오차를 명령으로 바꾼다.
    (rollout 중 로봇은 데모 궤적에서 벗어나므로 이 재계산이 필수 — 이것이
    C8 을 open-loop replay 가 아닌 진짜 closed-loop 컨트롤러로 만든다.)

    Args:
        goal: (GOAL_DIM,) hindsight_goal 출력 한 행
        cur_pos/cur_rot: 지금 실제 EE 포즈
        chunk_start_pos/chunk_start_rot: goal 이 기준 삼은 포즈
        kp: pure-pursuit 이득. 1.0 = 남은 오차를 한 스텝에 다 명령(클리핑됨).
            overshoot 이 보이면 0.5~0.8 로 낮춘다.

    Returns:
        (7,) [-1,1] 로 정규화된 OSC_POSE 명령
    """
    dp, dR6, gw = goal[:3], goal[3:9], float(goal[9])

    # 1) 델타 goal → 절대 목표 포즈
    tgt_pos = chunk_start_pos + dp
    dR = orthonormalize(sixd_to_R(dR6))
    tgt_rot = chunk_start_rot @ dR if ee_frame_rotation else dR @ chunk_start_rot

    # 2) 현재 실제 포즈 대비 오차 → 명령
    err_p = kp * (tgt_pos - cur_pos)
    err_R = R_to_rotvec(tgt_rot @ cur_rot.T) * kp

    a_pos = np.clip(err_p / OSC_POS_MAX, -1.0, 1.0)
    a_rot = np.clip(err_R / OSC_ROT_MAX, -1.0, 1.0)
    a_grip = GRIPPER_CLOSE if gw < GRIPPER_WIDTH_CLOSED else GRIPPER_OPEN
    return np.concatenate([a_pos, a_rot, [a_grip]]).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 정규화
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class NormStats:
    """
    per-dim 평균/표준편차. action 과 goal 은 스케일이 제각각(mm 단위 병진 vs
    O(1) 회전 성분)이라 정규화 없이는 L1 손실이 병진에 지배된다.
    """
    action_mean: np.ndarray
    action_std: np.ndarray
    proprio_mean: np.ndarray
    proprio_std: np.ndarray
    goal_mean: np.ndarray
    goal_std: np.ndarray

    @staticmethod
    def _ms(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = x.mean(0).astype(np.float32)
        s = x.std(0).astype(np.float32)
        return m, np.maximum(s, 1e-3)      # 상수 차원에서 0 division 방지

    @classmethod
    def fit(cls, actions: np.ndarray, proprio: np.ndarray, goal: np.ndarray) -> "NormStats":
        am, asd = cls._ms(actions)
        pm, psd = cls._ms(proprio)
        gm, gsd = cls._ms(goal)
        return cls(am, asd, pm, psd, gm, gsd)

    def to_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(**{k: np.asarray(v, np.float32) for k, v in d.items()})


# ═════════════════════════════════════════════════════════════════════════════
# 에피소드 컨테이너
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Episode:
    """데모 1개. 이미지는 uint8 로 들고 있어야 500 데모가 RAM 에 들어간다."""
    task_id: int
    demo_id: int
    rgb: np.ndarray          # (T, N_CAMERAS, H, W, 3) uint8
    proprio: np.ndarray      # (T, PROPRIO_DIM) float32
    actions: np.ndarray      # (T, ACTION_DIM) float32
    goal: np.ndarray         # (T, GOAL_DIM) float32   ← hindsight oracle (학습용)
    ee_pos: np.ndarray       # (T, 3)     rollout 시 절대 목표 포즈 조회용
    ee_rot: np.ndarray       # (T, 3, 3)
    grip_width: np.ndarray   # (T,) [m]
    init_state: Optional[np.ndarray] = None   # LIBERO set_init_state 용
    suite: str = ""          # "libero_spatial" 등
    task_key: str = ""       # hdf5 파일 stem. ⚠️ rollout 시 bddl 매칭에 쓴다 —
                             #    hdf5 정렬 순서와 benchmark 의 task index 가
                             #    같다는 보장이 없으므로 **이름으로** 맞춘다

    def oracle_at(self, t: int, cur_pos: np.ndarray, cur_rot: np.ndarray,
                  horizon: int = ORACLE_HORIZON) -> np.ndarray:
        """
        rollout 시점 t 의 oracle goal. relative_goal() 설명 참고 —
        데모의 델타를 그대로 주는 게 아니라 **절대 목표를 현재 실제 포즈
        기준으로 다시 표현**한다.
        """
        k = min(t + horizon, len(self.ee_pos) - 1)
        return relative_goal(self.ee_pos[k], self.ee_rot[k], float(self.grip_width[k]),
                             cur_pos, cur_rot)

    def __len__(self) -> int:
        return len(self.actions)


class E0Dataset:
    """
    (episode, t) 평탄화 인덱스 + chunk 슬라이싱.

    torch Dataset 을 상속하지 않는다 — torch 없이도 --smoke 가 돌아야 하기
    때문이다. train.py 가 필요할 때 얇은 torch wrapper 로 감싼다.
    """

    def __init__(self, episodes: list[Episode], chunk_k: int = CHUNK_K,
                 norm: Optional[NormStats] = None):
        self.episodes = episodes
        self.chunk_k = chunk_k
        self.index: list[tuple[int, int]] = [
            (ei, t) for ei, ep in enumerate(episodes) for t in range(len(ep))
        ]
        self.norm = norm if norm is not None else self._fit_norm()

    def _fit_norm(self) -> NormStats:
        a = np.concatenate([e.actions for e in self.episodes], 0)
        p = np.concatenate([e.proprio for e in self.episodes], 0)
        g = np.concatenate([e.goal for e in self.episodes], 0)
        return NormStats.fit(a, p, g)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ei, t = self.index[i]
        ep = self.episodes[ei]
        if ep.rgb.size == 0 and len(ep.rgb.shape) < 5:
            raise RuntimeError(
                "이 데이터셋은 load_rgb=False 로 읽혔습니다 (eval split 기본값). "
                "학습에 쓰려면 split='train' 또는 load_rgb=True 로 다시 읽으십시오.")
        T, K = len(ep), self.chunk_k

        # action chunk: 끝에서는 마지막 action 을 반복해 padding.
        # pad_mask 로 손실에서 빼주므로 padding 값 자체는 학습에 영향이 없다.
        idx = np.minimum(np.arange(t, t + K), T - 1)
        act = (ep.actions[idx] - self.norm.action_mean) / self.norm.action_std
        pad = (np.arange(t, t + K) < T).astype(np.float32)

        return {
            "rgb": ep.rgb[t],                                    # (N_CAM,H,W,3) uint8
            "proprio": ((ep.proprio[t] - self.norm.proprio_mean)
                        / self.norm.proprio_std).astype(np.float32),
            "goal": ((ep.goal[t] - self.norm.goal_mean)
                     / self.norm.goal_std).astype(np.float32),
            "action": act.astype(np.float32),                    # (K, ACTION_DIM)
            "pad_mask": pad,                                     # (K,)
        }

    def batches(self, batch_size: int, rng: np.random.Generator,
                steps: Optional[int] = None) -> Iterator[dict]:
        """numpy 전용 무한 배치 이터레이터. --smoke 폴백 학습에서 쓴다."""
        n = 0
        while steps is None or n < steps:
            sel = rng.integers(0, len(self), size=batch_size)
            items = [self[int(i)] for i in sel]
            yield {k: np.stack([it[k] for it in items]) for k in items[0]}
            n += 1


# ═════════════════════════════════════════════════════════════════════════════
# LIBERO 어댑터
#
# ⚠️ 확인 필요 — HDF5 키 이름은 LIBERO 배포판마다 미묘하게 다르다. 아래는
#    후보를 순서대로 탐색하고 무엇을 찾았는지 출력한다. 회사 머신 첫 실행 시
#    `python dataset.py --inspect --data <경로>` 로 실제 키를 먼저 확인할 것.
# ═════════════════════════════════════════════════════════════════════════════

_RGB_KEYS = [("agentview_rgb", "eye_in_hand_rgb"),
             ("agentview_image", "robot0_eye_in_hand_image")]
_JOINT_KEYS = ["joint_states", "robot0_joint_pos"]
_GRIP_KEYS = ["gripper_states", "robot0_gripper_qpos"]
_EE_POS_KEYS = ["ee_pos", "robot0_eef_pos"]
_EE_ORI_KEYS = ["ee_ori", "robot0_eef_ori"]        # 축각 3-vector 로 가정
_EE_STATES_KEYS = ["ee_states"]                    # (T,6) = pos(3) + 축각(3)


def _pick(grp, keys: list[str]):
    for k in keys:
        if k in grp:
            return k
    return None


def _ee_from_obs(obs) -> tuple[np.ndarray, np.ndarray]:
    """
    obs 그룹에서 EE 포즈(위치, 회전행렬)를 꺼낸다.

    LIBERO 는 버전에 따라 ee_pos/ee_ori 를 따로 주거나 ee_states(T,6) 로 준다.
    둘 다 없으면 즉시 실패시킨다 — oracle 없이 E0 는 성립하지 않으므로
    조용히 대충 넘어가면 안 된다.
    """
    kp, ko = _pick(obs, _EE_POS_KEYS), _pick(obs, _EE_ORI_KEYS)
    if kp and ko:
        pos = np.asarray(obs[kp], np.float64)
        ori = np.asarray(obs[ko], np.float64)
    else:
        ks = _pick(obs, _EE_STATES_KEYS)
        if ks is None:
            raise KeyError(
                f"EE 포즈 키를 찾을 수 없습니다. 사용 가능한 키: {list(obs.keys())}\n"
                f"  → dataset.py 의 _EE_POS_KEYS/_EE_STATES_KEYS 에 추가하십시오."
            )
        s = np.asarray(obs[ks], np.float64)
        pos, ori = s[:, :3], s[:, 3:6]
    rot = np.stack([rotvec_to_R(v) for v in ori])
    return pos.astype(np.float32), rot.astype(np.float32)


def load_libero_episodes(data_root: str,
                         suites: tuple[str, ...] = SUITES,
                         demo_slice: slice = slice(0, 25),
                         horizon: int = ORACLE_HORIZON,
                         max_tasks: Optional[int] = None,
                         load_rgb: bool = True,
                         verbose: bool = True) -> list[Episode]:
    """
    LIBERO HDF5 → Episode 리스트.

    Args:
        data_root: libero 데이터셋 루트 (하위에 libero_spatial/ 등)
        demo_slice: train 은 slice(0,25), eval 은 slice(25,50) 이 기본.
            이유는 파일 상단 설계 결정 3) 참고.

    이미지 방향은 canonical_rgb() 하나로 통제한다 (FLIP_RGB 설명 참고).
    """
    _require_libero()
    import h5py

    episodes: list[Episode] = []
    task_id = 0
    for suite in suites:
        sdir = os.path.join(data_root, suite)
        if not os.path.isdir(sdir):
            raise SystemExit(f"[dataset] 스위트 디렉토리 없음: {sdir}\n{_LIBERO_INSTALL_HELP}")
        files = sorted(f for f in os.listdir(sdir) if f.endswith(".hdf5"))
        if max_tasks:
            files = files[:max_tasks]
        for fn in files:
            with h5py.File(os.path.join(sdir, fn), "r") as f:
                data = f["data"]
                names = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
                for dn in names[demo_slice]:
                    d = data[dn]
                    obs = d["obs"]
                    rgb_pair = next((p for p in _RGB_KEYS if p[0] in obs), None)
                    if rgb_pair is None:
                        raise KeyError(f"RGB 키 없음. 사용 가능: {list(obs.keys())}")
                    if load_rgb:
                        imgs = [canonical_rgb(np.asarray(obs[k])) for k in rgb_pair]
                        rgb = np.ascontiguousarray(
                            np.stack(imgs, axis=1)).astype(np.uint8)  # (T,2,H,W,3)
                    else:
                        # eval 은 rollout 관측을 env 에서 받으므로 데모 이미지가
                        # 전혀 필요 없다. 500 데모 × ~150 스텝 × 2 cam × 128²×3
                        # ≈ 7GB 를 안 읽는다 (로딩 시간도 그만큼 절약).
                        rgb = np.empty((len(obs[rgb_pair[0]]), 0), np.uint8)

                    jq = np.asarray(obs[_pick(obs, _JOINT_KEYS)], np.float32)
                    gq = np.asarray(obs[_pick(obs, _GRIP_KEYS)], np.float32)
                    ee_pos, ee_rot = _ee_from_obs(obs)

                    # Panda gripper qpos 는 finger 2개 → 폭 = 두 손가락 거리
                    gw = np.abs(gq[:, 0] - gq[:, 1]) if gq.shape[1] >= 2 else gq[:, 0]

                    proprio = np.concatenate(
                        [jq, gq[:, :2], ee_pos,
                         np.stack([R_to_6d(R) for R in ee_rot])], axis=1
                    ).astype(np.float32)

                    acts = np.asarray(d["actions"], np.float32)
                    T = min(len(acts), len(rgb), len(proprio))
                    ep = Episode(
                        task_id=task_id, demo_id=int(dn.split("_")[1]),
                        rgb=rgb[:T], proprio=proprio[:T], actions=acts[:T],
                        goal=hindsight_goal(ee_pos[:T], ee_rot[:T], gw[:T], horizon),
                        ee_pos=ee_pos[:T], ee_rot=ee_rot[:T], grip_width=gw[:T],
                        init_state=(np.asarray(d["states"][0])
                                    if "states" in d else None),
                        suite=suite, task_key=fn[:-5],   # ".hdf5" 제거
                    )
                    episodes.append(ep)
            task_id += 1
        if verbose:
            print(f"[dataset] {suite}: {len(files)} 태스크 로드")
    if verbose:
        tot = sum(len(e) for e in episodes)
        print(f"[dataset] 에피소드 {len(episodes)}개, transition {tot}개")
    return episodes


def inspect_hdf5(data_root: str, suite: str = SUITES[0]) -> None:
    """실제 HDF5 키 구조를 출력한다. 어댑터를 고치기 전에 먼저 이걸 돌릴 것."""
    _require_libero()
    import h5py
    sdir = os.path.join(data_root, suite)
    fn = sorted(f for f in os.listdir(sdir) if f.endswith(".hdf5"))[0]
    with h5py.File(os.path.join(sdir, fn), "r") as f:
        print(f"[inspect] {fn}")
        d = f["data"]["demo_0"]
        for k in d.keys():
            if k == "obs":
                print("  obs/")
                for ok in d["obs"].keys():
                    print(f"    {ok:28s} {d['obs'][ok].shape} {d['obs'][ok].dtype}")
            else:
                print(f"  {k:30s} {getattr(d[k], 'shape', '?')}")


# ═════════════════════════════════════════════════════════════════════════════
# 합성 데이터 — --smoke 전용
#
# LIBERO 없이 데이터→학습→평가→판정 전체 배선을 검증한다. 3090 에서 clone
# 하자마자 "코드가 맞는지" 를 몇 초 만에 확인할 수 있어야 한다.
#
# 일부러 **goal 이 실제로 정보를 갖도록** 만들었다: 정답 action 이 goal 의
# 선형함수 + 이미지 의존 항으로 구성돼 있어서 C6 가 C0 를 이기는 게 정상이다.
# 즉 smoke 는 배선뿐 아니라 판정 로직이 말이 되는 출력을 내는지도 본다.
# (이 숫자들은 **합성이므로 어떤 과학적 의미도 없다.**)
# ═════════════════════════════════════════════════════════════════════════════


def make_synthetic_episodes(n_episodes: int = 24,
                            ep_len: int = 40,
                            img_size: int = 32,
                            n_tasks: int = 4,
                            seed: int = 0) -> list[Episode]:
    """합성 에피소드. 이미지는 32×32 로 줄여 CPU 에서도 몇 초에 돈다."""
    rng = np.random.default_rng(seed)
    eps: list[Episode] = []
    for i in range(n_episodes):
        T = ep_len
        tid = i % n_tasks
        # EE 궤적: 태스크별 목표점으로 향하는 부드러운 곡선
        tgt = np.array([0.4, 0.0, 0.2]) + 0.15 * rng.standard_normal(3)
        p0 = np.array([0.1, 0.0, 0.4]) + 0.05 * rng.standard_normal(3)
        s = np.linspace(0, 1, T)[:, None] ** 1.5
        ee_pos = (p0 + (tgt - p0) * s).astype(np.float32)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        ee_rot = np.stack([rotvec_to_R(axis * (0.6 * float(a))) for a in s[:, 0]]).astype(np.float32)
        gw = np.clip(0.08 - 0.08 * np.clip((s[:, 0] - 0.6) / 0.4, 0, 1), 0, 0.08).astype(np.float32)

        goal = hindsight_goal(ee_pos, ee_rot, gw, ORACLE_HORIZON)

        rgb = rng.integers(0, 255, (T, N_CAMERAS, img_size, img_size, 3), dtype=np.uint8)
        # 이미지에 태스크 식별 패치를 심는다 (vision 경로가 실제로 쓰이도록)
        rgb[:, :, : 4 + 2 * tid, :4] = 250

        jq = rng.standard_normal((T, 7)).astype(np.float32) * 0.1
        gq = np.stack([gw / 2, -gw / 2], 1).astype(np.float32)
        proprio = np.concatenate(
            [jq, gq, ee_pos, np.stack([R_to_6d(R) for R in ee_rot])], 1).astype(np.float32)

        # 정답 action = **이 궤적을 실제로 만들어내는 OSC_POSE 명령**.
        # 임의의 선형사상으로 만들면 데모를 완벽히 모방해도 로봇이 목표에
        # 도달하지 못해서 smoke 가 아무것도 검증하지 못한다.
        acts = np.zeros((T, ACTION_DIM), np.float32)
        dpos = np.diff(ee_pos, axis=0, append=ee_pos[-1:])
        acts[:, :3] = np.clip(dpos / (OSC_POS_MAX * SYNTH_GAIN), -1, 1)
        dR = np.einsum("tij,tkj->tik", np.roll(ee_rot, -1, 0), ee_rot)
        dR[-1] = np.eye(3)
        acts[:, 3:6] = np.clip(np.stack([R_to_rotvec(R) for R in dR])
                               / (OSC_ROT_MAX * SYNTH_GAIN), -1, 1)
        acts[:, 6] = np.where(gw < GRIPPER_WIDTH_CLOSED, GRIPPER_CLOSE, GRIPPER_OPEN)

        eps.append(Episode(task_id=tid, demo_id=i, rgb=rgb, proprio=proprio,
                           actions=acts, goal=goal, ee_pos=ee_pos, ee_rot=ee_rot,
                           grip_width=gw))
    return eps


def build_dataset(smoke: bool = False,
                  data_root: Optional[str] = None,
                  split: str = "train",
                  train_demos: int = 25,
                  seed: int = 0,
                  norm: Optional[NormStats] = None,
                  **kw) -> E0Dataset:
    """
    train.py / evaluate.py 의 단일 진입점.

    Args:
        split: "train" → demos[:train_demos], "eval" → demos[train_demos:50]
        norm: eval 은 반드시 train 의 통계를 물려받아야 한다. 안 그러면
              평가 시점에 입력 분포가 미묘하게 달라져 성공률이 왜곡된다.
    """
    if smoke:
        eps = make_synthetic_episodes(
            n_episodes=kw.pop("n_episodes", 48 if split == "train" else 30),
            seed=seed + (0 if split == "train" else 777))
    else:
        if not data_root:
            raise SystemExit("[dataset] --data <LIBERO 데이터 루트> 가 필요합니다.")
        sl = slice(0, train_demos) if split == "train" else slice(train_demos, N_DEMOS_PER_TASK)
        kw.setdefault("load_rgb", split == "train")
        eps = load_libero_episodes(data_root, demo_slice=sl, **kw)
    return E0Dataset(eps, norm=norm)


# ═════════════════════════════════════════════════════════════════════════════


def _self_test() -> int:
    """oracle 수학이 맞는지 왕복 검증. --smoke 의 첫 관문."""
    print("── dataset self-test ──")
    rng = np.random.default_rng(0)
    T = 50
    pos = np.cumsum(rng.standard_normal((T, 3)) * 0.01, 0).astype(np.float32)
    rot = np.stack([rotvec_to_R(rng.standard_normal(3) * 0.2) for _ in range(T)]).astype(np.float32)
    gw = np.linspace(0.08, 0.0, T).astype(np.float32)

    g = hindsight_goal(pos, rot, gw, ORACLE_HORIZON)
    assert g.shape == (T, GOAL_DIM), g.shape

    # 1) goal 델타를 다시 절대 포즈로 복원하면 t+K 의 실제 포즈와 같아야 한다
    t = 5
    k = min(t + ORACLE_HORIZON, T - 1)
    rec_p = pos[t] + g[t, :3]
    rec_R = orthonormalize(sixd_to_R(g[t, 3:9])) @ rot[t]
    e_p = float(np.linalg.norm(rec_p - pos[k]))
    e_R = float(np.linalg.norm(rec_R - rot[k]))
    print(f"  oracle 왕복 오차   : 위치 {e_p:.2e} m, 회전 {e_R:.2e}")
    assert e_p < 1e-5 and e_R < 1e-5, "oracle 델타↔절대 변환 불일치"

    # 2) 축각 log/exp 왕복
    v = rng.standard_normal(3) * 1.2
    e_rv = float(np.linalg.norm(R_to_rotvec(rotvec_to_R(v)) - v))
    print(f"  축각 log/exp 왕복  : {e_rv:.2e}")
    assert e_rv < 1e-8

    # 3) C8: 현재 포즈 == chunk 시작 포즈면 명령이 goal 방향과 일치해야 한다
    a = oracle_to_osc_action(g[t], pos[t], rot[t], pos[t], rot[t])
    want = np.clip(g[t, :3] / OSC_POS_MAX, -1, 1)
    assert np.allclose(a[:3], want, atol=1e-6), (a[:3], want)
    assert a.shape == (ACTION_DIM,)
    print(f"  C8 명령 예시       : {np.round(a, 3)}")

    # 4) rollout 시 oracle 재표현: ref == 데모 포즈면 학습 goal 과 같아야 한다
    ep = make_synthetic_episodes(n_episodes=1, seed=3)[0]
    t = 7
    same = ep.oracle_at(t, ep.ee_pos[t], ep.ee_rot[t])
    e_same = float(np.abs(same - ep.goal[t]).max())
    print(f"  oracle_at 정합성   : {e_same:.2e}  (ref=데모포즈일 때 학습 goal 과 일치)")
    assert e_same < 1e-5, "학습 goal 과 rollout goal 정의가 어긋났다"

    # 로봇이 5cm 뒤처져 있으면 델타는 그만큼 더 커져야 한다 (분포 정합의 핵심)
    off = np.array([0.05, 0.0, 0.0], np.float32)
    drift = ep.oracle_at(t, ep.ee_pos[t] - off, ep.ee_rot[t])
    assert np.allclose(drift[:3] - same[:3], off, atol=1e-6), drift[:3] - same[:3]
    print(f"  drift 보정         : OK (+{off[0]*100:.0f}cm 뒤처짐 → 델타 +{off[0]*100:.0f}cm)")

    # 5) quat/rotvec 상호 검증
    v = rng.standard_normal(3) * 0.7
    Rv = rotvec_to_R(v)
    th = np.linalg.norm(v); ax = v / th
    q = np.concatenate([ax * np.sin(th / 2), [np.cos(th / 2)]])   # xyzw
    assert np.allclose(quat_to_R(q), Rv, atol=1e-9), "quat_to_R 규약 불일치(xyzw?)"
    print("  quat(xyzw)↔rotvec  : OK")

    # 6) 데이터셋 배선
    ds = build_dataset(smoke=True, split="train")
    b = next(ds.batches(4, np.random.default_rng(0), steps=1))
    print(f"  합성 데이터셋      : {len(ds)} transition, "
          f"rgb{b['rgb'].shape} goal{b['goal'].shape} action{b['action'].shape}")
    assert b["action"].shape == (4, CHUNK_K, ACTION_DIM)
    assert b["goal"].shape == (4, GOAL_DIM)
    assert b["proprio"].shape == (4, PROPRIO_DIM)
    print("── dataset self-test 통과 ──")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="E0 데이터 계층")
    ap.add_argument("--smoke", action="store_true", help="LIBERO 없이 배선 검증")
    ap.add_argument("--data", default=None, help="LIBERO 데이터 루트")
    ap.add_argument("--inspect", action="store_true", help="HDF5 키 구조만 출력")
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--train-demos", type=int, default=25)
    a = ap.parse_args()

    if a.inspect:
        if not a.data:
            raise SystemExit("--inspect 에는 --data 가 필요합니다.")
        inspect_hdf5(a.data)
        return 0
    if a.smoke or not a.data:
        return _self_test()

    ds = build_dataset(data_root=a.data, split=a.split, train_demos=a.train_demos)
    print(f"[dataset] {a.split}: {len(ds.episodes)} 에피소드 / {len(ds)} transition")
    print(json.dumps({k: np.round(v, 4).tolist()
                      for k, v in [("action_mean", ds.norm.action_mean),
                                   ("action_std", ds.norm.action_std)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
