"""
Galbot G1 기구학 — FK, psi-구속 IK, arm angle (psi)

이 모듈이 존재하는 이유
───────────────────────
분석 결과 두 가지가 확정됐다:

1. G1 팔은 canonical S–R–S (Spherical–Revolute–Spherical) 7-DoF 다.
   어깨 j1,j2,j3 축이 한 점에 집중, 팔꿈치 j4, 손목 j5,j6,j7 집중.
   → `(EE pose, psi)` 가 주어지면 7개 관절각의 **closed-form 해**가 존재한다.

2. 6-DoF 포즈만으로는 7-DoF 형상이 결정되지 않는다 (1차원 self-motion manifold).
   손목 중심을 고정하면 팔꿈치가 어깨→손목 축을 중심으로 원을 그린다.
   → 인터페이스에 arm angle psi 스칼라 1개를 추가하면 완전 결정된다.

따라서 이 모듈은:
  - 학습된 Model 2가 이겨야 할 **학습 파라미터 0 인 결정론적 baseline** (E0의 C8 arm)
  - psi 라벨을 텔레옵 기록에서 공짜로 뽑는 도구 (FK 1회, 사람 라벨링 0)
  - L2 회귀가 self-motion manifold 위에서 붕괴하는 크기를 정량화하는 도구

⚠️ 용어 주의 — 여기 구현된 IK 는 **closed-form 이 아니라 psi-구속 DLS** 다.
   S–R–S 구조에서 (포즈, psi) 로부터 해석해가 **존재한다**는 것은 참인 수학적
   사실이고 README/GLOSSARY 가 말하는 것도 그것이지만, 이 모듈은 그 해석해를
   구현하지 않았다 (`docs/PLAN.md` 의 M5 는 아직 미완료 항목이다).
   실제 구현은 7×7 (6 포즈 + 1 psi) DLS 반복이다 — `ik()` 참조.
   학습 모델 대비 3자릿수 빠르지만 "해석해"는 아니다.

⚠️ **이 모듈의 모든 값은 URDF 자기일관성만 보장한다.**
   URDF 자체가 실기체와 다르면 test_kinematics.py 가 6/6 통과해도 아무것도
   보증하지 않는다. 외부 ground truth 대조(SDK FK)는 아직 안 했다 —
   `docs/RUNBOOK.md` §2-1 의 probe 절차 참조.

모든 상수는 공식 Apache-2.0 `galbot_one_golf_description` URDF에서 추출했다
(robot/assets/g1_joints_raw.json). 하드코딩된 값은 없다.

좌표계
──────
  torso_base_link  : 양팔과 목이 갈라지는 단일 강체. 이것이 우리의 기준 프레임.
                     다리/허리 5관절과 평면 베이스 3-DoF 는 전부 이 아래에 있어
                     head→arm_base 상대 변환에서 소거된다.
  {L,R}_arm_base   : torso 에서 fixed 로 갈라진 어깨 마운트
  head_link2       : 목 pan/tilt 뒤의 헤드 프레임 (카메라 마운트)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_JOINTS_JSON = os.path.join(_HERE, "assets", "g1_joints_raw.json")


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) 유틸
# ─────────────────────────────────────────────────────────────────────────────


def rpy_to_R(rpy: Iterable[float]) -> np.ndarray:
    """URDF rpy (fixed-axis X-Y-Z, 즉 R = Rz @ Ry @ Rx)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T_from(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def T_rot(axis: np.ndarray, theta: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_axis(axis, theta)
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def pose_error(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """(병진 오차 m, 회전 오차 rad)."""
    dp = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
    Rd = Ta[:3, :3].T @ Tb[:3, :3]
    c = (np.trace(Rd) - 1.0) / 2.0
    return dp, float(math.acos(max(-1.0, min(1.0, c))))


# ─────────────────────────────────────────────────────────────────────────────
# URDF 체인
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Link:
    """체인의 한 단계: 고정 변환 + (있으면) 회전 축."""
    name: str
    T_fixed: np.ndarray
    axis: Optional[np.ndarray]          # None 이면 고정 조인트
    lower: float = -math.inf
    upper: float = math.inf


class Chain:
    def __init__(self, links: list[Link]):
        self.links = links
        self.dof_idx = [i for i, l in enumerate(links) if l.axis is not None]

    @property
    def n_dof(self) -> int:
        return len(self.dof_idx)

    @property
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.links[i].lower for i in self.dof_idx])
        hi = np.array([self.links[i].upper for i in self.dof_idx])
        return lo, hi

    def fk(self, q: np.ndarray, upto: Optional[int] = None) -> np.ndarray:
        """base → tip 변환. upto 는 self.links 인덱스(배타)."""
        T = np.eye(4)
        qi = 0
        end = len(self.links) if upto is None else upto
        for i, l in enumerate(self.links[:end]):
            T = T @ l.T_fixed
            if l.axis is not None:
                T = T @ T_rot(l.axis, float(q[qi]))
                qi += 1
        return T

    def fk_all(self, q: np.ndarray) -> list[np.ndarray]:
        """각 링크 뒤의 누적 변환 목록."""
        out, T, qi = [], np.eye(4), 0
        for l in self.links:
            T = T @ l.T_fixed
            if l.axis is not None:
                T = T @ T_rot(l.axis, float(q[qi]))
                qi += 1
            out.append(T.copy())
        return out


def _load_joints() -> dict:
    with open(_JOINTS_JSON) as f:
        return json.load(f)


def _path(joints: dict, target_link: str, root: str) -> list[tuple[str, dict]]:
    by_child = {v["child"]: (n, v) for n, v in joints.items()}
    out, cur = [], target_link
    while cur != root:
        if cur not in by_child:
            raise ValueError(f"{root} 에서 {target_link} 로 가는 경로 없음 (막힌 곳: {cur})")
        n, v = by_child[cur]
        out.append((n, v))
        cur = v["parent"]
    return list(reversed(out))


def build_chain(target_link: str, root: str = "torso_base_link",
                joints: Optional[dict] = None) -> Chain:
    joints = joints or _load_joints()
    links = []
    for n, v in _path(joints, target_link, root):
        axis = None
        lo, hi = -math.inf, math.inf
        if v["type"] in ("revolute", "continuous") and v.get("axis"):
            axis = np.asarray(v["axis"], dtype=float)
            if np.linalg.norm(axis) > 0:
                L = v.get("limit") or {}
                lo = L.get("lower", -math.inf)
                hi = L.get("upper", math.inf)
            else:
                axis = None
        links.append(Link(n, T_from(v["xyz"], v["rpy"]), axis, lo, hi))
    return Chain(links)


# ─────────────────────────────────────────────────────────────────────────────
# tip 프레임 판별표
#
# 왜 필요한가
# ───────────
# 우리 FK 의 tip 은 `*_gripper_tcp_link` 로 **골라 잡은 것**이다. SDK 의
# `forward_kinematics_by_state` 가 어느 프레임을 돌려주는지는 미확인이고,
# 팔 끝단 후보가 4개인데 전부 `fixed` joint 로 연결돼 있다 — 즉 **관절각과
# 무관한 상수 오차**만 낸다. 이게 왜 위험하냐면, 상수 오차는 학습이 그대로
# 흡수해버려서 조용히 틀린 채로 수렴하기 때문이다.
#
# 다행히 상수라는 점이 그대로 진단 도구가 된다. 아래 표는 URDF 에서 계산한
# 후보 간 상대 오프셋이고, 값이 서로 충분히 떨어져 있어 **SDK FK 표본 하나만
# 있어도** 어느 프레임인지 유일하게 판별된다.
#
#   기준: left_arm_link7 (팔 마지막 회전 관절의 자식)
#
#   후보                                누적 |t|      link7 기준 R
#   ─────────────────────────────────────────────────────────────────────
#   arm_end_effector_mount_link         109.26 mm    diag(-1, +1, -1)
#   gripper_flange_link                 109.26 mm    mount 에서 Rx(+90°)
#   gripper_base_link                   115.76 mm    flange 에서 +6.5mm
#   gripper_tcp_link  ← 우리 기본값     255.72 mm    diag(-1, -1, +1)
#
#   쌍별 오프셋 (tcp 를 기준으로 본 잔차)
#   ─────────────────────────────────────────────────────────────────────
#   tcp ↔ mount                         146.46 mm    180°
#   tcp ↔ flange                        146.46 mm     90°
#   tcp ↔ gripper_base                  139.96 mm     90°
#   flange ↔ gripper_base                 6.50 mm      0°
#
# 사용법: SDK FK 와 우리 FK(tip=tcp)의 잔차를 재서 `identify_tip_frame()` 에
# 넣으면 어느 프레임인지 알려준다. 잔차가 표에 없으면 tip 문제가 아니라
# **기준(root) 프레임 문제**일 가능성이 크다 — 그때는 다리 관절만 흔들어
# EE 포즈가 변하는지 보면 된다 (변하면 base/world 기준, 안 변하면 torso 기준).
# ─────────────────────────────────────────────────────────────────────────────

TIP_CANDIDATES = ("arm_end_effector_mount_link", "gripper_flange_link",
                  "gripper_base_link", "gripper_tcp_link")

# tcp 기준 잔차 (mm, deg) → 실제 SDK tip 프레임 이름
TIP_SIGNATURE_FROM_TCP = {
    (0.00, 0.0): "gripper_tcp_link",
    (146.46, 180.0): "arm_end_effector_mount_link",
    (146.46, 90.0): "gripper_flange_link",
    (139.96, 90.0): "gripper_base_link",
}


def identify_tip_frame(pos_err_mm: float, rot_err_deg: float,
                       tol_mm: float = 3.0, tol_deg: float = 10.0) -> Optional[str]:
    """SDK FK 와 우리 FK(tip=tcp)의 잔차로 SDK 의 tip 프레임을 판별한다.

    **최근접 매칭 + 모호성 검사.** 임계 이내를 순회하며 첫 항목을 고르면 안 된다 —
    flange(146.46mm)와 gripper_base(139.96mm)의 간격이 6.5mm 뿐이라, 임계를
    그보다 크게 잡으면 둘이 섞이고 dict 순서에 따라 답이 갈린다.

    반환:
      프레임 이름 = 유일하게 판별됨
      None        = 표에 없거나 두 후보 사이에서 모호함.
                    tip 문제가 아니라 **root(기준) 프레임 문제**를 의심할 것 —
                    다리 관절만 흔들어 EE 포즈가 변하는지 보면 갈린다.
    """
    def cost(sig: tuple[float, float]) -> float:
        dmm, ddeg = sig
        # 위치/회전을 각자의 허용치로 정규화해 합산
        return (abs(pos_err_mm - dmm) / tol_mm) + (abs(rot_err_deg - ddeg) / tol_deg)

    ranked = sorted(TIP_SIGNATURE_FROM_TCP.items(), key=lambda kv: cost(kv[0]))
    (best_sig, best_name), (_, runner_name) = ranked[0], ranked[1]
    if cost(best_sig) > 1.0:
        return None                                  # 어느 후보와도 안 맞음
    if cost(ranked[1][0]) - cost(best_sig) < 0.5:
        return None                                  # 1·2위가 붙어 있어 모호
    del runner_name
    return best_name


# ─────────────────────────────────────────────────────────────────────────────
# G1 팔
# ─────────────────────────────────────────────────────────────────────────────


class G1Arm:
    """
    한쪽 팔의 FK / S–R–S IK / arm angle.

    S–R–S 구조 (URDF 에서 검증):
      어깨 : joint1,2,3 축이 한 점에 집중  (residual 0.000 mm)
      팔꿈치: joint4
      손목 : joint5,6,7 축이 한 점에 집중
      상완 d_se = 0.350 m, 전완 d_ew = 0.360 m
    """

    def __init__(self, side: str = "left", tip: Optional[str] = None):
        assert side in ("left", "right")
        self.side = side
        self.tip = tip or f"{side}_gripper_tcp_link"
        self.joints = _load_joints()
        self.chain = build_chain(self.tip, joints=self.joints)

        # 7개 팔 관절만의 인덱스
        self._arm_dof = [i for i in self.chain.dof_idx
                         if f"{side}_arm_joint" in self.chain.links[i].name]
        if len(self._arm_dof) != 7:
            raise ValueError(f"팔 관절 7개를 기대했으나 {len(self._arm_dof)}개 발견")

        self._geom = self._measure_geometry()

    # ── 기하 (URDF 에서 측정, 하드코딩 없음) ────────────────────────────────
    def _measure_geometry(self) -> dict:
        """영 자세에서 어깨/팔꿈치/손목 중심과 링크 길이를 측정."""
        q0 = np.zeros(self.chain.n_dof)
        Ts = self.chain.fk_all(q0)
        idx = {self.chain.links[i].name: i for i in range(len(self.chain.links))}

        def origin_of(jname: str) -> np.ndarray:
            return Ts[idx[jname]][:3, 3].copy()

        s = origin_of(f"{self.side}_arm_joint1")   # 어깨 중심 (j1,2,3 집중점)
        e = origin_of(f"{self.side}_arm_joint4")   # 팔꿈치
        w = origin_of(f"{self.side}_arm_joint5")   # 손목 중심 (j5,6,7 집중점)
        return {
            "shoulder": s, "elbow": e, "wrist": w,
            "d_se": float(np.linalg.norm(e - s)),
            "d_ew": float(np.linalg.norm(w - e)),
        }

    @property
    def d_se(self) -> float:
        return self._geom["d_se"]

    @property
    def d_ew(self) -> float:
        return self._geom["d_ew"]

    @property
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.chain.links[i].lower for i in self._arm_dof])
        hi = np.array([self.chain.links[i].upper for i in self._arm_dof])
        return lo, hi

    # ── FK ──────────────────────────────────────────────────────────────────
    def fk(self, q7: np.ndarray) -> np.ndarray:
        """팔 관절 7개 → torso_base_link 기준 TCP 변환."""
        q7 = np.asarray(q7, dtype=float).reshape(7)
        full = np.zeros(self.chain.n_dof)
        for k, i in enumerate(self._arm_dof):
            full[self.chain.dof_idx.index(i)] = q7[k]
        return self.chain.fk(full)

    def joint_centers(self, q7: np.ndarray) -> dict:
        """현재 자세의 어깨/팔꿈치/손목 중심."""
        q7 = np.asarray(q7, dtype=float).reshape(7)
        full = np.zeros(self.chain.n_dof)
        for k, i in enumerate(self._arm_dof):
            full[self.chain.dof_idx.index(i)] = q7[k]
        Ts = self.chain.fk_all(full)
        idx = {self.chain.links[i].name: i for i in range(len(self.chain.links))}
        return {
            "shoulder": Ts[idx[f"{self.side}_arm_joint1"]][:3, 3].copy(),
            "elbow":    Ts[idx[f"{self.side}_arm_joint4"]][:3, 3].copy(),
            "wrist":    Ts[idx[f"{self.side}_arm_joint5"]][:3, 3].copy(),
        }

    # ── arm angle (psi) ─────────────────────────────────────────────────────
    def arm_angle(self, q7: np.ndarray) -> float:
        """
        기록된 관절각에서 arm angle psi 를 추출한다 (FK 1회, 사람 라벨링 0).

        정의: 어깨→손목 축을 중심으로 한 팔꿈치의 signed angle.
        기준 평면(psi=0)은 어깨→손목 직선과 **torso 의 수직축**을 포함하는 평면.
        torso 에 고정되므로 목 운동에 불변이고 토르소 높이에 걸쳐 일관된다.
        """
        c = self.joint_centers(q7)
        s, e, w = c["shoulder"], c["elbow"], c["wrist"]

        axis = w - s
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return float("nan")           # 어깨-손목 일치: psi 미정의
        axis = axis / n

        # 팔꿈치의 축 직교 성분
        v = e - s
        v_perp = v - np.dot(v, axis) * axis
        if np.linalg.norm(v_perp) < 1e-9:
            return float("nan")           # 완전 신전: 특이
        v_perp = v_perp / np.linalg.norm(v_perp)

        # 기준: torso 수직축(+Z)을 축에 직교 투영
        ref = np.array([0.0, 0.0, 1.0])
        ref_perp = ref - np.dot(ref, axis) * axis
        if np.linalg.norm(ref_perp) < 1e-6:
            # 어깨→손목이 수직축과 평행 — 대체 기준 사용
            ref = np.array([1.0, 0.0, 0.0])
            ref_perp = ref - np.dot(ref, axis) * axis
        ref_perp = ref_perp / np.linalg.norm(ref_perp)

        cross = np.cross(ref_perp, v_perp)
        return float(math.atan2(np.dot(cross, axis), np.dot(ref_perp, v_perp)))

    # ── IK ──────────────────────────────────────────────────────────────────
    def ik(self, T_target: np.ndarray, psi: float,
           q_seed: Optional[np.ndarray] = None,
           tol_pos: float = 1e-5, tol_rot: float = 1e-5,
           max_iter: int = 200) -> Optional[np.ndarray]:
        """
        `(목표 TCP 포즈, psi)` → 7 관절각.

        구현 노트
        ─────────
        엄밀한 Shimizu 형태의 해석해는 이 URDF 의 축 정렬·오프셋을 하나하나
        대수적으로 풀어야 한다. 여기서는 **psi 를 hard constraint 로 강제한
        damped least-squares** 로 같은 답에 도달한다:

          - 6-DoF 포즈 오차 + psi 오차 = 7개 방정식, 7개 미지수 → 정사각.
          - 따라서 해가 국소적으로 **유일**하다 (미결정이 아님).
          - q_seed 로 시작하므로 branch 는 연속성으로 고정된다
            (argmin||q - q_prev|| 와 동등한 효과).

        해석해 대비 장점: URDF 축 관례에 자동으로 맞고 관절 한계를 그대로 쓴다.
        단점: 반복법이라 마이크로초가 아니라 수십 마이크로초.
        측정된 실제 비용은 tests(T6) 참조 — 학습 모델 대비 3자릿수 빠르다.
        ⚠️ SDK `inverse_kinematics_by_state` 와는 **미비교**다. SDK IK 는
        sampling + seed 기반이라 psi 를 지정받지 않고 redundancy 해가
        비결정론적이므로, µs 만 나란히 적으면 오독을 부른다.

        실패 시 None 을 반환한다 (**타입 있는 infeasibility 신호** — 회귀
        네트워크가 구조적으로 만들 수 없는 것).
        """
        lo, hi = self.limits
        q = np.array(q_seed, dtype=float).reshape(7) if q_seed is not None \
            else np.clip(np.zeros(7), lo, hi)

        lam = 1e-3
        for _ in range(max_iter):
            T_cur = self.fk(q)
            dp = T_target[:3, 3] - T_cur[:3, 3]
            Rd = T_target[:3, :3] @ T_cur[:3, :3].T
            ang = math.acos(max(-1.0, min(1.0, (np.trace(Rd) - 1.0) / 2.0)))
            if ang < 1e-12:
                dw = np.zeros(3)
            else:
                dw = ang / (2 * math.sin(ang)) * np.array([
                    Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]])

            psi_cur = self.arm_angle(q)
            dpsi = 0.0 if math.isnan(psi_cur) else _wrap(psi - psi_cur)

            err = np.concatenate([dp, dw, [dpsi]])
            if (np.linalg.norm(dp) < tol_pos and abs(ang) < tol_rot
                    and abs(dpsi) < 1e-4):
                return np.clip(q, lo, hi)

            Jt = self._jacobian7(q)                 # 7x7 (6 pose + 1 psi)
            JJt = Jt @ Jt.T + lam * lam * np.eye(7)
            dq = Jt.T @ np.linalg.solve(JJt, err)

            step = float(np.linalg.norm(dq))
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, lo, hi)

        return None

    def _jacobian7(self, q: np.ndarray) -> np.ndarray:
        """[포즈 6 ; psi 1] × 관절 7 자코비안 (유한 차분)."""
        eps = 1e-6
        T0 = self.fk(q)
        psi0 = self.arm_angle(q)
        J = np.zeros((7, 7))
        for k in range(7):
            qp = q.copy()
            qp[k] += eps
            T1 = self.fk(qp)
            J[0:3, k] = (T1[:3, 3] - T0[:3, 3]) / eps
            Rd = T1[:3, :3] @ T0[:3, :3].T
            J[3:6, k] = np.array([Rd[2, 1] - Rd[1, 2],
                                  Rd[0, 2] - Rd[2, 0],
                                  Rd[1, 0] - Rd[0, 1]]) / (2 * eps)
            psi1 = self.arm_angle(qp)
            J[6, k] = 0.0 if (math.isnan(psi1) or math.isnan(psi0)) \
                else _wrap(psi1 - psi0) / eps
        return J


def _wrap(a: float) -> float:
    """각도를 (-pi, pi] 로."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ─────────────────────────────────────────────────────────────────────────────
# 헤드 → 팔 베이스 변환
# ─────────────────────────────────────────────────────────────────────────────


class G1Head:
    """
    목 pan/tilt 와 head→arm_base 변환.

    핵심 사실 (URDF 로 검증): head→arm_base 는 **목 관절 2개에만** 의존한다.
    다리/허리 5관절과 평면 베이스 3-DoF 는 목과 양팔의 공통 조상이므로
    상대 변환에서 대수적으로 소거된다.

    즉 Model 2 가 헤드 프레임 목표를 팔 관절로 바꾸려면
    **스칼라 2개(head_joint1, head_joint2)만 있으면 충분하다.**
    현재 제안된 입력(팔 관절 14 + 이미지 + F/T)에는 그 2개가 없다.
    """

    def __init__(self, tip: str = "head_end_effector_mount_link"):
        self.joints = _load_joints()
        self.chain = build_chain(tip, joints=self.joints)

    def T_torso_head(self, q_head: np.ndarray) -> np.ndarray:
        """목 관절 → torso_base_link 기준 헤드 프레임."""
        return self.chain.fk(np.asarray(q_head, dtype=float).reshape(-1))

    def head_to_armbase(self, q_head: np.ndarray, arm: G1Arm) -> np.ndarray:
        """
        헤드 프레임 → 팔 베이스 프레임 변환.

        이것이 인터페이스에서 해야 할 계산이다: Model 1 이 헤드 프레임으로
        목표를 내면, **촬영 시각의** q_head 로 이 변환을 합성해 Model 2 에는
        torso_base_link 기준 목표를 넘긴다. 4x4 곱셈 1회 (마이크로초).
        """
        T_th = self.T_torso_head(q_head)
        return se3_inv(T_th)     # head → torso. arm 은 torso 기준으로 푼다.

    def target_head_to_torso(self, T_head_target: np.ndarray,
                             q_head: np.ndarray) -> np.ndarray:
        """헤드 프레임 목표 포즈 → torso_base_link 기준 목표 포즈."""
        return self.T_torso_head(q_head) @ T_head_target


# ─────────────────────────────────────────────────────────────────────────────


def summary() -> str:
    L, R = G1Arm("left"), G1Arm("right")
    H = G1Head()
    lo, hi = L.limits
    g = L._geom
    lines = [
        "Galbot G1 기구학 (공식 URDF 에서 측정)",
        f"  상완 d_se        : {L.d_se:.4f} m",
        f"  전완 d_ew        : {L.d_ew:.4f} m",
        f"  팔 길이 합        : {L.d_se + L.d_ew:.4f} m",
        f"  어깨 중심 (L)     : {np.round(g['shoulder'], 5)}",
        f"  어깨 간격         : {np.linalg.norm(L._geom['shoulder'] - R._geom['shoulder']):.4f} m",
        f"  팔 관절 하한      : {np.round(lo, 4)}",
        f"  팔 관절 상한      : {np.round(hi, 4)}",
        f"  목 DoF            : {H.chain.n_dof}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
